#!/usr/bin/env python3
"""Compact, dependency-free web crawler and archive.

The scraper is designed for information gathering rather than mirroring pages:
it keeps a ranked extractive summary, source metadata, and selected links, then
packs all records into one LZMA-compressed binary archive.  The format favors
storage efficiency over random access or preserving the original HTML.

Example:
    python compact_web_scraper.py scrape https://en.wikipedia.org/wiki/Genocide \
        --max-pages 1000 --max-depth 2 --output history.lca
    python compact_web_scraper.py query history.lca "genocide history"
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import lzma
import re
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, deque
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


MAGIC = b"LCA\x01\x00\x00\x00\x00"
FORMAT_VERSION = 1
MAX_RESPONSE_BYTES = 8_000_000
DEFAULT_MAX_BYTES = 10_000_000
DEFAULT_SUMMARY_CHARS = 900
TOKEN_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)?", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "me", "might", "more", "most", "my", "of", "on", "or",
    "our", "ours", "she", "should", "so", "some", "than", "that", "the", "their",
    "theirs", "them", "then", "there", "these", "they", "this", "those", "to",
    "was", "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your", "yours",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tokens(value: str) -> list[str]:
    return [
        match.group(0).lower().replace("’", "'")
        for match in TOKEN_RE.finditer(value)
        if match.group(0).lower() not in STOPWORDS and len(match.group(0)) > 1
    ]


def normalize_url(value: str, base: str | None = None) -> str | None:
    joined = urllib.parse.urljoin(base or "", value.strip())
    parsed = urllib.parse.urlparse(joined)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    if parsed.username or parsed.password:
        return None
    host = parsed.netloc.lower()
    path = parsed.path or "/"
    return urllib.parse.urlunparse((parsed.scheme.lower(), host, path, "", parsed.query, ""))


class PageText(HTMLParser):
    """Extract visible text while dropping navigation scripts and styling."""

    ignored = {
        "script", "style", "noscript", "svg", "template", "head", "nav", "header",
        "footer", "aside", "form",
    }
    blocks = {
        "address", "article", "aside", "blockquote", "br", "div", "dl", "dt", "dd",
        "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3",
        "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre",
        "section", "table", "td", "th", "tr", "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.ignored:
            self.depth += 1
        elif self.depth == 0 and tag in self.blocks:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.depth == 0 and tag.lower() in self.blocks:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.ignored and self.depth:
            self.depth -= 1
        elif self.depth == 0 and tag in self.blocks:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        value = html.unescape("".join(self.parts))
        value = re.sub(r"[\t\r ]+", " ", value)
        value = re.sub(r"\n[ ]+", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        lines: list[str] = []
        seen_lines: set[str] = set()
        for raw_line in value.splitlines():
            line = raw_line.strip()
            if not line:
                if lines and lines[-1] != "":
                    lines.append("")
                continue
            key = re.sub(r"\s+", " ", line).casefold()
            # Captions, menus, and duplicated mobile/desktop fragments often
            # appear more than once. Keep the first occurrence of substantial
            # repeated lines while preserving short structural labels.
            if len(line) >= 28 and key in seen_lines:
                continue
            seen_lines.add(key)
            lines.append(line)
        return "\n".join(lines).strip()


class PageLinks(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        values = {key.lower(): (value or "") for key, value in attrs}
        target = normalize_url(values.get("href", ""), self.base_url)
        if target and target not in self.links:
            self.links.append(target)


def page_title(raw_html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    value = html.unescape(re.sub(r"<[^>]+>", "", match.group(1)))
    return re.sub(r"\s+", " ", value).strip()[:300]


def fetch_page(url: str, timeout: float) -> dict[str, Any] | None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "compact-web-scraper/0.1 (local read-only archive)",
            "Accept": "text/html,text/plain,application/xhtml+xml;q=0.9",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                return None
            final_url = normalize_url(response.geturl()) or url
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
        return None
    try:
        decoded = raw.decode(charset, errors="replace")
    except LookupError:
        decoded = raw.decode("utf-8", errors="replace")
    title = page_title(decoded) if content_type != "text/plain" else ""
    links: list[str] = []
    if content_type != "text/plain":
        link_parser = PageLinks(final_url)
        link_parser.feed(decoded)
        link_parser.close()
        links = link_parser.links
        text_parser = PageText()
        text_parser.feed(decoded)
        text_parser.close()
        readable = text_parser.text()
    else:
        readable = decoded.strip()
    if len(readable) < 40:
        return None
    return {
        "url": final_url,
        "title": title,
        "text": readable,
        "links": links,
        "retrieved_at": now_iso(),
    }


def sentence_candidates(text: str) -> list[str]:
    result: list[str] = []
    for raw in SENTENCE_RE.split(text):
        value = re.sub(r"\s+", " ", raw).strip(" \t\r\n-•")
        term_count = len(tokens(value))
        if value.endswith("?") or value.startswith(("What ", "How ", "Why ", "Investigate ")):
            continue
        if 45 <= len(value) <= 1_200 and term_count >= 6:
            result.append(value)
    return result


def summarize(text: str, max_chars: int) -> tuple[str, float]:
    """Extract high-information sentences without inventing new text."""
    raw_candidates = sentence_candidates(text)
    candidates: list[str] = []
    seen: set[str] = set()
    seen_term_sets: list[set[str]] = []
    for sentence in raw_candidates:
        key = fingerprint(sentence)
        sentence_terms = set(tokens(sentence))
        if key in seen or any(
            len(sentence_terms & previous) / max(1, len(sentence_terms | previous)) >= 0.86
            for previous in seen_term_sets
        ):
            continue
        seen.add(key)
        seen_term_sets.append(sentence_terms)
        candidates.append(sentence)
    if not candidates:
        return re.sub(r"\s+", " ", text).strip()[:max_chars], 0.0
    frequency = Counter(token for sentence in candidates for token in tokens(sentence))
    scored: list[tuple[float, int, str]] = []
    total = max(1, len(candidates) - 1)
    for index, sentence in enumerate(candidates):
        sentence_terms = tokens(sentence)
        if not sentence_terms:
            continue
        centrality = sum(frequency[term] for term in set(sentence_terms)) / len(set(sentence_terms))
        specificity = len(set(sentence_terms)) / len(sentence_terms)
        position = 1.0 - (index / total) * 0.15
        score = centrality * (0.8 + 0.2 * specificity) * position
        scored.append((score, index, sentence))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected: list[tuple[int, str]] = []
    used = 0
    for _, index, sentence in scored:
        separator = 1 if selected else 0
        if used + len(sentence) + separator > max_chars:
            continue
        selected.append((index, sentence))
        used += len(sentence) + separator
    if not selected:
        selected = [(scored[0][1], scored[0][2][:max_chars])]
    selected.sort(key=lambda item: item[0])
    result = " ".join(sentence for _, sentence in selected)
    return result[:max_chars], min(1.0, len(selected) / max(1, min(12, len(candidates))))


def fingerprint(sentence: str) -> str:
    terms = sorted(set(tokens(sentence)))
    return hashlib.blake2s(" ".join(terms).encode("utf-8"), digest_size=10).hexdigest()


def same_host(url: str, seed_hosts: set[str]) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower()
    return host in seed_hosts


def robots_allowed(url: str, cache: dict[str, Any], user_agent: str, timeout: float) -> bool:
    """Best-effort robots check; failures leave the page available."""
    parsed = urllib.parse.urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin not in cache:
        robots_url = origin + "/robots.txt"
        try:
            request = urllib.request.Request(robots_url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                cache[origin] = response.read(512_000).decode("utf-8", errors="replace")
        except Exception:
            cache[origin] = None
    rules = cache[origin]
    if rules is None:
        return True
    # Avoid an extra dependency on urllib.robotparser while handling the common
    # disallow form. Unknown or malformed directives are ignored.
    active = False
    for line in rules.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = [piece.strip() for piece in line.split(":", 1)]
        key, value = key.lower(), value.strip()
        if key == "user-agent":
            active = value in {"*", user_agent.lower()}
        elif key == "disallow" and active and value:
            path = urllib.parse.urlparse(url).path or "/"
            if path.startswith(value):
                return False
    return True


def crawl(seeds: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    normalized_seeds = [normalize_url(seed) for seed in seeds]
    normalized_seeds = [seed for seed in normalized_seeds if seed]
    if not normalized_seeds:
        raise ValueError("Provide at least one valid http(s) seed URL")
    seed_hosts = {urllib.parse.urlparse(seed).netloc.lower() for seed in normalized_seeds}
    pending: deque[tuple[str, int]] = deque((seed, 0) for seed in normalized_seeds)
    visited: set[str] = set()
    records: list[dict[str, Any]] = []
    seen_sentences: set[str] = set()
    seen_sentence_terms: list[set[str]] = []
    robots_cache: dict[str, Any] = {}
    while pending and len(records) < args.max_pages:
        url, depth = pending.popleft()
        if url in visited:
            continue
        visited.add(url)
        if not args.follow_external and not same_host(url, seed_hosts):
            continue
        if args.respect_robots and not robots_allowed(url, robots_cache, "compact-web-scraper", args.timeout):
            continue
        page = fetch_page(url, args.timeout)
        if page is None:
            continue
        summary, quality = summarize(page["text"], args.summary_chars)
        unique_summary_sentences: list[str] = []
        for sentence in sentence_candidates(summary):
            key = fingerprint(sentence)
            sentence_terms = set(tokens(sentence))
            if key in seen_sentences or any(
                len(sentence_terms & previous) / max(1, len(sentence_terms | previous)) >= 0.86
                for previous in seen_sentence_terms
            ):
                continue
            seen_sentences.add(key)
            seen_sentence_terms.append(sentence_terms)
            unique_summary_sentences.append(sentence)
        summary = " ".join(unique_summary_sentences)[: args.summary_chars] or summary[: args.summary_chars]
        records.append(
            {
                "url": page["url"],
                "title": page["title"],
                "summary": summary,
                "depth": depth,
                "quality": round(quality, 3),
                "retrieved_at": page["retrieved_at"],
            }
        )
        if depth < args.max_depth:
            for link in page["links"]:
                if args.follow_external or same_host(link, seed_hosts):
                    pending.append((link, depth + 1))
        if args.delay:
            time.sleep(args.delay)
    return records


def serialize_records(records: list[dict[str, Any]]) -> bytes:
    lines = [json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def container_bytes(records: list[dict[str, Any]], query: str = "") -> bytes:
    payload = serialize_records(records)
    compressed = lzma.compress(payload, format=lzma.FORMAT_XZ, preset=9)
    header = json.dumps(
        {
            "format": FORMAT_VERSION,
            "codec": "xz/lzma2",
            "records": len(records),
            "raw_bytes": len(payload),
            "compressed_bytes": len(compressed),
            "created_at": now_iso(),
            "query": query,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return MAGIC + struct.pack("<I", len(header)) + header + compressed


def fit_to_budget(records: list[dict[str, Any]], max_bytes: int, query: str) -> tuple[list[dict[str, Any]], bytes]:
    """Trim summaries, then records, until the complete container fits."""
    working = [dict(record) for record in records]
    for summary_chars in (DEFAULT_SUMMARY_CHARS, 700, 500, 350, 250, 180, 120):
        for record in working:
            record["summary"] = str(record.get("summary", ""))[:summary_chars]
        packed = container_bytes(working, query=query)
        if len(packed) <= max_bytes:
            return working, packed
    while working:
        packed = container_bytes(working, query=query)
        if len(packed) <= max_bytes:
            return working, packed
        working.pop()
    packed = container_bytes([], query=query)
    if len(packed) > max_bytes:
        raise ValueError("The requested archive budget is too small for its header")
    return working, packed


def write_archive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with open(fd, "wb", closefd=True) as handle:
            handle.write(data)
        Path(temp_name).replace(path)
    finally:
        try:
            Path(temp_name).unlink()
        except FileNotFoundError:
            pass


def read_archive(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = path.read_bytes()
    if not data.startswith(MAGIC) or len(data) < len(MAGIC) + 4:
        raise ValueError("Not a compact web archive")
    offset = len(MAGIC)
    header_len = struct.unpack("<I", data[offset : offset + 4])[0]
    offset += 4
    header = json.loads(data[offset : offset + header_len].decode("utf-8"))
    offset += header_len
    payload = lzma.decompress(data[offset:], format=lzma.FORMAT_XZ).decode("utf-8")
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    return header, records


def command_scrape(args: argparse.Namespace) -> int:
    try:
        records = crawl(args.seeds, args)
        fitted, packed = fit_to_budget(records, args.max_bytes, " ".join(args.seeds))
        output = Path(args.output)
        write_archive(output, packed)
    except (OSError, ValueError) as exc:
        print(f"Scrape failed: {exc}", file=sys.stderr)
        return 1
    raw_bytes = len(serialize_records(fitted))
    ratio = raw_bytes / max(1, len(packed))
    print(f"Archived {len(fitted):,} page(s) to {output.resolve()}")
    print(f"Raw summaries: {raw_bytes:,} bytes; archive: {len(packed):,} bytes; ratio: {ratio:.1f}x")
    if len(fitted) < len(records):
        print(f"Budget trimmed {len(records) - len(fitted):,} page(s)")
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    try:
        header, records = read_archive(Path(args.archive))
    except (OSError, ValueError, lzma.LZMAError, json.JSONDecodeError) as exc:
        print(f"Could not read archive: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(header, indent=2, ensure_ascii=False))
    print(f"Loaded records: {len(records):,}")
    for record in records[: args.show]:
        print(f"\n{record.get('title') or '(untitled)'}\n{record.get('url')}")
        print(record.get("summary", ""))
    return 0


def command_query(args: argparse.Namespace) -> int:
    try:
        _, records = read_archive(Path(args.archive))
    except (OSError, ValueError, lzma.LZMAError, json.JSONDecodeError) as exc:
        print(f"Could not read archive: {exc}", file=sys.stderr)
        return 1
    wanted = set(tokens(args.query))
    ranked: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        haystack = f"{record.get('title', '')} {record.get('summary', '')}"
        terms = set(tokens(haystack))
        overlap = len(wanted & terms) / max(1, len(wanted)) if wanted else 0.0
        if overlap:
            ranked.append((overlap, record))
    ranked.sort(key=lambda item: (-item[0], item[1].get("title", "")))
    print(f"Query: {args.query}")
    if not ranked:
        print("No matching summaries.")
        return 0
    for index, (score, record) in enumerate(ranked[: args.limit], start=1):
        print(f"\n{index}. [{score:.3f}] {record.get('title') or '(untitled)'}")
        print(record.get("url"))
        print(record.get("summary", ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compact dependency-free web scraper")
    sub = parser.add_subparsers(dest="command", required=True)

    scrape = sub.add_parser("scrape", help="crawl seeds and write one compressed archive")
    scrape.add_argument("seeds", nargs="+", help="one or more http(s) seed URLs")
    scrape.add_argument("--output", default="compact.lca")
    scrape.add_argument("--max-pages", type=int, default=100)
    scrape.add_argument("--max-depth", type=int, default=1)
    scrape.add_argument("--summary-chars", type=int, default=DEFAULT_SUMMARY_CHARS)
    scrape.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    scrape.add_argument("--timeout", type=float, default=20.0)
    scrape.add_argument("--delay", type=float, default=0.25)
    scrape.add_argument("--follow-external", action="store_true", help="allow links off seed domains")
    scrape.add_argument(
        "--ignore-robots",
        dest="respect_robots",
        action="store_false",
        help="do not consult robots.txt (read-only caps still apply)",
    )
    scrape.set_defaults(func=command_scrape, respect_robots=True)

    inspect = sub.add_parser("inspect", help="show archive metadata and sample records")
    inspect.add_argument("archive")
    inspect.add_argument("--show", type=int, default=3)
    inspect.set_defaults(func=command_inspect)

    query = sub.add_parser("query", help="search summaries inside an archive")
    query.add_argument("archive")
    query.add_argument("query")
    query.add_argument("--limit", type=int, default=10)
    query.set_defaults(func=command_query)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

