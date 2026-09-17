#!/usr/bin/env python3
"""A tiny, dependency-free local web-review and belief-journal prototype.

This is deliberately an evidence engine rather than a pretrained language model.
It can read user-selected public pages, keep a local corpus, retrieve evidence,
and record a working position for a topic.  It contains no moral or personality
policy.  The only built-in limits are technical: it does not execute commands,
send requests that change remote state, or modify its own source code.

Run ``python local_opinion_ai.py --help`` for the CLI.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
import os
import re
import sys
import tempfile
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable


APP_VERSION = "0.1.0"
DEFAULT_HOME = ".local_opinion_ai"
MAX_DOWNLOAD_BYTES = 5_000_000
MAX_STORED_CHARS = 500_000

TOKEN_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)?", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")

# These words are used only for retrieval and phrase selection.  They encode no
# moral position and do not filter the text that is stored.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "can", "could", "did", "do", "does", "for", "from", "had",
    "has", "have", "he", "her", "hers", "him", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "me", "might", "more",
    "most", "my", "of", "on", "or", "our", "ours", "she", "should", "so",
    "some", "than", "that", "the", "their", "theirs", "them", "then", "there",
    "these", "they", "this", "those", "to", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you",
    "your", "yours",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def app_home(value: str | None) -> Path:
    """Return the data directory without relying on a broad filesystem path."""
    chosen = value or os.environ.get("LOCAL_OPINION_HOME") or DEFAULT_HOME
    path = Path(chosen).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    """Write atomically so an interrupted run does not corrupt the journal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def corpus_path(home: Path) -> Path:
    return home / "corpus.json"


def beliefs_path(home: Path) -> Path:
    return home / "beliefs.json"


def load_corpus(home: Path) -> list[dict[str, Any]]:
    data = read_json(corpus_path(home), [])
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a list in {corpus_path(home)}")
    return data


def load_beliefs(home: Path) -> list[dict[str, Any]]:
    data = read_json(beliefs_path(home), [])
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a list in {beliefs_path(home)}")
    return data


class VisibleText(HTMLParser):
    """Small HTML-to-text extractor from the Python standard library."""

    ignored_tags = {"script", "style", "noscript", "svg", "template", "head"}
    block_tags = {
        "address", "article", "aside", "blockquote", "br", "div", "dl", "dt",
        "dd", "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2",
        "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol",
        "p", "pre", "section", "table", "td", "th", "tr", "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.ignored_tags:
            self._ignored_depth += 1
        elif self._ignored_depth == 0 and tag in self.block_tags:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._ignored_depth == 0 and tag.lower() in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1
        elif self._ignored_depth == 0 and tag in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        value = html.unescape("".join(self.parts))
        value = re.sub(r"[\t\r ]+", " ", value)
        value = re.sub(r"\n[ ]+", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()


def html_to_text(raw: str) -> str:
    parser = VisibleText()
    parser.feed(raw)
    parser.close()
    return parser.text()


class SearchResults(HTMLParser):
    """Parse DuckDuckGo's simple HTML result page without an SDK."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._active_href: str | None = None
        self._active_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        values = {key.lower(): (value or "") for key, value in attrs}
        classes = set(values.get("class", "").split())
        rels = set(values.get("rel", "").split())
        if "result__a" in classes or ("nofollow" in rels and values.get("href")):
            self._active_href = values.get("href") or ""
            self._active_text = []

    def handle_data(self, data: str) -> None:
        if self._active_href is not None:
            self._active_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._active_href is None:
            return
        title = re.sub(r"\s+", " ", "".join(self._active_text)).strip()
        href = html.unescape(self._active_href)
        if href.startswith("//"):
            href = "https:" + href
        parsed = urllib.parse.urlparse(href)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
            target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
            if target:
                href = urllib.parse.unquote(target)
        if title and href.startswith(("http://", "https://")):
            self.results.append({"title": title, "url": href})
        self._active_href = None
        self._active_text = []


class SeznamResults(HTMLParser):
    """Parse the server-rendered result headings from Seznam search."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._active_href: str | None = None
        self._active_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        values = {key.lower(): (value or "") for key, value in attrs}
        if values.get("data-e-a") == "heading":
            self._active_href = values.get("href") or ""
            self._active_text = []

    def handle_data(self, data: str) -> None:
        if self._active_href is not None:
            self._active_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._active_href is None:
            return
        title = re.sub(r"\s+", " ", "".join(self._active_text)).strip()
        href = html.unescape(self._active_href)
        if title and href.startswith(("http://", "https://")):
            self.results.append({"title": title, "url": href})
        self._active_href = None
        self._active_text = []


def parse_bing_results(page: str) -> list[dict[str, str]]:
    """Extract result links from Bing's server-rendered result list."""
    found: list[dict[str, str]] = []
    blocks = re.findall(
        r"<li[^>]*class=[\"'][^\"']*b_algo[^\"']*[\"'][^>]*>.*?</li>",
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block in blocks:
        match = re.search(
            r"<h2[^>]*>\s*<a[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            continue
        href = html.unescape(match.group(1))
        parsed_href = urllib.parse.urlparse(href)
        if parsed_href.netloc.lower().endswith("bing.com") and parsed_href.path.startswith("/ck/a"):
            encoded_target = urllib.parse.parse_qs(parsed_href.query).get("u", [""])[0]
            if encoded_target.startswith("a1"):
                try:
                    padded = encoded_target[2:] + "=" * (-len(encoded_target[2:]) % 4)
                    href = base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
                except (ValueError, UnicodeError):
                    pass
        title = html.unescape(re.sub(r"<[^>]+>", "", match.group(2)))
        title = re.sub(r"\s+", " ", title).strip()
        if href.startswith(("http://", "https://")) and title:
            found.append({"title": title, "url": href})
    return found


def search_web(query: str, limit: int = 8, timeout: float = 20.0) -> list[dict[str, str]]:
    """Find public URLs through server-rendered search pages.

    This is best-effort discovery, not a claim to index the whole internet. The
    returned pages still need to be fetched and evaluated by the local journal.
    The three server-rendered providers are queried concurrently.  This avoids
    waiting for a slow provider before trying the next one while keeping the
    dependency-free prototype usable when one provider is unavailable.
    """
    encoded = urllib.parse.quote_plus(query)
    providers = [
        (f"https://search.seznam.cz/?q={encoded}", "seznam"),
        (f"https://www.bing.com/search?q={encoded}&setlang=en-us&cc=us", "bing"),
        (f"https://html.duckduckgo.com/html/?q={encoded}", "duckduckgo"),
    ]
    def fetch_provider(endpoint: str, provider: str) -> list[dict[str, str]]:
        request = urllib.request.Request(
            endpoint,
            headers={
                "User-Agent": "local-opinion-ai/0.1 (local read-only research prototype)",
                "Accept": "text/html",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read(2_000_000)
                charset = response.headers.get_content_charset() or "utf-8"
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise RuntimeError(f"{provider}: {type(exc).__name__}") from exc
        try:
            page = raw.decode(charset, errors="replace")
        except LookupError:
            page = raw.decode("utf-8", errors="replace")
        if provider == "seznam":
            parser = SeznamResults()
            parser.feed(page)
            parser.close()
            parsed_results = parser.results
        elif provider == "bing":
            parsed_results = parse_bing_results(page)
        else:
            parser = SearchResults()
            parser.feed(page)
            parser.close()
            parsed_results = parser.results
        return [
            {"title": str(result.get("title", "")), "url": str(result.get("url", "")), "provider": provider}
            for result in parsed_results
            if result.get("title") and result.get("url")
        ]

    provider_results: dict[str, list[dict[str, str]]] = {}
    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=len(providers), thread_name_prefix="search") as executor:
        futures = {
            executor.submit(fetch_provider, endpoint, provider): provider
            for endpoint, provider in providers
        }
        for future in as_completed(futures):
            provider = futures[future]
            try:
                provider_results[provider] = future.result()
            except Exception as exc:
                errors.append(exc)

    safe_limit = max(1, int(limit))
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    tracking = {"fbclid", "gclid", "mc_cid", "mc_eid", "msclkid", "ref", "ref_"}
    # Preserve the stable provider preference while merging the fastest
    # responses.  URL keys discard common analytics parameters.
    for _, provider in providers:
        for result in provider_results.get(provider, []):
            parsed = urllib.parse.urlparse(result["url"])
            query_parts = [
                (key, value)
                for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
                if key.casefold() not in tracking and not key.casefold().startswith("utm_")
            ]
            query_parts.sort()
            key = urllib.parse.urlunparse((parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path or "/", "", urllib.parse.urlencode(query_parts), ""))
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(result)
            if len(unique) >= safe_limit:
                return unique
    if unique:
        return unique
    if errors:
        raise RuntimeError(f"Could not search the web: {errors[0]}") from errors[0]
    return []


def fetch_url(url: str, timeout: float = 20.0) -> tuple[str, str, str]:
    """Fetch a page without requiring third-party packages.

    Returns (final_url, content_type, decoded_text).  Only GET is used, and the
    caller controls each URL explicitly; there is no site-wide crawler.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Only explicit http:// and https:// URLs are supported")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded credentials are not accepted")

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "local-opinion-ai/0.1 (local read-only research prototype)",
            "Accept": "text/html,text/plain,application/xhtml+xml;q=0.9,*/*;q=0.1",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(raw) > MAX_DOWNLOAD_BYTES:
                raise ValueError(f"Response is larger than {MAX_DOWNLOAD_BYTES:,} bytes")
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            final_url = response.geturl()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} while fetching {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not fetch {url}: {exc.reason}") from exc

    try:
        decoded = raw.decode(charset, errors="replace")
    except LookupError:
        decoded = raw.decode("utf-8", errors="replace")
    if content_type in {"text/html", "application/xhtml+xml"}:
        decoded = html_to_text(decoded)
    else:
        decoded = decoded.strip()
    return final_url, content_type, decoded[:MAX_STORED_CHARS]


def fetch_local(path: Path) -> tuple[str, str, str]:
    if not path.is_file():
        raise ValueError(f"Not a regular file: {path}")
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in {".html", ".htm", ".xhtml"}:
        raw = html_to_text(raw)
        content_type = "text/html"
    else:
        content_type = "text/plain"
    return path.resolve().as_uri(), content_type, raw.strip()[:MAX_STORED_CHARS]


def word_tokens(value: str, *, keep_stopwords: bool = False) -> list[str]:
    words = [match.group(0).lower().replace("’", "'") for match in TOKEN_RE.finditer(value)]
    if keep_stopwords:
        return words
    return [word for word in words if word not in STOPWORDS and len(word) > 1]


def sentence_list(text: str) -> list[str]:
    sentences: list[str] = []
    for item in SENTENCE_RE.split(text):
        cleaned = re.sub(r"\s+", " ", item).strip(" \t\r\n-•")
        if 35 <= len(cleaned) <= 800 and len(word_tokens(cleaned)) >= 5:
            sentences.append(cleaned)
    return sentences


def source_name(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc or parsed.path


def ingest_one(home: Path, location: str, timeout: float) -> dict[str, Any]:
    if re.match(r"^https?://", location, flags=re.IGNORECASE):
        final_url, content_type, text = fetch_url(location, timeout=timeout)
    else:
        final_url, content_type, text = fetch_local(Path(location))
    if not text:
        raise ValueError(f"No readable text found at {location}")

    digest = hashlib.sha256((final_url + "\n" + text).encode("utf-8")).hexdigest()
    corpus = load_corpus(home)
    for document in corpus:
        if document.get("id") == digest[:16]:
            return document

    document = {
        "id": digest[:16],
        "url": final_url,
        "source": source_name(final_url),
        "content_type": content_type,
        "retrieved_at": now_iso(),
        "characters": len(text),
        "text": text,
    }
    corpus.append(document)
    write_json(corpus_path(home), corpus)
    return document


def sentence_score(sentence: str, query_terms: set[str], document_terms: set[str]) -> float:
    terms = set(word_tokens(sentence))
    if not terms:
        return 0.0
    overlap = len(terms & query_terms) / max(1, len(query_terms)) if query_terms else 0.0
    density = len(terms & document_terms) / max(1, len(terms))
    return (overlap * 0.75) + (density * 0.25)


def search_sentences(corpus: Iterable[dict[str, Any]], query: str, limit: int = 8) -> list[dict[str, Any]]:
    query_terms = set(word_tokens(query))
    results: list[dict[str, Any]] = []
    for document in corpus:
        doc_sentences = sentence_list(str(document.get("text", "")))
        document_terms = set(word_tokens(str(document.get("text", ""))))
        for sentence in doc_sentences:
            sentence_terms = set(word_tokens(sentence))
            if query_terms and not (sentence_terms & query_terms):
                continue
            score = sentence_score(sentence, query_terms, document_terms)
            if query_terms and score <= 0:
                continue
            results.append(
                {
                    "score": score,
                    "document_id": document.get("id"),
                    "source": document.get("source"),
                    "url": document.get("url"),
                    "sentence": sentence,
                }
            )
    results.sort(key=lambda item: (-item["score"], item["source"] or "", item["sentence"]))
    return results[: max(1, limit)]


def working_position(corpus: list[dict[str, Any]], topic: str, limit: int) -> dict[str, Any]:
    evidence = search_sentences(corpus, topic, limit=max(12, limit * 3))
    if not evidence:
        return {
            "topic": topic,
            "position": None,
            "coverage": 0.0,
            "sources": [],
            "evidence": [],
            "method": "lexical centrality over locally stored text",
        }

    # Select the sentence with the strongest query overlap, preferring a
    # substantive sentence when several candidates tie. This is intentionally
    # extractive: it never invents a conclusion or quietly supplies a worldview.
    chosen = max(
        evidence,
        key=lambda item: (
            item["score"],
            min(60, len(word_tokens(item["sentence"]))),
            bool(re.search(r"[.!?]$", item["sentence"])),
        ),
    )
    source_ids = {item["document_id"] for item in evidence if item.get("document_id")}
    coverage = min(1.0, len(source_ids) / 5.0)
    return {
        "topic": topic,
        "position": chosen["sentence"],
        "coverage": round(coverage, 3),
        "document_count": len(source_ids),
        "sources": sorted({item["source"] for item in evidence if item.get("source")}),
        "evidence": evidence[:limit],
        "method": "lexical centrality over locally stored text",
    }


def print_evidence(items: list[dict[str, Any]], *, prefix: str = "") -> None:
    for index, item in enumerate(items, start=1):
        print(f"{prefix}{index}. [{item['score']:.3f}] {item['source']}\n")
        print(textwrap.fill(item["sentence"], width=100, subsequent_indent="   "))
        print(f"   {item['url']}")


def command_init(args: argparse.Namespace) -> int:
    home = app_home(args.home)
    if not corpus_path(home).exists():
        write_json(corpus_path(home), [])
    if not beliefs_path(home).exists():
        write_json(beliefs_path(home), [])
    print(f"Local opinion journal ready at {home.resolve()}")
    return 0


def command_ingest(args: argparse.Namespace) -> int:
    home = app_home(args.home)
    successes = 0
    for location in args.locations:
        try:
            document = ingest_one(home, location, timeout=args.timeout)
            print(f"Stored {document['id']} from {document['url']} ({document['characters']:,} chars)")
            successes += 1
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Could not ingest {location}: {exc}", file=sys.stderr)
    return 0 if successes else 1


def command_search(args: argparse.Namespace) -> int:
    home = app_home(args.home)
    try:
        results = search_web(args.query, limit=args.limit, timeout=args.timeout)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not results:
        print("No web results found.")
        return 1

    print(f"Web results for: {args.query}")
    for index, result in enumerate(results, start=1):
        print(f"{index}. {result['title']}\n   {result['url']}")

    if args.no_ingest:
        return 0
    print("\nIngesting results into the local journal:")
    successes = 0
    for result in results:
        try:
            document = ingest_one(home, result["url"], timeout=args.timeout)
            print(f"Stored {document['id']} from {document['url']} ({document['characters']:,} chars)")
            successes += 1
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Skipped {result['url']}: {exc}", file=sys.stderr)
    return 0 if successes else 1


def command_review(args: argparse.Namespace) -> int:
    home = app_home(args.home)
    corpus = load_corpus(home)
    if not corpus:
        print("The local corpus is empty. Ingest pages first.", file=sys.stderr)
        return 1
    result = working_position(corpus, args.topic, args.limit)
    print(f"Topic: {result['topic']}")
    print("Working position (an extractive selection, not a trained semantic judgment):")
    if result["position"] is None:
        print("  No matching evidence was found.")
    else:
        print(textwrap.fill(result["position"], width=100, initial_indent="  ", subsequent_indent="  "))
        print(
            f"Evidence coverage: {result['coverage']:.0%} across "
            f"{result.get('document_count', len(result['sources']))} document(s) "
            f"from {len(result['sources'])} source(s)"
        )
        print("\nEvidence:")
        print_evidence(result["evidence"], prefix="  ")

    if args.save:
        beliefs = load_beliefs(home)
        result["recorded_at"] = now_iso()
        topic_key = " ".join(word_tokens(result["topic"], keep_stopwords=True))
        replaced = False
        for index, previous in enumerate(beliefs):
            previous_key = " ".join(word_tokens(str(previous.get("topic", "")), keep_stopwords=True))
            if previous_key == topic_key:
                beliefs[index] = result
                replaced = True
                break
        if not replaced:
            beliefs.append(result)
        write_json(beliefs_path(home), beliefs)
        action = "Updated" if replaced else "Recorded"
        print(f"\n{action} belief in {beliefs_path(home).resolve()}")
    return 0


def command_beliefs(args: argparse.Namespace) -> int:
    beliefs = load_beliefs(app_home(args.home))
    if args.topic:
        wanted = set(word_tokens(args.topic))
        beliefs = [belief for belief in beliefs if wanted & set(word_tokens(str(belief.get("topic", ""))))]
    if not beliefs:
        print("No beliefs recorded.")
        return 0
    for index, belief in enumerate(beliefs, start=1):
        print(f"{index}. {belief.get('topic', '(untitled)')} [{belief.get('coverage', 0):.0%} coverage]")
        position = belief.get("position") or "(no position)"
        print(textwrap.fill(position, width=100, initial_indent="   ", subsequent_indent="   "))
        print(f"   recorded {belief.get('recorded_at', 'unknown')}\n")
    return 0


def command_ask(args: argparse.Namespace) -> int:
    home = app_home(args.home)
    corpus = load_corpus(home)
    beliefs = load_beliefs(home)
    print(f"Query: {args.query}")
    matching = [belief for belief in beliefs if set(word_tokens(args.query)) & set(word_tokens(str(belief.get("topic", ""))))]
    if matching:
        print("\nStored working positions:")
        for belief in matching[:3]:
            print(textwrap.fill(str(belief.get("position") or "(no position)"), width=100, initial_indent="  ", subsequent_indent="  "))
    results = search_sentences(corpus, args.query, limit=args.limit)
    print("\nRetrieved evidence:")
    if results:
        print_evidence(results, prefix="  ")
    else:
        print("  No matching evidence in the local corpus.")
    return 0


def command_chat(args: argparse.Namespace) -> int:
    home = app_home(args.home)
    print("Local opinion journal. Commands: review TOPIC, ask QUERY, beliefs, quit")
    while True:
        try:
            line = input("opinion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        command, _, value = line.partition(" ")
        command = command.lower()
        value = value.strip()
        if command in {"quit", "exit"}:
            return 0
        if command == "beliefs":
            command_beliefs(argparse.Namespace(home=str(home), topic=None))
        elif command == "review" and value:
            command_review(argparse.Namespace(home=str(home), topic=value, limit=5, save=False))
        elif command == "ask" and value:
            command_ask(argparse.Namespace(home=str(home), query=value, limit=5))
        else:
            print("Use: review TOPIC, ask QUERY, beliefs, or quit")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dependency-free local web-review and belief journal")
    parser.add_argument("--home", default=None, help=f"journal directory (default: {DEFAULT_HOME})")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_command_home(command_parser: argparse.ArgumentParser) -> None:
        # argparse normally accepts a global option only before the subcommand.
        # Repeating it here makes both `--home X ingest ...` and
        # `ingest --home X ...` behave as users expect. SUPPRESS preserves the
        # global value when the command-level option is omitted.
        command_parser.add_argument(
            "--home",
            default=argparse.SUPPRESS,
            help=f"journal directory (default: {DEFAULT_HOME})",
        )

    init = sub.add_parser("init", help="create an empty local journal")
    add_command_home(init)
    init.set_defaults(func=command_init)

    ingest = sub.add_parser("ingest", help="read explicit URLs or local text/HTML files")
    add_command_home(ingest)
    ingest.add_argument("locations", nargs="+", help="http(s) URL(s) or local file path(s)")
    ingest.add_argument("--timeout", type=float, default=20.0)
    ingest.set_defaults(func=command_ingest)

    search = sub.add_parser("search", help="search the public web and optionally ingest results")
    add_command_home(search)
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=8)
    search.add_argument("--timeout", type=float, default=20.0)
    search.add_argument(
        "--no-ingest",
        action="store_true",
        help="show result URLs without fetching them into the local journal",
    )
    search.set_defaults(func=command_search)

    review = sub.add_parser("review", help="select a working position from stored evidence")
    add_command_home(review)
    review.add_argument("topic")
    review.add_argument("--limit", type=int, default=5)
    review.add_argument("--save", action="store_true", help="record the working position")
    review.set_defaults(func=command_review)

    beliefs = sub.add_parser("beliefs", help="show recorded working positions")
    add_command_home(beliefs)
    beliefs.add_argument("--topic")
    beliefs.set_defaults(func=command_beliefs)

    ask = sub.add_parser("ask", help="retrieve evidence and related stored positions")
    add_command_home(ask)
    ask.add_argument("query")
    ask.add_argument("--limit", type=int, default=5)
    ask.set_defaults(func=command_ask)

    chat = sub.add_parser("chat", help="open a tiny interactive local shell")
    add_command_home(chat)
    chat.set_defaults(func=command_chat)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

