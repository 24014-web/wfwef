#!/usr/bin/env python3
"""Appendable, lossless web-text archive using only the Python standard library.

Unlike ``compact_web_scraper.py``, this module does not summarize or rank
sentences.  In its default training mode it stores cleaned source/article text
and records the transform, quality gate, provenance, and automatic
topic/document-type labels.  ``--raw-visible`` keeps the complete visible DOM
text when an unfiltered compatibility copy is wanted.

Each page is stored as one ZIP member.  The member is tested with DEFLATE,
BZIP2, and LZMA and the smallest result is selected independently for that
page.  ZIP itself is the container; pre-compressing the text first would
usually make the archive larger and would prevent ZIP from finding patterns.

Examples::

    python lossless_web_archive.py add https://example.com/article \
        --archive knowledge.zip
    python lossless_web_archive.py crawl https://example.com \
        --archive knowledge.zip --max-pages 1000 --max-depth 2
    python lossless_web_archive.py crawl --topic "causes of ocean pollution" \
        --archive ocean.zip --workers 5
    python lossless_web_archive.py inspect knowledge.zip
    python lossless_web_archive.py export knowledge.zip training.jsonl
    python lossless_web_archive.py verify knowledge.zip
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import html
import json
import posixpath
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from zipfile import (
    ZIP_BZIP2,
    ZIP_DEFLATED,
    ZIP_LZMA,
    ZIP_STORED,
    BadZipFile,
    ZipFile,
)

from compact_web_scraper import (
    MAX_RESPONSE_BYTES,
    PageLinks,
    PageText,
    normalize_url,
    now_iso,
    page_title,
    robots_allowed,
    same_host,
)


ARCHIVE_VERSION = 2
RECORD_SCHEMA = 2
ARCHIVE_META = "archive.json"
RECORD_PREFIX = "records/"
JSON_KWARGS = {"ensure_ascii": False, "separators": (",", ":")}

# ``fetch_full_page`` accepts ordinary HTML and plain text, but public source
# archives also expose JSON, source code, and raw files with an
# ``application/octet-stream`` content type.  These are text resources and can
# be kept losslessly after UTF-8 decoding.  Binary files still stay out of the
# text archive.
TEXTUAL_CONTENT_TYPES = {
    "application/ecmascript",
    "application/graphql",
    "application/javascript",
    "application/json",
    "application/ld+json",
    "application/sql",
    "application/toml",
    "application/xml",
    "application/xhtml+xml",
    "application/x-httpd-php",
    "application/x-javascript",
    "application/x-sh",
    "application/x-yaml",
    "application/yaml",
    "text/css",
    "text/csv",
    "text/event-stream",
    "text/javascript",
    "text/markdown",
    "text/plain",
    "text/xml",
    "text/yaml",
}
TEXTUAL_EXTENSIONS = {
    ".bash", ".c", ".cc", ".cfg", ".conf", ".cpp", ".cs", ".css", ".csv",
    ".bat", ".dart", ".dockerfile", ".go", ".gradle", ".graphql", ".h", ".hpp", ".htm", ".html", ".ini",
    ".ipynb", ".java", ".jl", ".js", ".json", ".jsx", ".kt", ".kts", ".less",
    ".lua", ".m", ".md", ".php", ".pl", ".pm", ".ps1", ".py", ".r",
    ".rmd", ".rproj", ".rb", ".rs", ".sass", ".scala", ".scss", ".sh", ".sql", ".svelte",
    ".swift", ".tcl", ".tex", ".toml", ".ts", ".tsx", ".txt", ".tsv",
    ".vim", ".vue", ".xml", ".yaml", ".yml",
}

# File types that are useful to keep as training material when their source is
# available.  The archive still accepts other textual files in raw-visible
# mode, but training mode can use this distinction to avoid spending tokens on
# generated metadata, lock files, and embedded media.
CODE_EXTENSIONS = {
    ".bash", ".bat", ".c", ".cc", ".cpp", ".cs", ".css", ".dart", ".dockerfile",
    ".go", ".gradle", ".h", ".hpp", ".html", ".htm", ".java", ".jl", ".js", ".jsx",
    ".kt", ".kts", ".less", ".lua", ".m", ".makefile", ".mk", ".php", ".pl", ".pm",
    ".ps1", ".psm1", ".py", ".r", ".rb", ".rmd", ".rproj", ".rs", ".sass", ".scala",
    ".scss", ".sh", ".sql", ".sol", ".svelte", ".swift", ".tcl", ".tex", ".ts", ".tsx",
    ".vim", ".vue", ".zig",
}
DATA_EXTENSIONS = {
    ".csv", ".graphql", ".ini", ".json", ".md", ".toml", ".tsv", ".txt",
    ".xml", ".yaml", ".yml",
}
NOTEBOOK_EXTENSIONS = {".ipynb"}
TRAINING_ASSET_EXTENSIONS = {".gif", ".ico", ".jpeg", ".jpg", ".png", ".svg", ".webp"}
GENERATED_FILE_NAMES = {
    ".ds_store", ".editorconfig", ".gitattributes", ".gitignore", ".nvmrc",
    ".nojekyll", ".npmignore", ".prettierrc", ".travis.yml", "cargo.lock",
    "composer.lock", "desktop.ini", "gemfile.lock", "package-lock.json", "pnpm-lock.yaml",
    "poetry.lock", "thumbs.db", "yarn.lock",
}
TRACKING_QUERY_PARAMS = {
    "fbclid", "gclid", "mc_cid", "mc_eid", "msclkid", "msockid", "ref",
    "ref_", "utm_campaign", "utm_content", "utm_medium", "utm_source",
    "utm_term",
}

# Code-only mode is deliberately strict about what gets stored.  Navigation
# pages on these hosts are still fetched when they contain links to source
# files, but they are not written to the training archive.
CODE_BEARING_KINDS = {"code", "source", "notebook"}
CODE_HOSTS = {
    "github.com",
    "raw.githubusercontent.com",
    "gitlab.com",
    "bitbucket.org",
    "archive.softwareheritage.org",
    "sourceforge.net",
}
CODE_PAGE_MARKERS = (
    "/blob/", "/raw/", "/-/blob/", "/browse/content/", "/api/1/content/",
)

METHODS: dict[str, int] = {
    "stored": ZIP_STORED,
    "deflate": ZIP_DEFLATED,
    "bzip2": ZIP_BZIP2,
    "lzma": ZIP_LZMA,
}
METHOD_NAMES = {value: key for key, value in METHODS.items()}


# These are intentionally transparent heuristics.  They label pages for
# sorting; they do not alter the text stored in the archive.
TOPIC_KEYWORDS: dict[str, dict[str, int]] = {
    "history": {
        "history": 3, "historical": 2, "empire": 2, "ancient": 2,
        "century": 1, "war": 1, "genocide": 3, "revolution": 2,
    },
    "war_and_conflict": {
        "war": 3, "battle": 2, "military": 2, "conflict": 2,
        "invasion": 2, "army": 1, "terrorism": 2, "ceasefire": 2,
    },
    "science": {
        "science": 3, "research": 2, "experiment": 2, "physics": 2,
        "chemistry": 2, "biology": 2, "astronomy": 2, "mathematics": 2,
    },
    "technology": {
        "technology": 3, "software": 2, "computer": 2, "programming": 2,
        "algorithm": 2, "database": 2, "internet": 1, "artificial intelligence": 3,
    },
    "health": {
        "health": 3, "medical": 2, "medicine": 2, "disease": 2,
        "patient": 1, "clinical": 2, "treatment": 2, "diagnosis": 2,
    },
    "politics": {
        "politics": 3, "government": 2, "election": 2, "president": 2,
        "parliament": 2, "democracy": 2, "party": 1, "policy": 1,
    },
    "law": {
        "law": 2, "legal": 2, "court": 2, "legislation": 2,
        "regulation": 2, "constitution": 2, "rights": 1, "treaty": 2,
    },
    "economics": {
        "economy": 3, "economic": 3, "business": 2, "market": 2,
        "finance": 2, "company": 1, "trade": 2, "inflation": 2,
    },
    "environment": {
        "environment": 3, "climate": 3, "ecology": 2, "pollution": 2,
        "biodiversity": 2, "conservation": 2, "emissions": 2,
    },
    "education": {
        "education": 3, "school": 2, "university": 2, "student": 1,
        "teaching": 2, "curriculum": 2, "academic": 1,
    },
    "culture": {
        "culture": 3, "art": 2, "music": 2, "film": 2,
        "literature": 2, "language": 1, "museum": 2,
    },
    "sports": {
        "sport": 3, "football": 2, "soccer": 2, "basketball": 2,
        "cricket": 2, "tennis": 2, "athlete": 2, "tournament": 2,
    },
}


def slug(value: str, fallback: str = "other") -> str:
    result = re.sub(r"[^a-z0-9_-]+", "_", value.casefold()).strip("_")
    return result or fallback


def canonicalize_url(value: str) -> str:
    """Return a stable URL key for crawl deduplication.

    Query parameters that are normally analytics or session markers are
    removed, while meaningful parameters such as Software Heritage's
    ``origin_url`` and ``path`` are retained.  The displayed/source URL is
    never changed; this value is only used as a deduplication key.
    """
    raw = str(value or "").strip()
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        if parsed.scheme == "file" and parsed.path:
            return urllib.parse.urlunparse(("file", "", parsed.path, "", "", ""))
        return ""
    host = parsed.hostname.casefold() if parsed.hostname else parsed.netloc.casefold()
    if parsed.port and not ((parsed.scheme == "http" and parsed.port == 80) or (parsed.scheme == "https" and parsed.port == 443)):
        host = f"{host}:{parsed.port}"
    path = parsed.path or "/"
    pairs = [
        (key, val)
        for key, val in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in TRACKING_QUERY_PARAMS and not key.casefold().startswith("utm_")
    ]
    pairs.sort()
    query = urllib.parse.urlencode(pairs, doseq=True)
    return urllib.parse.urlunparse((parsed.scheme.casefold(), host, path, "", query, ""))


def normalize_filter_values(values: Iterable[str] | str | None, limit: int = 200) -> list[str]:
    """Normalize comma/newline separated block-list values.

    The crawler accepts both repeated CLI flags and multiline UI fields.  A
    small cap keeps an accidental paste from turning every URL check into a
    large regular-expression workload.
    """
    raw_values = [values] if isinstance(values, str) else list(values or [])
    result: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for piece in re.split(r"[\r\n,]+", str(raw)):
            cleaned = re.sub(r"\s+", " ", piece).strip().casefold()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
                if len(result) >= max(1, int(limit)):
                    return result
    return result


def normalize_block_sites(values: Iterable[str] | str | None, limit: int = 200) -> list[str]:
    """Normalize domain/URL block entries while retaining optional paths."""
    result: list[str] = []
    seen: set[str] = set()
    for value in normalize_filter_values(values, limit=limit):
        candidate = value.strip().rstrip("/")
        if not candidate:
            continue
        if "://" in candidate:
            parsed = urllib.parse.urlparse(candidate)
            host = (parsed.hostname or "").casefold()
            path = (parsed.path or "").rstrip("/")
            candidate = host + path if host else candidate
        else:
            candidate = candidate.removeprefix("//")
            if candidate.startswith("*."):
                candidate = candidate[2:]
        candidate = candidate.strip().rstrip("/")
        if candidate and candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return result


def _block_site_matches(url: str, pattern: str) -> bool:
    """Match a domain or optional domain path against a URL."""
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").casefold().rstrip(".")
    if not host:
        return False
    raw = str(pattern or "").strip().casefold().rstrip("/")
    if not raw:
        return False
    if "://" in raw:
        target = urllib.parse.urlparse(raw)
        target_host = (target.hostname or "").casefold().lstrip("*.").rstrip(".")
        target_path = (target.path or "").rstrip("/")
    else:
        target = urllib.parse.urlparse("//" + raw)
        target_host = (target.hostname or "").casefold().lstrip("*.").rstrip(".")
        target_path = (target.path or "").rstrip("/")
    if not target_host or not (host == target_host or host.endswith("." + target_host)):
        return False
    if not target_path:
        return True
    path = (parsed.path or "/").rstrip("/") or "/"
    return path == target_path or path.startswith(target_path + "/")


def _block_word_matches(text: str, term: str) -> bool:
    """Match a configured word or phrase without matching inside a word."""
    candidate = re.sub(r"\s+", " ", str(term or "").strip().casefold())
    if not candidate:
        return False
    haystack = re.sub(r"\s+", " ", str(text or "").casefold())
    if re.fullmatch(r"[\w\s-]+", candidate, flags=re.UNICODE):
        pattern = re.escape(candidate).replace(r"\ ", r"\s+")
        return re.search(r"(?<!\w)" + pattern + r"(?!\w)", haystack, flags=re.UNICODE) is not None
    return candidate in haystack


def block_reason_for_url(url: str, blocked_sites: Iterable[str] | None = None, blocked_words: Iterable[str] | None = None) -> tuple[str, str] | None:
    """Return ``(reason, value)`` when a URL matches a configured block."""
    for site in blocked_sites or ():
        if _block_site_matches(url, str(site)):
            return "site", str(site)
    parsed = urllib.parse.urlparse(url)
    url_blob = f"{parsed.netloc}{parsed.path}?{parsed.query}".casefold()
    for word in blocked_words or ():
        if _block_word_matches(url_blob, str(word)):
            return "word", str(word)
    return None


def block_reason_for_page(page: dict[str, Any], blocked_words: Iterable[str] | None = None) -> tuple[str, str] | None:
    """Check a fetched page's URL, title, and visible text for block words."""
    url = str(page.get("url", ""))
    url_reason = block_reason_for_url(url, blocked_words=blocked_words)
    if url_reason:
        return url_reason
    title = str(page.get("title", ""))
    text = str(page.get("text", ""))[:100_000]
    for word in blocked_words or ():
        if _block_word_matches(f"{title}\n{text}", str(word)):
            return "word", str(word)
    return None


def page_has_code(page: dict[str, Any]) -> bool:
    """Return whether a page is a code/source resource worth storing.

    Repository and archive directory pages are navigation only.  They remain
    crawlable so their raw/blob/content links can be reached, but code-only
    mode stores only source-bearing pages and known raw code endpoints.
    """
    kind = str(page.get("content_kind", "") or "").casefold()
    if kind in CODE_BEARING_KINDS:
        return True
    url = str(page.get("url", ""))
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.casefold()
    if _text_extension(url) in CODE_EXTENSIONS:
        return True
    host = (parsed.hostname or "").casefold()
    if host in CODE_HOSTS and any(marker in path for marker in CODE_PAGE_MARKERS):
        return True
    return False


def keyword_score(blob: str, keywords: dict[str, int]) -> int:
    score = 0
    for term, weight in keywords.items():
        # A word boundary works for ordinary words and still handles phrases
        # such as "artificial intelligence".
        count = len(re.findall(r"(?<!\w)" + re.escape(term) + r"(?!\w)", blob))
        score += min(count, 12) * weight
    return score


def _text_extension(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    candidates: list[str] = []
    for value in urllib.parse.parse_qs(parsed.query).get("filename", []):
        candidates.append(urllib.parse.unquote(value))
    for value in urllib.parse.parse_qs(parsed.query).get("path", []):
        candidates.append(urllib.parse.unquote(value))
    candidates.append(urllib.parse.unquote(parsed.path))
    for candidate in candidates:
        suffix = Path(candidate).suffix.casefold()
        if suffix:
            return suffix
    return ""


def _resource_name(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    candidates: list[str] = []
    for key in ("filename", "path"):
        candidates.extend(urllib.parse.unquote(value) for value in urllib.parse.parse_qs(parsed.query).get(key, []))
    candidates.append(urllib.parse.unquote(parsed.path))
    for candidate in candidates:
        name = Path(candidate).name
        if name:
            return name.casefold()
    return ""


def infer_content_kind(url: str, content_type: str = "", text: str = "", title: str = "") -> str:
    """Classify a page/resource for training filtering and export metadata."""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.casefold()
    name = _resource_name(url)
    extension = _text_extension(url)
    if "archive.softwareheritage.org" in parsed.netloc.casefold():
        if "/browse/search" in path or "/browse/origin/" in path or "/browse/directory/" in path or "/browse/snapshot/" in path or "/browse/revision/" in path:
            if "/browse/content/" not in path:
                return "archive_index"
        if "/browse/content/" in path or "/api/1/content/" in path:
            if extension in NOTEBOOK_EXTENSIONS:
                return "notebook"
            if name in GENERATED_FILE_NAMES or name.endswith(".lock"):
                return "generated"
            if extension in TRAINING_ASSET_EXTENSIONS:
                return "asset"
            if extension in CODE_EXTENSIONS and not (content_type in {"text/html", "application/xhtml+xml"} and extension in {".html", ".htm"}):
                return "code"
            if extension in DATA_EXTENSIONS:
                return "data"
            return "source"
    if name in GENERATED_FILE_NAMES or name.endswith(".lock"):
        return "generated"
    if extension in NOTEBOOK_EXTENSIONS:
        return "notebook"
    if extension in TRAINING_ASSET_EXTENSIONS:
        return "asset"
    if extension in CODE_EXTENSIONS and not (content_type in {"text/html", "application/xhtml+xml"} and extension in {".html", ".htm"}):
        return "code"
    if extension in DATA_EXTENSIONS and content_type not in {"text/html", "application/xhtml+xml"}:
        return "data"
    if re.search(r"/(?:kniha|knihy|autori|product|products)(?:/|$)", path):
        return "catalog"
    if content_type in {"application/json", "application/ld+json"}:
        return "data"
    if content_type and content_type.startswith("text/") and content_type not in {"text/html", "application/xhtml+xml"}:
        return "text"
    return "html"


def detect_language(text: str, content_kind: str = "") -> str:
    """Small standard-library language hint, suitable for filtering only."""
    if content_kind in {"code", "notebook", "data", "generated", "source", "asset", "archive_index"}:
        return "code" if content_kind in {"code", "notebook", "source"} else "und"
    sample = text[:100_000]
    if not sample.strip():
        return "und"
    counts = Counter()
    for char in sample:
        codepoint = ord(char)
        if 0x3040 <= codepoint <= 0x30ff:
            counts["ja"] += 2
        elif 0xac00 <= codepoint <= 0xd7af:
            counts["ko"] += 2
        elif 0x0400 <= codepoint <= 0x04ff:
            counts["ru"] += 2
        elif 0x0600 <= codepoint <= 0x06ff:
            counts["ar"] += 2
        elif char.isalpha() and char.isascii():
            counts["latin"] += 1
    if counts["ja"]:
        return "ja"
    if counts["ko"]:
        return "ko"
    if counts["ru"]:
        return "ru"
    if counts["ar"]:
        return "ar"
    if not counts["latin"]:
        return "und"
    words = re.findall(r"[A-Za-zÀ-ž]+", sample.casefold())
    stopwords = {
        "en": {"the", "and", "of", "to", "is", "in", "for", "with"},
        "cs": {"a", "je", "se", "na", "pro", "že", "z", "v", "k"},
        "de": {"der", "die", "das", "und", "ist", "von", "mit", "für"},
        "fr": {"le", "la", "les", "des", "et", "est", "dans", "pour"},
        "es": {"el", "la", "los", "las", "y", "es", "de", "para"},
    }
    scores = {lang: sum(word in values for word in words) for lang, values in stopwords.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "und"


def _long_base64_chars(text: str) -> int:
    return sum(len(match.group(0)) for match in re.finditer(r"[A-Za-z0-9+/]{200,}={0,2}", text))


def _repeated_line_ratio(text: str) -> float:
    lines = [re.sub(r"\s+", " ", line.strip()).casefold() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeated = sum(len(line) for line in lines if len(line) >= 28 and counts[line] > 1)
    return repeated / max(1, sum(len(line) for line in lines))


def assess_training_text(url: str, title: str, text: str, content_type: str = "", content_kind: str = "") -> dict[str, Any]:
    """Return explainable quality metadata used by training-mode crawls."""
    kind = content_kind or infer_content_kind(url, content_type, text, title)
    char_count = len(text)
    replacement_count = text.count("\ufffd")
    base64_count = _long_base64_chars(text)
    repeated_ratio = _repeated_line_ratio(text)
    reasons: list[str] = []
    critical = False
    if kind in {"asset", "generated", "archive_index"}:
        reasons.append(f"low_value_{kind}")
        critical = True
    if char_count < 120:
        reasons.append("too_short")
        critical = True
    if replacement_count:
        reasons.append("encoding_replacements")
        if replacement_count / max(1, char_count) > 0.005:
            critical = True
    if base64_count / max(1, char_count) > 0.05:
        reasons.append("embedded_base64_or_binary")
        if base64_count / max(1, char_count) > 0.20:
            critical = True
    if repeated_ratio > 0.50:
        reasons.append("high_repeated_line_ratio")
    score = 100
    score -= min(40, int(100 * replacement_count / max(1, char_count) * 8))
    score -= min(35, int(100 * base64_count / max(1, char_count) * 0.8))
    score -= min(25, int(100 * repeated_ratio * 0.35))
    if kind in {"asset", "generated", "archive_index"}:
        score -= 55
    if char_count < 120:
        score -= 35
    score = max(0, min(100, score))
    return {
        "include": not critical,
        "score": score,
        "content_kind": kind,
        "language": detect_language(text, kind),
        "characters": char_count,
        "replacement_characters": replacement_count,
        "base64_like_characters": base64_count,
        "repeated_line_ratio": round(repeated_ratio, 4),
        "reasons": reasons,
    }


def record_training_assessment(record: dict[str, Any]) -> dict[str, Any]:
    """Return training metadata, upgrading records from the pre-training schema."""
    existing = record.get("training")
    if isinstance(existing, dict) and "include" in existing and "score" in existing:
        return dict(existing)
    url = str(record.get("url", ""))
    title = str(record.get("title", ""))
    text = str(record.get("text", ""))
    content_type = str(record.get("content_type", ""))
    content_kind = str(record.get("content_kind", "")) or infer_content_kind(
        url, content_type, text, title
    )
    assessment = assess_training_text(url, title, text, content_type, content_kind)
    if isinstance(existing, dict) and existing.get("transform"):
        assessment["transform"] = existing["transform"]
    return assessment


def classify_page(title: str, text: str, url: str) -> dict[str, Any]:
    content_kind = infer_content_kind(url, "", text, title)
    title_blob = title.casefold()
    body_blob = text[:24_000].casefold()
    scored: list[tuple[int, str]] = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = keyword_score(title_blob, keywords) * 4 + keyword_score(body_blob, keywords)
        scored.append((score, topic))
    scored.sort(key=lambda item: (-item[0], item[1]))
    topics = [topic for score, topic in scored if score > 0][:3]
    primary = topics[0] if topics else "other"

    parsed = urllib.parse.urlparse(url)
    haystack = f"{url} {title} {text[:5_000]}".casefold()
    if content_kind in {"code", "source"}:
        document_type = "code"
    elif content_kind == "notebook":
        document_type = "notebook"
    elif content_kind == "data":
        document_type = "data"
    elif content_kind == "generated":
        document_type = "generated"
    elif content_kind == "catalog":
        document_type = "catalog"
    elif content_kind == "archive_index":
        document_type = "archive_index"
    elif "wikipedia.org/wiki/" in parsed.netloc + parsed.path or "encyclopedia" in haystack:
        document_type = "encyclopedia"
    elif re.search(r"/(docs?|documentation|api)(/|$)|\bapi reference\b", haystack):
        document_type = "documentation"
    elif re.search(r"\b(doi|abstract|peer[- ]review|journal article|methodology)\b", haystack):
        document_type = "academic"
    elif re.search(r"\b(breaking news|newsroom|news article|reported on)\b", haystack):
        document_type = "news"
    elif re.search(r"\b(forum|discussion|thread|comments?)\b", haystack):
        document_type = "forum"
    elif re.search(r"\b(act|statute|regulation|case law|ordinance)\b", haystack):
        document_type = "legal_reference"
    else:
        document_type = "article"
    return {
        "topic": primary,
        "topics": topics or ["other"],
        "document_type": document_type,
        "content_kind": content_kind,
    }


AI_DISCLOSURE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("explicit_ai_generation", r"\b(?:ai|a\.i\.)[- ]?(?:generated|written|created|produced)\b"),
    ("chatgpt_or_openai", r"\b(?:chatgpt|openai|gpt[- ]?[2345])\b"),
    ("other_ai_writer", r"\b(?:claude|gemini|copilot|jasper|llm|large language model)\b"),
    ("synthetic_content", r"\b(?:synthetic|machine[- ]generated)\s+(?:text|content|media)\b"),
)
HUMAN_SIGNAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("author_metadata", r"<meta[^>]+(?:name|property)=[\"'][^\"']*author[^\"']*[\"'][^>]+content=|rel=[\"']author[\"']"),
    ("publisher_metadata", r"<meta[^>]+(?:name|property)=[\"'][^\"']*(?:publisher|site_name)[^\"']*[\"'][^>]+content="),
    ("publication_date", r"(?:datepublished|article:published_time|pubdate|published_time|published[-_ ]date)"),
    # Anchor a plain-text byline to a line boundary so license prose such as
    # "published by the Foundation" is not mistaken for authorship.
    ("byline_text", r"(?:^|[\r\n])\s*(?:by|written by|reported by|edited by)\s+[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,3}"),
)


def assess_source(raw: str, readable: str, title: str = "") -> dict[str, Any]:
    """Return visible, explainable provenance signals for a fetched page.

    A web page cannot prove that every character was written by a person.  The
    result is therefore a screening signal, not an authorship guarantee.
    """
    haystack = f"{raw}\n{title}\n{readable[:20_000]}"
    folded = haystack.casefold()
    ai_signals = [label for label, pattern in AI_DISCLOSURE_PATTERNS if re.search(pattern, folded, flags=re.IGNORECASE)]
    human_signals = []
    for label, pattern in HUMAN_SIGNAL_PATTERNS:
        # Raw HTML is needed for metadata, while a byline should be checked
        # against extracted text to avoid matching license/footer prose.
        signal_source = readable if label == "byline_text" else haystack
        if re.search(pattern, signal_source, flags=re.IGNORECASE):
            human_signals.append(label)
    if ai_signals:
        verdict = "ai_flagged"
        reason = "The page contains an AI-generation, AI-tool, or synthetic-content signal."
    elif human_signals:
        verdict = "human_signals"
        reason = "The page exposes authorship or publication metadata, with no explicit AI-generation disclosure found."
    else:
        verdict = "unknown"
        reason = "No explicit AI disclosure or strong authorship metadata was found. This is not proof of human authorship."
    return {
        "verdict": verdict,
        "reason": reason,
        "ai_signals": ai_signals,
        "human_signals": human_signals,
        "archive_signals": [],
    }


def add_archive_provenance(url: str, verification: dict[str, Any]) -> dict[str, Any]:
    """Label a known public archive without claiming its content is human-written."""
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.casefold() not in {"archive.softwareheritage.org", "www.archive.softwareheritage.org"}:
        return verification
    signals = list(verification.get("archive_signals", []))
    if "software_heritage_archive" not in signals:
        signals.append("software_heritage_archive")
    verification["archive_signals"] = signals
    if verification.get("verdict") == "unknown":
        verification["verdict"] = "archive_signals"
        verification["reason"] = (
            "The page is served from the public Software Heritage archive; "
            "the archived content's authorship is not verified."
        )
    return verification


def is_software_heritage_search_url(url: str) -> bool:
    """Return whether *url* is Software Heritage's JavaScript origin search.

    The HTML page is intentionally only a shell: the result table is filled by
    JavaScript after it calls the public JSON API.  Recognising that one page
    lets the dependency-free crawler follow the same result links without
    pretending that the spinner text is the search result.
    """
    parsed = urllib.parse.urlparse(url)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.netloc.casefold() in {"archive.softwareheritage.org", "www.archive.softwareheritage.org"}
        and parsed.path.rstrip("/") == "/browse/search"
    )


def _query_flag(values: dict[str, list[str]], name: str) -> bool:
    raw = values.get(name, [])
    if not raw:
        return False
    return raw[-1].casefold() not in {"", "0", "false", "no", "off"}


def _software_heritage_browse_url(origin_url: str, visit_type: str = "") -> str:
    query: dict[str, str] = {"origin_url": origin_url}
    if visit_type and visit_type != "???":
        query["visit_type"] = visit_type
    return "https://archive.softwareheritage.org/browse/origin/directory/?" + urllib.parse.urlencode(query)


def software_heritage_search(url: str, timeout: float, limit: int = 100) -> dict[str, Any] | None:
    """Expand a Software Heritage origin-search URL through its JSON API.

    The returned rows are the API's origin records (URL, visit types, and
    snapshot id) plus archive-browser links.  No page text is summarised or
    rewritten.  ``None`` means the URL is not an SWH search page; an ``error``
    field in the returned mapping means it was an SWH page but its API call
    failed.
    """
    if not is_software_heritage_search_url(url):
        return None
    parsed = urllib.parse.urlparse(url)
    query_values = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    query = query_values.get("q", [""])[-1].strip()
    if not query:
        return {
            "provider": "software_heritage",
            "query": "",
            "results": [],
            "links": [],
            "returned": 0,
            "total_count": 0,
            "error": "The Software Heritage search URL has no q= query.",
        }

    safe_limit = max(1, min(int(limit), 100))
    metadata_search = _query_flag(query_values, "search_metadata") or _query_flag(
        query_values, "search_in_metadata"
    )
    if metadata_search:
        api_path = "/api/1/origin/metadata-search/"
        path_query = {"fulltext": query}
    else:
        api_path = "/api/1/origin/search/" + urllib.parse.quote(query, safe="") + "/"
        path_query = {"use_ql": "true" if _query_flag(query_values, "use_ql") else "false"}
    path_query.update(
        {
            "fields": "url,visit_types,snapshot_id",
            "limit": str(safe_limit),
            "with_visit": "true" if _query_flag(query_values, "with_visit") else "false",
            "with_content": "true" if _query_flag(query_values, "with_content") else "false",
        }
    )
    visit_type = query_values.get("visit_type", [""])[-1].strip().rstrip(",.;")
    if visit_type and visit_type != "any":
        path_query["visit_type"] = visit_type
    api_url = "https://archive.softwareheritage.org" + api_path + "?" + urllib.parse.urlencode(path_query)
    request = urllib.request.Request(
        api_url,
        headers={
            "User-Agent": "lossless-web-archive/0.1 (local read-only archive)",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                return {
                    "provider": "software_heritage",
                    "query": query,
                    "api_url": api_url,
                    "results": [],
                    "links": [],
                    "returned": 0,
                    "total_count": 0,
                    "error": "The Software Heritage API response exceeded the local size limit.",
                }
            charset = response.headers.get_content_charset() or "utf-8"
            total_header = response.headers.get("X-Total-Count", "")
            link_header = response.headers.get("Link", "")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        return {
            "provider": "software_heritage",
            "query": query,
            "api_url": api_url,
            "results": [],
            "links": [],
            "returned": 0,
            "total_count": 0,
            "error": f"Software Heritage API request failed: {type(exc).__name__}.",
        }
    try:
        payload = json.loads(raw.decode(charset, errors="replace"))
    except (LookupError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "provider": "software_heritage",
            "query": query,
            "api_url": api_url,
            "results": [],
            "links": [],
            "returned": 0,
            "total_count": 0,
            "error": f"Software Heritage API returned invalid JSON: {type(exc).__name__}.",
        }
    rows = payload if isinstance(payload, list) else payload.get("results", []) if isinstance(payload, dict) else []
    results: list[dict[str, Any]] = []
    links: list[str] = []
    seen: set[str] = set()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        origin = str(row.get("url", row.get("origin_url", ""))).strip()
        if not origin or origin in seen:
            continue
        seen.add(origin)
        visit_types = row.get("visit_types", [])
        if not isinstance(visit_types, list):
            visit_types = [visit_types] if visit_types else []
        visit_types = [str(value) for value in visit_types if str(value)]
        browse_url = _software_heritage_browse_url(origin, visit_types[0] if visit_types else "")
        item = {
            "url": origin,
            "origin_url": origin,
            "visit_types": visit_types,
            "snapshot_id": row.get("snapshot_id"),
            "browse_url": browse_url,
        }
        results.append(item)
        links.append(browse_url)
    next_url = ""
    match = re.search(r"<([^>]+)>\s*;\s*rel=[\"']next[\"']", link_header, flags=re.IGNORECASE)
    if match:
        next_url = match.group(1)
    try:
        total_count = int(total_header)
    except (TypeError, ValueError):
        total_count = len(results)
    return {
        "provider": "software_heritage",
        "query": query,
        "api_url": api_url,
        "with_visit": _query_flag(query_values, "with_visit"),
        "with_content": _query_flag(query_values, "with_content"),
        "search_metadata": metadata_search,
        "returned": len(results),
        "total_count": total_count,
        "next_url": next_url,
        "results": results,
        "links": links,
        "api_source_bytes": len(raw),
    }


def make_record(page: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    text = str(page.get("text", ""))
    text_bytes = text.encode("utf-8")
    digest = hashlib.sha256(text_bytes).hexdigest()
    labels = classify_page(str(page.get("title", "")), text, str(page.get("url", "")))
    content_kind = str(page.get("content_kind") or labels.get("content_kind") or "html")
    training = dict(page.get("training") or assess_training_text(
        str(page.get("url", "")),
        str(page.get("title", "")),
        text,
        str(page.get("content_type", "")),
        content_kind,
    ))
    record = {
        "schema": RECORD_SCHEMA,
        "url": str(page.get("url", "")),
        "canonical_url": canonicalize_url(str(page.get("url", ""))),
        "title": str(page.get("title", "")),
        **labels,
        "content_kind": content_kind,
        "language": str(training.get("language") or detect_language(text, content_kind)),
        "training": training,
        "depth": int(depth),
        "retrieved_at": str(page.get("retrieved_at", now_iso())),
        "text_encoding": "utf-8",
        "text_bytes": len(text_bytes),
        "text_sha256": digest,
        "source_bytes": int(page.get("source_bytes", 0)),
        "source_page_bytes": int(page.get("source_page_bytes", page.get("source_bytes", 0))),
        "source_encoding": str(page.get("source_encoding", "utf-8")),
        "content_type": str(page.get("content_type", "")),
        "http_status": int(page.get("http_status", 0) or 0),
        "source_verification": dict(page.get("source_verification", {})),
        "text": text,
    }
    for key in ("source_url", "content_url"):
        if page.get(key):
            record[key] = str(page[key])
    # Site adapters may add exact API/discovery metadata.  Keeping it beside
    # the text makes a replayable crawl auditable without changing the text.
    if page.get("deep_search"):
        record["deep_search"] = page["deep_search"]
    if page.get("dynamic_hints"):
        record["dynamic_hints"] = list(page["dynamic_hints"])
    if page.get("archive_tree"):
        record["archive_tree"] = dict(page["archive_tree"])
    return record


class FullPageText(PageText):
    """Visible-text extractor that keeps repeated lines and sentences."""

    def text(self) -> str:
        value = html.unescape("".join(self.parts))
        value = re.sub(r"[\t\r ]+", " ", value)
        value = re.sub(r"\n[ ]+", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()


class TrainingPageText(PageText):
    """Prefer the semantic page body and omit obvious site chrome."""

    _noise_tokens = (
        "nav", "menu", "header", "footer", "sidebar", "cookie", "modal", "banner",
        "breadcrumb", "pagination", "search", "login", "cart", "recommend", "related",
        "share", "social", "promo", "advert", "filter", "wishlist", "drawer", "popup",
    )
    _content_tokens = (
        "entry-content", "article-body", "post-content", "page-content", "main-content",
        "content-body", "article-content", "main",
    )
    _void_tags = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self._noise_depths: list[int] = []
        self.scope_depth: int | None = None
        self.scope_closed = False

    @classmethod
    def _attribute_blob(cls, attrs: list[tuple[str, str | None]]) -> str:
        selected = []
        for key, value in attrs:
            if key.casefold() in {"id", "class", "role", "aria-label"}:
                selected.append(value or "")
        return " ".join(selected).casefold()

    @classmethod
    def _is_noise(cls, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in cls.ignored:
            return True
        blob = cls._attribute_blob(attrs)
        tokens = set(re.findall(r"[a-z0-9_-]+", blob))
        if tokens.intersection(cls._noise_tokens):
            return True
        return any(token in blob for token in ("cookie-banner", "consent-dialog", "newsletter-form"))

    @classmethod
    def _is_content_scope(cls, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in {"main", "article"}:
            return True
        pairs = {key.casefold(): (value or "").casefold() for key, value in attrs}
        if pairs.get("role") == "main" or pairs.get("id") in {"main", "content", "article"}:
            return True
        blob = cls._attribute_blob(attrs)
        return any(token in blob for token in cls._content_tokens)

    def _inside_scope(self) -> bool:
        return self.scope_depth is None or self.depth >= self.scope_depth

    @property
    def excluded_depth(self) -> int:
        return len(self._noise_depths)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag in self._void_tags:
            if not self._is_noise(tag, attrs) and self.excluded_depth == 0 and self._inside_scope() and tag in self.blocks:
                self.parts.append("\n")
            return
        self.depth += 1
        if self.scope_depth is None and not self.scope_closed and self._is_content_scope(tag, attrs):
            self.scope_depth = self.depth
            # A malformed or template-generated header can leave an ancestor
            # noise block open in the parser.  Once a semantic main/article
            # scope is found, ancestor chrome must not suppress its content.
            self._noise_depths.clear()
        if self._is_noise(tag, attrs):
            self._noise_depths.append(self.depth)
            return
        if self.excluded_depth == 0 and self._inside_scope() and tag in self.blocks:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if not self._is_noise(tag, attrs) and self.excluded_depth == 0 and self._inside_scope() and tag in self.blocks:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in self._void_tags:
            return
        if self._noise_depths:
            if self._noise_depths[-1] == self.depth:
                self._noise_depths.pop()
            self.depth = max(0, self.depth - 1)
            return
        if self._inside_scope() and tag in self.blocks:
            self.parts.append("\n")
        if self.scope_depth == self.depth:
            self.scope_closed = True
        self.depth = max(0, self.depth - 1)

    def handle_data(self, data: str) -> None:
        if self.excluded_depth == 0 and self._inside_scope():
            self.parts.append(data)

    def text(self, url: str = "") -> str:
        value = html.unescape("".join(self.parts))
        value = re.sub(r"[\t\r ]+", " ", value)
        value = re.sub(r"\n[ ]+", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        lines: list[str] = []
        seen: set[str] = set()
        is_swh = "archive.softwareheritage.org" in urllib.parse.urlparse(url).netloc.casefold()
        for raw_line in value.splitlines():
            line = raw_line.strip()
            if not line:
                if lines and lines[-1] != "":
                    lines.append("")
                continue
            if is_swh and _is_software_heritage_noise(line):
                continue
            key = re.sub(r"\s+", " ", line).casefold()
            if len(line) >= 40 and key in seen:
                continue
            seen.add(key)
            lines.append(line)
        return "\n".join(lines).strip()


SWH_NOISE_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^raw file download\b.*",
        r"^cancelok$", r"^cancel$", r"^ok$", r"^×$",
        r"^(?:branches|releases|visits)(?: \(\d+\))?$",
        r"^no releases to show$", r"^display warning$", r"^formatbibtexcsl json$",
        r"^copy citation$", r"^copy identifiercopy permalink$", r"^generating citation \.\.\.$",
        r"^take a new snapshot of a software origin$", r"^processing \"take a new snapshot\" request \.\.\.$",
        r"^the requested archive is no longer available.*", r"^download link has expired$",
        r"^do you want to cook it again \?$", r"^invalid email !$", r"^the provided email is not well-formed\.$",
        r"^iframe embedding$", r"^permalinks$", r"^citations$", r"^branch: .*$", r"^refs/heads/.*",
        r"^visit type.*", r"^code$", r"^loading .* \.\.\.$",
    )
)


def _is_software_heritage_noise(line: str) -> bool:
    return any(pattern.match(line) for pattern in SWH_NOISE_PATTERNS)


def _decode_bytes(raw: bytes, declared_charset: str | None = None) -> tuple[str, str]:
    """Decode a response while handling common Japanese and legacy aliases."""
    aliases = {
        "windows-31j": "cp932", "x-sjis": "cp932", "shift_jis": "cp932",
        "shift-jis": "cp932", "ms_kanji": "cp932", "utf8": "utf-8",
    }
    candidates: list[str] = []
    if declared_charset:
        candidates.append(aliases.get(declared_charset.casefold(), declared_charset))
    head = raw[:8192].decode("ascii", errors="ignore")
    match = re.search(r"charset\s*=\s*[\"']?\s*([A-Za-z0-9._:-]+)", head, flags=re.IGNORECASE)
    if match:
        candidates.append(aliases.get(match.group(1).casefold(), match.group(1)))
    candidates.extend(["utf-8", "cp932", "iso-8859-1"])
    best_text = ""
    best_encoding = "utf-8"
    best_score: tuple[int, int] | None = None
    seen: set[str] = set()
    for encoding in candidates:
        key = encoding.casefold()
        if key in seen:
            continue
        seen.add(key)
        try:
            decoded = raw.decode(encoding, errors="replace")
        except (LookupError, UnicodeError):
            continue
        replacement = decoded.count("\ufffd")
        controls = sum(1 for char in decoded if ord(char) < 32 and char not in "\n\r\t")
        score = (replacement, controls)
        if best_score is None or score < best_score:
            best_text, best_encoding, best_score = decoded, encoding, score
    return best_text, best_encoding


def _clean_notebook(decoded: str) -> str:
    """Extract notebook markdown/code/output text while dropping embedded media."""
    try:
        payload = json.loads(decoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        return decoded.strip()
    cells = payload.get("cells") if isinstance(payload, dict) else None
    if not isinstance(cells, list):
        return decoded.strip()
    parts: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        cell_type = str(cell.get("cell_type", "")).casefold()
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(str(item) for item in source)
        source = str(source).strip()
        if source:
            if cell_type == "markdown":
                parts.append(source)
            elif cell_type == "code":
                parts.append("```python\n" + source + "\n```")
            else:
                parts.append(source)
        for output in cell.get("outputs", []) if isinstance(cell.get("outputs"), list) else []:
            if not isinstance(output, dict):
                continue
            if isinstance(output.get("text"), list):
                value = "".join(str(item) for item in output["text"]).strip()
                if value:
                    parts.append(value)
            data = output.get("data")
            if isinstance(data, dict):
                for key in ("text/plain", "text/markdown"):
                    value = data.get(key)
                    if isinstance(value, list):
                        value = "".join(str(item) for item in value)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
            if isinstance(output.get("traceback"), list):
                value = "\n".join(str(item) for item in output["traceback"]).strip()
                if value:
                    parts.append(value)
    return "\n\n".join(parts).strip()


def clean_training_text(decoded: str, url: str, content_type: str, content_kind: str) -> tuple[str, str]:
    """Return text and an optional transformation note for training mode."""
    if content_kind == "notebook":
        return _clean_notebook(decoded), "notebook_cells_without_embedded_media"
    if content_type in {"text/html", "application/xhtml+xml"}:
        parser = TrainingPageText()
        parser.feed(decoded)
        parser.close()
        return parser.text(url), "semantic_main_content_and_chrome_filter"
    return decoded.replace("\r\n", "\n").replace("\r", "\n").strip(), "normalized_newlines"


def _looks_like_text(raw: bytes) -> bool:
    """Conservative sniff for raw endpoints labelled octet-stream."""
    sample = raw[:20_000]
    if not sample or b"\x00" in sample:
        return False
    allowed_controls = {7, 8, 9, 10, 12, 13, 27}
    controls = sum(1 for value in sample if value < 32 and value not in allowed_controls)
    return controls / max(1, len(sample)) < 0.01


def _is_textual_response(content_type: str, url: str, raw: bytes) -> bool:
    if content_type in TEXTUAL_CONTENT_TYPES or content_type.startswith("text/"):
        return True
    if content_type in {"application/octet-stream", "binary/octet-stream"}:
        return _text_extension(url) in TEXTUAL_EXTENSIONS or _looks_like_text(raw)
    return False


def _prioritize_software_heritage_links(page_url: str, links: list[str]) -> list[str]:
    """Move archive tree/file links ahead of navigation chrome."""
    parsed = urllib.parse.urlparse(page_url)
    if parsed.netloc.casefold() not in {"archive.softwareheritage.org", "www.archive.softwareheritage.org"}:
        return links
    path = parsed.path.casefold()
    if "/browse/origin/directory" not in path and "/browse/directory/" not in path and "/browse/content/" not in path:
        return links

    def bucket(link: str) -> int:
        target = urllib.parse.urlparse(link)
        if target.netloc.casefold() not in {"archive.softwareheritage.org", "www.archive.softwareheritage.org"}:
            return 4
        target_path = target.path.casefold()
        target_query = urllib.parse.parse_qs(target.query)
        has_file_path = bool(target_query.get("path"))
        if has_file_path and "/browse/origin/content" in target_path:
            return 0
        if "/browse/content/sha1_git:" in target_path or "/api/1/content/" in target_path:
            return 0
        if "/browse/origin/directory" in target_path:
            return 1
        if "/browse/directory/" in target_path:
            return 1
        if "/browse/origin/" in target_path:
            return 2
        return 3

    return sorted(links, key=bucket)


SWH_HOSTS = {"archive.softwareheritage.org", "www.archive.softwareheritage.org"}


def _swh_link_kind(url: str) -> str:
    """Return the useful archive-tree kind for a Software Heritage URL."""
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.casefold() not in SWH_HOSTS:
        return ""
    path = parsed.path.casefold()
    values = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    has_origin = bool(values.get("origin_url"))
    has_path = bool(values.get("path") or values.get("filename"))
    if "/browse/origin/directory" in path and has_origin:
        return "directory"
    if "/browse/origin/content" in path and has_origin and has_path:
        return "content"
    if "/browse/content/sha1_git:" in path and has_path:
        return "content"
    if "/api/1/content/" in path and "/raw/" in path and has_path:
        return "raw"
    if "/browse/directory/" in path and has_origin:
        return "directory"
    return ""


def is_software_heritage_tree_url(url: str) -> bool:
    """Whether *url* is a file/directory URL in the SWH archive tree."""
    return bool(_swh_link_kind(url))


def _swh_link_key(url: str) -> str:
    """Build a per-page logical key to collapse browse/raw URL variants."""
    kind = _swh_link_kind(url)
    if not kind:
        return canonicalize_url(url) or url
    parsed = urllib.parse.urlparse(url)
    values = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    path_value = (values.get("path") or values.get("filename") or [""])[-1]
    path_value = urllib.parse.unquote(path_value).strip("/")
    origin = (values.get("origin_url") or [""])[-1]
    visit_type = (values.get("visit_type") or [""])[-1]
    # The repository-relative path is the stable identity within one origin;
    # retaining the origin prevents collisions when a page has several roots.
    if path_value:
        return f"swh:{kind}:{origin}:{visit_type}:{path_value}"
    return f"swh:{kind}:{origin}:{visit_type}:{parsed.path.casefold()}"


def crawl_link_depth(parent_url: str, child_url: str, depth: int) -> int:
    """Count an SWH repository tree as one logical crawl level.

    A Software Heritage search shell opens an origin directory and the files
    below that origin stay at the current level, so nested repositories do not
    require guessing their filesystem depth. Ordinary web links retain the
    normal depth increment.
    """
    parent_kind = _swh_link_kind(parent_url)
    child_kind = _swh_link_kind(child_url)
    if child_kind and (parent_kind == "directory" or is_software_heritage_search_url(parent_url)):
        return depth
    if child_kind and parent_kind in {"content", "raw"} and child_kind == "directory":
        return depth
    return depth + 1


def crawl_children(
    parent_url: str,
    links: list[str],
    depth: int,
    max_depth: int,
    seed_hosts: set[str],
    follow_external: bool,
) -> list[tuple[str, int]]:
    """Filter, deduplicate, and depth-label links for the crawl frontier."""
    if depth >= max_depth:
        return []
    ordered = _prioritize_software_heritage_links(parent_url, links)
    parent_kind = _swh_link_kind(parent_url)
    candidates: list[tuple[str, int]] = []
    seen_keys: set[str] = set()
    for link in ordered:
        if not follow_external and not same_host(link, seed_hosts):
            continue
        child_kind = _swh_link_kind(link)
        # Search shells should open origins only.  A directory exposes files
        # and child directories.  A file page only needs a route back to a
        # directory; following its navigation chrome wastes the page budget.
        if is_software_heritage_search_url(parent_url) and child_kind != "directory":
            continue
        if parent_kind == "directory" and child_kind not in {"directory", "content"}:
            continue
        if parent_kind in {"content", "raw"}:
            # Source pages are leaves.  Directory expansion happens at the
            # parent listing, so following their breadcrumb back to the root
            # only burns requests and can re-open the same repository.
            continue
        if parent_kind == "directory" and child_kind == "directory":
            parent_values = urllib.parse.parse_qs(urllib.parse.urlparse(parent_url).query, keep_blank_values=True)
            child_values = urllib.parse.parse_qs(urllib.parse.urlparse(link).query, keep_blank_values=True)
            parent_path = urllib.parse.unquote((parent_values.get("path") or [""])[-1]).strip("/")
            child_path = urllib.parse.unquote((child_values.get("path") or [""])[-1]).strip("/")
            if child_path == parent_path:
                continue
        key = _swh_link_key(link)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        child_depth = crawl_link_depth(parent_url, link, depth)
        if child_depth <= max_depth:
            candidates.append((link, child_depth))
    return candidates


def swh_frontload(parent_url: str) -> bool:
    """Use depth-first ordering while walking an SWH search/tree frontier."""
    return is_software_heritage_search_url(parent_url) or is_software_heritage_tree_url(parent_url)


def _fetch_payload(url: str, timeout: float) -> dict[str, Any] | None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "lossless-web-archive/0.2 (local read-only archive)",
            "Accept": "text/html,application/xhtml+xml,text/plain,application/json,application/javascript,text/javascript,application/xml,text/xml,text/css,text/markdown,application/octet-stream;q=0.8",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                return None
            return {
                "raw": raw,
                "url": normalize_url(response.geturl()) or url,
                "content_type": response.headers.get_content_type(),
                "charset": response.headers.get_content_charset(),
                "http_status": int(getattr(response, "status", None) or response.getcode() or 0),
            }
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def _software_heritage_directory_links(
    page_url: str,
    links: list[str],
    timeout: float,
) -> tuple[list[str], dict[str, Any]]:
    """Expand an SWH origin directory into links for every nested file.

    The browser's directory table is partly rendered by JavaScript and can
    omit entries from the initial HTML.  Each directory page exposes a
    ``swh:1:dir:<id>`` permalink; resolving that identifier through the
    public API lets us walk nested directories without spending the crawl
    budget on directory chrome.  The generated links point at the exact
    archived content object, so ``fetch_full_page`` can retrieve raw bytes.
    """
    parsed = urllib.parse.urlparse(page_url)
    if parsed.netloc.casefold() not in SWH_HOSTS or (
        "/browse/origin/directory" not in parsed.path.casefold()
        and "/browse/directory/" not in parsed.path.casefold()
    ):
        return [], {}
    values = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    origin_url = (values.get("origin_url") or [""])[-1].strip()
    if not origin_url:
        return [], {}
    visit_type = (values.get("visit_type") or [""])[-1].strip().rstrip(",.;")
    current_path = urllib.parse.unquote((values.get("path") or [""])[-1]).strip("/")

    directory_id = ""
    # The permalink for the current directory is the most reliable ID.  SWH
    # also places a link to the repository root in the navigation, which must
    # not win when the requested path is a nested directory.
    for source in links:
        source_path = urllib.parse.urlparse(source).path
        match = re.search(r"swh:1:dir:([0-9a-f]{40})", source_path, flags=re.IGNORECASE)
        if match:
            directory_id = match.group(1).lower()
            break
    if not directory_id:
        for source in [page_url, *links]:
            source_path = urllib.parse.urlparse(source).path
            match = re.search(r"(?:/browse/directory/|/api/1/directory/)([0-9a-f]{40})", source_path, flags=re.IGNORECASE)
            if match:
                directory_id = match.group(1).lower()
                break
    if not directory_id:
        return [], {}
    expanded: list[str] = []
    visited_directories: set[str] = set()
    directory_requests = 0
    directory_entries = 0
    file_entries = 0
    source_bytes = 0
    errors: list[str] = []
    max_directory_requests = 2_000
    max_file_entries = 50_000

    def object_hash(value: str) -> str:
        """Extract a raw sha1_git from either an API hash or an SWHID."""
        match = re.fullmatch(
            r"(?:swh:1:(?:dir|cnt|rev):)?([0-9a-f]{40})",
            str(value or "").strip(),
            flags=re.IGNORECASE,
        )
        return match.group(1).lower() if match else ""

    def append_file(relative_path: str, target: str, entry_type: str) -> None:
        nonlocal file_entries
        if file_entries >= max_file_entries:
            return
        # A revision/submodule target is a Git object, not file contents.  It
        # has no useful raw text endpoint, so leave it for a future revision
        # adapter instead of queueing a guaranteed shell/error page.
        if entry_type in {"rev", "revision"}:
            return
        name = Path(relative_path).name.casefold()
        extension = Path(relative_path).suffix.casefold()
        if name in GENERATED_FILE_NAMES or name.endswith(".lock") or extension in TRAINING_ASSET_EXTENSIONS:
            return
        query: dict[str, str] = {"origin_url": origin_url, "path": relative_path}
        if visit_type and visit_type != "???":
            query["visit_type"] = visit_type
        target_hash = object_hash(target)
        if target_hash:
            route = f"/browse/content/sha1_git:{target_hash}/"
        else:
            route = "/browse/origin/content/"
        expanded.append(f"https://{parsed.netloc}{route}?{urllib.parse.urlencode(query)}")
        file_entries += 1

    def walk(current_id: str, prefix: str) -> None:
        nonlocal directory_requests, directory_entries, source_bytes
        current_id = current_id.casefold()
        if current_id in visited_directories or directory_requests >= max_directory_requests:
            return
        if not re.fullmatch(r"[0-9a-f]{40}", current_id):
            return
        visited_directories.add(current_id)
        directory_requests += 1
        api_url = f"https://{parsed.netloc}/api/1/directory/{current_id}/"
        payload = _fetch_payload(api_url, timeout)
        if not payload:
            errors.append("directory_api_unavailable")
            return
        raw = payload.get("raw", b"")
        source_bytes += len(raw)
        try:
            decoded, _ = _decode_bytes(raw, payload.get("charset"))
            entries = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            errors.append("directory_api_invalid_json")
            return
        if not isinstance(entries, list):
            errors.append("directory_api_unexpected_shape")
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            if not name or name in {".", ".."} or "/" in name or "\\" in name:
                continue
            directory_entries += 1
            relative_path = posixpath.join(prefix, name) if prefix else name
            entry_type = str(entry.get("type", "")).casefold()
            target = str(entry.get("target", "")).strip()
            if entry_type in {"dir", "directory", "tree"}:
                target_hash = object_hash(target)
                if target_hash:
                    walk(target_hash, relative_path)
            else:
                append_file(relative_path, target, entry_type)

    walk(directory_id, current_path)
    root_api_url = f"https://{parsed.netloc}/api/1/directory/{directory_id}/"
    metadata: dict[str, Any] = {
        "directory_id": directory_id,
        "api_url": root_api_url,
        "recursive": not errors and directory_requests < max_directory_requests and file_entries < max_file_entries,
        "directories": len(visited_directories),
        "entries": directory_entries,
        "files": file_entries,
        "source_bytes": source_bytes,
    }
    if errors:
        metadata["error"] = errors[0]
    if directory_requests >= max_directory_requests or file_entries >= max_file_entries:
        metadata["truncated"] = True
    return expanded, metadata


def _software_heritage_raw_url(page_url: str, links: list[str]) -> str:
    for link in links:
        parsed = urllib.parse.urlparse(link)
        if (
            parsed.netloc.casefold() in {"archive.softwareheritage.org", "www.archive.softwareheritage.org"}
            and "/api/1/content/" in parsed.path
            and "/raw/" in parsed.path
        ):
            return link
    parsed = urllib.parse.urlparse(page_url)
    match = re.search(r"/browse/content/(sha1_git:[^/]+)", parsed.path, flags=re.IGNORECASE)
    path = urllib.parse.parse_qs(parsed.query).get("path", [""])[-1]
    if not match or not path:
        return ""
    query = urllib.parse.urlencode({"filename": path})
    return f"https://{parsed.netloc}/api/1/content/{match.group(1)}/raw/?{query}"


def fetch_full_page(
    url: str,
    timeout: float,
    training_mode: bool = False,
    expand_archives: bool = True,
) -> dict[str, Any] | None:
    """Fetch a page, optionally transforming it into training-oriented text.

    Raw-visible mode retains the complete visible DOM text for compatibility.
    Training mode uses semantic page bodies and, for Software Heritage content
    pages, fetches the exact raw source instead of archiving the archive UI.
    ``expand_archives`` controls the recursive directory API walk; source
    checks can leave it off so reviewing search results does not download an
    entire repository before the user starts a crawl.
    """
    payload = _fetch_payload(url, timeout)
    if payload is None:
        return None
    raw = payload["raw"]
    final_url = payload["url"]
    content_type = payload["content_type"]
    declared_charset = payload.get("charset")
    http_status = payload["http_status"]
    is_html = content_type in {"text/html", "application/xhtml+xml"}
    is_textual = _is_textual_response(content_type, final_url, raw)
    if not is_textual:
        return None
    decoded, source_encoding = _decode_bytes(raw, declared_charset)
    title = page_title(decoded) if is_html else ""
    if not title and not is_html:
        title = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlparse(final_url).query).get("filename", [""])[-1])
        if not title:
            title = Path(urllib.parse.urlparse(final_url).path).name
    links: list[str] = []
    if is_html:
        link_parser = PageLinks(final_url)
        link_parser.feed(decoded)
        link_parser.close()
        links = link_parser.links
        content_kind = infer_content_kind(final_url, content_type, decoded, title)
        if training_mode:
            readable, transform = clean_training_text(decoded, final_url, content_type, content_kind)
        else:
            text_parser = FullPageText()
            text_parser.feed(decoded)
            text_parser.close()
            readable = text_parser.text()
            transform = "visible_text"
    else:
        content_kind = infer_content_kind(final_url, content_type, decoded, title)
        if training_mode:
            readable, transform = clean_training_text(decoded, final_url, content_type, content_kind)
        else:
            readable = decoded.strip()
            transform = "raw_text"

    source_page_url = final_url
    content_url = ""
    source_page_bytes = len(raw)
    archive_tree: dict[str, Any] = {}
    swh_raw_unavailable = False
    # Software Heritage's browser page wraps source files in several thousand
    # characters of navigation and citation UI.  The linked raw API gives the
    # same archived bytes without that wrapper.
    if training_mode and is_html and "/browse/content/" in urllib.parse.urlparse(final_url).path.casefold():
        raw_url = _software_heritage_raw_url(final_url, links)
        raw_payload = _fetch_payload(raw_url, timeout) if raw_url else None
        if raw_payload and _is_textual_response(raw_payload["content_type"], raw_payload["url"], raw_payload["raw"]):
            raw_content = raw_payload["raw"]
            raw_decoded, raw_encoding = _decode_bytes(raw_content, raw_payload.get("charset"))
            raw_url = raw_payload["url"]
            raw_kind = infer_content_kind(raw_url, raw_payload["content_type"], raw_decoded, _resource_name(raw_url))
            # The raw API can label an archived source file such as
            # ``index.html`` as ``text/html``.  That MIME type describes the
            # file being stored, not a browser page to render.  Keep source
            # bytes verbatim so markup, templates, and scripts remain useful
            # training material instead of being reduced to visible DOM text.
            raw_content_type = raw_payload["content_type"]
            if raw_kind in {"code", "source", "data", "generated"}:
                raw_content_type = "text/plain"
            readable, transform = clean_training_text(
                raw_decoded, raw_url, raw_content_type, raw_kind
            )
            content_kind = raw_kind
            content_url = raw_url
            source_encoding = raw_encoding
            decoded = raw_decoded
            source_page_bytes = len(raw)
            raw = raw_content
            content_type = raw_payload["content_type"]
            http_status = raw_payload["http_status"]
            title = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query).get("filename", [""])[-1]) or title
        else:
            transform = "semantic_main_content_and_chrome_filter_raw_unavailable"
            swh_raw_unavailable = True

    if is_html and training_mode and expand_archives:
        api_links, archive_tree = _software_heritage_directory_links(final_url, links, timeout)
        if api_links:
            # API entries come first so a depth-first frontier reaches actual
            # source files before the browser's navigation/footer links.
            api_keys = {_swh_link_key(item) for item in api_links}
            if archive_tree.get("recursive"):
                # The recursive API walk already supplied every file below
                # this directory.  Keep non-tree navigation out of the
                # frontier and avoid fetching each directory listing again.
                links = api_links + [link for link in links if not _swh_link_kind(link)]
            else:
                links = api_links + [link for link in links if _swh_link_key(link) not in api_keys]

    deep_search = software_heritage_search(final_url, timeout)
    dynamic_hints: list[str] = []
    if is_software_heritage_search_url(final_url):
        dynamic_hints.append("javascript_rendered_origin_results")
        if deep_search and deep_search.get("error"):
            dynamic_hints.append("origin_api_unavailable")
    if archive_tree:
        dynamic_hints.append("software_heritage_directory_api")
        if archive_tree.get("error"):
            dynamic_hints.append(str(archive_tree["error"]))
    if deep_search:
        # Put API-discovered archive results first.  The page also contains a
        # large navigation footer; breadth-first limits should reach the
        # requested origins before spending the budget on that chrome.
        discovered_links = [link for link in deep_search.get("links", []) if link]
        links = discovered_links + [link for link in links if link not in discovered_links]
    links = _prioritize_software_heritage_links(final_url, links)
    # Ordinary navigation shells with no readable body are ignored.  A raw
    # text endpoint or an SWH search shell is retained because its exact
    # payload/metadata still provides useful crawl frontier information.
    if len(readable.strip()) < 40 and not deep_search and not (not is_html and is_textual):
        if not (training_mode and "archive.softwareheritage.org" in urllib.parse.urlparse(final_url).netloc.casefold() and links):
            return None
    source_verification = add_archive_provenance(
        final_url, assess_source(decoded, readable, title)
    )
    training = assess_training_text(final_url, title, readable, content_type, content_kind)
    if swh_raw_unavailable:
        reasons = list(training.get("reasons", []))
        if "swh_raw_source_unavailable" not in reasons:
            reasons.append("swh_raw_source_unavailable")
        training["include"] = False
        training["score"] = min(int(training.get("score", 0)), 15)
        training["reasons"] = reasons
    training["transform"] = transform
    page = {
        "url": final_url,
        "title": title,
        "text": readable,
        "links": links,
        "retrieved_at": now_iso(),
        "source_bytes": len(raw),
        "source_page_bytes": source_page_bytes,
        "source_encoding": source_encoding,
        "content_type": content_type,
        "http_status": http_status,
        "content_kind": content_kind,
        "training": training,
        "source_verification": source_verification,
    }
    if source_page_url != final_url:
        page["source_url"] = source_page_url
    if content_url:
        page["content_url"] = content_url
    if deep_search:
        page["deep_search"] = deep_search
    if dynamic_hints:
        page["dynamic_hints"] = dynamic_hints
    if archive_tree:
        page["archive_tree"] = archive_tree
    return page


def serialize(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, **JSON_KWARGS) + "\n").encode("utf-8")


def measure_member(data: bytes, method: int) -> int | None:
    """Return the exact ZIP compressed size for one candidate method."""
    import io

    try:
        buffer = io.BytesIO()
        with ZipFile(buffer, "w", compression=method, compresslevel=9, allowZip64=True) as archive:
            archive.writestr("candidate", data, compress_type=method, compresslevel=9)
            return archive.infolist()[0].compress_size
    except (NotImplementedError, RuntimeError, ValueError, OSError):
        return None


def choose_method(record: dict[str, Any]) -> tuple[str, bytes, dict[str, int]]:
    sizes: dict[str, int] = {}
    payloads: dict[str, bytes] = {}
    for name, method in METHODS.items():
        candidate = dict(record)
        candidate["compression"] = name
        payload = serialize(candidate)
        size = measure_member(payload, method)
        if size is not None:
            sizes[name] = size
            payloads[name] = payload
    if not sizes:
        raise RuntimeError("No ZIP compression method is available in this Python build")
    chosen = min(sizes, key=lambda name: (sizes[name], name == "stored"))
    return chosen, payloads[chosen], sizes


def archive_header() -> bytes:
    return (
        json.dumps(
            {
                "format": "mixed-lossless-web-text",
                "version": ARCHIVE_VERSION,
                "record_encoding": "JSON Lines encoded as UTF-8 ZIP members",
                "record_path": "records/<topic>/<sha256>.json",
                "compression_methods": ["stored", "deflate", "bzip2", "lzma"],
                "text_policy": "Training-mode records keep cleaned page/source text; raw-visible mode is available",
                "deduplication": "canonical URL and exact text hash",
                "created_at": now_iso(),
            },
            **JSON_KWARGS,
        )
        + "\n"
    ).encode("utf-8")


class LosslessArchive:
    """Small append API that commits each page as a ZIP member.

    ``compression='deflate'`` is the fast crawl default.  ``auto`` keeps the
    older smallest-member selection, which measures DEFLATE, BZIP2, and LZMA
    for every record and therefore costs more CPU.
    """

    def __init__(self, path: str | Path, deduplicate: bool = True, compression: str = "auto") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.deduplicate = deduplicate
        chosen_compression = str(compression or "auto").casefold()
        if chosen_compression != "auto" and chosen_compression not in METHODS:
            raise ValueError(f"Unknown compression method: {compression}")
        self.compression = chosen_compression
        # Crawls fetch in parallel.  ZIP central-directory updates and the
        # in-memory duplicate indexes must still be one writer at a time.
        self._lock = threading.RLock()
        self.known_hashes: set[str] = set()
        self.known_urls: set[str] = set()
        self.known_names: set[str] = set()
        if self.path.exists():
            try:
                with ZipFile(self.path, "r") as archive:
                    self.known_names = set(archive.namelist())
                    for name in self.known_names:
                        match = re.fullmatch(r"records/[^/]+/([0-9a-f]{64})\.json", name)
                        if not match:
                            continue
                        self.known_hashes.add(match.group(1))
                        if self.deduplicate:
                            try:
                                record = json.loads(archive.read(name).decode("utf-8"))
                            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                                continue
                            canonical = str(record.get("canonical_url") or canonicalize_url(record.get("url", "")))
                            if canonical:
                                self.known_urls.add(canonical)
            except BadZipFile as exc:
                raise ValueError(f"Archive is not a valid ZIP file: {self.path}") from exc
        else:
            with ZipFile(self.path, "w", compression=ZIP_DEFLATED, compresslevel=9, allowZip64=True) as archive:
                archive.writestr(ARCHIVE_META, archive_header(), compress_type=ZIP_DEFLATED, compresslevel=9)
            self.known_names.add(ARCHIVE_META)

    def add(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            digest = str(record["text_sha256"])
            canonical = str(record.get("canonical_url") or canonicalize_url(record.get("url", "")))
            if canonical and not record.get("canonical_url"):
                record["canonical_url"] = canonical
            if self.deduplicate and canonical and canonical in self.known_urls:
                return {"stored": False, "duplicate": True, "duplicate_reason": "url", "hash": digest}
            if self.deduplicate and digest in self.known_hashes:
                return {"stored": False, "duplicate": True, "duplicate_reason": "text", "hash": digest}
            topic = slug(str(record.get("topic", "other")))
            if self.compression == "auto":
                chosen, payload, sizes = choose_method(record)
            else:
                chosen = self.compression
                candidate = dict(record)
                candidate["compression"] = chosen
                payload = serialize(candidate)
                sizes = {chosen: 0}
            entry_name = f"{RECORD_PREFIX}{topic}/{digest}.json"
            # Opening and closing the ZIP for each page writes a fresh central
            # directory, so a successfully returned add() is immediately readable.
            with ZipFile(self.path, "a", compression=ZIP_DEFLATED, compresslevel=9, allowZip64=True) as archive:
                if entry_name in archive.namelist():
                    return {"stored": False, "duplicate": True, "duplicate_reason": "text", "hash": digest}
                archive.writestr(
                    entry_name,
                    payload,
                    compress_type=METHODS[chosen],
                    compresslevel=9,
                )
                sizes[chosen] = archive.getinfo(entry_name).compress_size
            self.known_names.add(entry_name)
            self.known_hashes.add(digest)
            if canonical:
                self.known_urls.add(canonical)
            return {
                "stored": True,
                "duplicate": False,
                "hash": digest,
                "entry": entry_name,
                "method": chosen,
                "member_bytes": sizes[chosen],
                "raw_bytes": len(payload),
            "all_sizes": sizes,
        }


def _topic_tokens(topic: str) -> list[str]:
    return [
        token.casefold()
        for token in re.findall(r"[^\W_]+", str(topic or ""), flags=re.UNICODE)
        if len(token) > 1
    ]


def discover_topic_sources(topic: str, limit: int = 20, timeout: float = 20.0) -> list[dict[str, Any]]:
    """Turn a short gathering goal into ranked, diverse crawl seeds.

    Discovery stays dependency-free by reusing the local HTML search adapter.
    The crawler still fetches and verifies each selected page before storing it;
    search-result text is never treated as training content.
    """
    cleaned = re.sub(r"\s+", " ", str(topic or "")).strip()
    if not cleaned:
        return []
    from local_opinion_ai import search_web

    safe_limit = max(1, min(int(limit), 100))
    candidates = search_web(cleaned, limit=max(safe_limit * 2, 12), timeout=timeout)
    if not isinstance(candidates, list):
        return []
    terms = _topic_tokens(cleaned)
    low_value_hosts = {
        "facebook.com", "instagram.com", "pinterest.com", "tiktok.com",
        "twitter.com", "x.com", "youtube.com",
    }

    def rank(item: dict[str, Any]) -> tuple[int, str]:
        title = str(item.get("title", ""))
        url = str(item.get("url", ""))
        blob = f"{title} {url}".casefold()
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").casefold()
        score = sum(3 if token in title.casefold() else 1 if token in blob else 0 for token in terms)
        if host.endswith((".edu", ".gov", ".org")):
            score += 2
        if any(part in parsed.path.casefold() for part in ("article", "research", "paper", "report", "docs", "wiki")):
            score += 1
        if any(host == blocked or host.endswith("." + blocked) for blocked in low_value_hosts):
            score -= 4
        return score, url

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in sorted((value for value in candidates if isinstance(value, dict)), key=rank, reverse=True):
        url = normalize_url(str(item.get("url", "")))
        if not url:
            continue
        key = canonicalize_url(url) or url
        if key in seen:
            continue
        seen.add(key)
        copy = dict(item)
        copy["url"] = url
        copy["discovery_score"] = rank(copy)[0]
        ordered.append(copy)

    # Prefer different sites early so the worker pool can make progress across
    # several hosts.  A second pass fills the requested limit with more pages
    # from a strong source when discovery returned too few domains.
    selected: list[dict[str, Any]] = []
    host_counts: Counter[str] = Counter()
    for item in ordered:
        host = urllib.parse.urlparse(item["url"]).netloc.casefold()
        if host_counts[host]:
            continue
        selected.append(item)
        host_counts[host] += 1
        if len(selected) >= safe_limit:
            break
    if len(selected) < safe_limit:
        for item in ordered:
            if item not in selected:
                selected.append(item)
            if len(selected) >= safe_limit:
                break
    return selected[:safe_limit]


class HostRateLimiter:
    """Reserve request slots per host without serializing different sites."""

    def __init__(self, interval: float = 0.0) -> None:
        self.interval = max(0.0, float(interval))
        self._lock = threading.Lock()
        self._next_allowed: dict[str, float] = {}

    def wait(self, url: str) -> None:
        if self.interval <= 0:
            return
        host = urllib.parse.urlparse(url).netloc.casefold()
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_allowed.get(host, now))
            self._next_allowed[host] = slot + self.interval
        remaining = slot - now
        if remaining > 0:
            time.sleep(remaining)


def run_parallel_crawl(
    seeds: Iterable[str] | None = None,
    *,
    topic: str = "",
    archive_path: str | Path = "knowledge.zip",
    max_pages: int = 500,
    max_depth: int = 2,
    timeout: float = 20.0,
    delay: float = 0.05,
    workers: int = 5,
    discover_limit: int = 20,
    follow_external: bool = False,
    respect_robots: bool = True,
    human_only: bool = False,
    training_mode: bool = True,
    blocked_words: Iterable[str] | None = None,
    blocked_sites: Iterable[str] | None = None,
    code_only: bool = False,
    stop_event: threading.Event | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Crawl a dynamic frontier with bounded concurrent fetch workers.

    The callback is invoked from the coordinator thread after each completed
    request, so callers can update a UI without sharing mutable archive state
    with network workers.  ``delay`` is the minimum spacing between requests
    to the same host; different hosts proceed concurrently.
    """
    stop_event = stop_event or threading.Event()
    safe_max_pages = max(1, int(max_pages))
    safe_max_depth = max(0, int(max_depth))
    safe_workers = max(1, min(int(workers), 64))
    safe_timeout = max(1.0, float(timeout))
    safe_delay = max(0.0, float(delay))
    safe_blocked_words = normalize_filter_values(blocked_words)
    safe_blocked_sites = normalize_block_sites(blocked_sites)
    safe_code_only = bool(code_only)
    manual_seeds: list[str] = []
    seed_values = [seeds] if isinstance(seeds, str) else list(seeds or [])
    for value in seed_values:
        normalized = normalize_url(str(value).strip())
        if normalized and normalized not in manual_seeds:
            manual_seeds.append(normalized)

    def emit(event: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(event)

    discovered: list[dict[str, Any]] = []
    if str(topic or "").strip():
        try:
            discovered = discover_topic_sources(str(topic), discover_limit, safe_timeout)
            emit({
                "status": "discovery",
                "topic": str(topic).strip(),
                "count": len(discovered),
                "results": discovered,
            })
            for item in discovered:
                url = normalize_url(str(item.get("url", "")))
                if url and url not in manual_seeds:
                    manual_seeds.append(url)
        except Exception as exc:
            emit({
                "status": "discovery_failed",
                "topic": str(topic).strip(),
                "error": f"{type(exc).__name__}: {exc}",
            })
            if not manual_seeds:
                raise ValueError(f"Topic discovery failed: {type(exc).__name__}") from exc
    if not manual_seeds:
        raise ValueError("Provide at least one valid http(s) seed URL or a topic to discover")

    seed_hosts = {urllib.parse.urlparse(seed).netloc.casefold() for seed in manual_seeds}
    pending: deque[tuple[str, int]] = deque((seed, 0) for seed in manual_seeds)
    visited: set[str] = set()
    robots_cache: dict[str, Any] = {}
    limiter = HostRateLimiter(safe_delay)
    archive = LosslessArchive(archive_path, compression="deflate")
    methods: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    started = time.monotonic()

    def enqueue(page_url: str, links: list[str], depth: int) -> None:
        children = crawl_children(
            page_url,
            links,
            depth,
            safe_max_depth,
            seed_hosts,
            follow_external,
        )
        if swh_frontload(page_url):
            for child in reversed(children):
                pending.appendleft(child)
        else:
            pending.extend(children)

    def fetch_one(url: str, depth: int) -> dict[str, Any]:
        try:
            limiter.wait(url)
            if stop_event.is_set():
                return {"url": url, "depth": depth, "cancelled": True, "page": None}
            return {
                "url": url,
                "depth": depth,
                "page": fetch_full_page(url, safe_timeout, training_mode=training_mode),
            }
        except Exception as exc:
            return {
                "url": url,
                "depth": depth,
                "page": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    scheduled = 0
    inflight: dict[Any, tuple[str, int]] = {}
    with ThreadPoolExecutor(max_workers=safe_workers, thread_name_prefix="web-fetch") as executor:
        while (pending or inflight) and not stop_event.is_set():
            while pending and len(inflight) < safe_workers and scheduled < safe_max_pages and not stop_event.is_set():
                url, depth = pending.popleft()
                visit_key = canonicalize_url(url) or url
                if visit_key in visited:
                    stats["duplicate_urls"] += 1
                    continue
                visited.add(visit_key)
                block_reason = block_reason_for_url(
                    url,
                    blocked_sites=safe_blocked_sites,
                    blocked_words=safe_blocked_words,
                )
                if block_reason:
                    reason_type, reason_value = block_reason
                    stats["blocked"] += 1
                    stats[f"blocked_{reason_type}s"] += 1
                    emit({
                        "status": "blocked",
                        "url": url,
                        "depth": depth,
                        "block_type": reason_type,
                        "block_value": reason_value,
                    })
                    continue
                if not follow_external and not same_host(url, seed_hosts):
                    stats["out_of_scope"] += 1
                    emit({"status": "out_of_scope", "url": url, "depth": depth})
                    continue
                if respect_robots and not robots_allowed(url, robots_cache, "lossless-web-archive", safe_timeout):
                    stats["robots_blocked"] += 1
                    emit({"status": "robots_blocked", "url": url, "depth": depth})
                    continue
                scheduled += 1
                stats["scheduled"] = scheduled
                future = executor.submit(fetch_one, url, depth)
                inflight[future] = (url, depth)
            if not inflight:
                break
            done, _ = wait(tuple(inflight), return_when=FIRST_COMPLETED)
            for future in done:
                url, depth = inflight.pop(future)
                outcome = future.result()
                stats["attempted"] += 1
                if outcome.get("cancelled"):
                    continue
                page = outcome.get("page")
                if page is None:
                    stats["failed"] += 1
                    emit({
                        "status": "fetch_failed",
                        "url": url,
                        "depth": depth,
                        "error": outcome.get("error", "request failed or unsupported content"),
                    })
                    continue
                stats["checked"] += 1
                page_url = str(page.get("url") or url)
                host = urllib.parse.urlparse(page_url).netloc.casefold() or "(local)"
                source_bytes = int(page.get("source_bytes", 0))
                text_bytes = len(str(page.get("text", "")).encode("utf-8"))
                stats["source_bytes"] += source_bytes
                stats["text_bytes"] += text_bytes
                page_block_reason = block_reason_for_page(page, safe_blocked_words)
                if page_block_reason:
                    reason_type, reason_value = page_block_reason
                    stats["blocked"] += 1
                    stats[f"blocked_{reason_type}s"] += 1
                    emit({
                        "status": "blocked",
                        "url": page_url,
                        "depth": depth,
                        "host": host,
                        "page": page,
                        "block_type": reason_type,
                        "block_value": reason_value,
                        "source_bytes": source_bytes,
                        "text_bytes": text_bytes,
                    })
                    continue
                training = dict(page.get("training") or {})
                if training_mode and not training.get("include", True):
                    stats["training_filtered"] += 1
                    emit({
                        "status": "training_filtered",
                        "url": page_url,
                        "depth": depth,
                        "host": host,
                        "page": page,
                        "training": training,
                        "source_bytes": source_bytes,
                        "text_bytes": text_bytes,
                    })
                    enqueue(page_url, page.get("links", []), depth)
                    continue
                verification = dict(page.get("source_verification", {}))
                verdict = str(verification.get("verdict", "unknown"))
                if human_only and verdict not in {"human_signals", "archive_signals"}:
                    stats["filtered"] += 1
                    emit({
                        "status": "filtered",
                        "url": page_url,
                        "depth": depth,
                        "host": host,
                        "page": page,
                        "training": training,
                        "verification": verification,
                        "source_bytes": source_bytes,
                        "text_bytes": text_bytes,
                    })
                    # Code-only crawls may need to traverse an unverified
                    # repository directory to reach its raw/blob files.
                    if page.get("deep_search") or safe_code_only:
                        enqueue(page_url, page.get("links", []), depth)
                    continue
                if safe_code_only and not page_has_code(page):
                    stats["code_filtered"] += 1
                    emit({
                        "status": "code_filtered",
                        "url": page_url,
                        "depth": depth,
                        "host": host,
                        "page": page,
                        "training": training,
                        "verification": verification,
                        "source_bytes": source_bytes,
                        "text_bytes": text_bytes,
                        "reason": "page is not a recognized source/code resource; links were retained for deeper traversal",
                    })
                    enqueue(page_url, page.get("links", []), depth)
                    continue
                record = make_record(page, depth=depth)
                result = archive.add(record)
                if result["duplicate"]:
                    stats["duplicates"] += 1
                    status = "duplicate"
                    compressed_bytes = 0
                else:
                    stats["stored"] += 1
                    compressed_bytes = int(result.get("member_bytes", 0))
                    stats["compressed_bytes"] += compressed_bytes
                    methods[str(result.get("method", ""))] += 1
                    topics[str(record.get("topic", "other"))] += 1
                    status = "stored"
                emit({
                    "status": status,
                    "url": page_url,
                    "depth": depth,
                    "host": host,
                    "page": page,
                    "record": record,
                    "archive_result": result,
                    "training": training,
                    "verification": verification,
                    "source_bytes": source_bytes,
                    "text_bytes": text_bytes,
                    "compressed_bytes": compressed_bytes,
                })
                enqueue(page_url, page.get("links", []), depth)

    elapsed = max(0.001, time.monotonic() - started)
    stats["elapsed_seconds"] = round(elapsed, 3)
    stats["pages_per_second"] = round(stats["attempted"] / elapsed, 2)
    stats["stopped"] = int(stop_event.is_set())
    stats["archive"] = str(Path(archive_path).resolve())
    stats["methods"] = dict(methods)
    stats["topics"] = dict(topics)
    stats["discovered"] = len(discovered)
    stats["blocked_words"] = list(safe_blocked_words)
    stats["blocked_sites"] = list(safe_blocked_sites)
    stats["code_only"] = int(safe_code_only)
    return dict(stats)


def iter_record_infos(archive: ZipFile) -> Iterator[Any]:
    for info in archive.infolist():
        if info.filename.startswith(RECORD_PREFIX) and info.filename.endswith(".json"):
            yield info


def load_local_source(source: str, training_mode: bool = False) -> dict[str, Any] | None:
    path = Path(source)
    local_url = path.resolve().as_uri()
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if len(raw) > MAX_RESPONSE_BYTES:
        return None
    decoded, source_encoding = _decode_bytes(raw)
    title = ""
    links: list[str] = []
    if path.suffix.casefold() in {".html", ".htm", ".xhtml"}:
        title = page_title(decoded)
        content_kind = infer_content_kind(local_url, "text/html", decoded, title)
        if training_mode:
            text, transform = clean_training_text(decoded, local_url, "text/html", content_kind)
        else:
            parser = FullPageText()
            parser.feed(decoded)
            parser.close()
            text = parser.text()
            transform = "visible_text"
        link_parser = PageLinks(local_url)
        link_parser.feed(decoded)
        link_parser.close()
        links = link_parser.links
    else:
        content_kind = infer_content_kind(local_url, "text/plain", decoded, path.name)
        if training_mode:
            text, transform = clean_training_text(decoded, path.as_uri(), "text/plain", content_kind)
        else:
            text = decoded
            transform = "raw_text"
    if len(text.strip()) < 40:
        return None
    content_type = "text/html" if path.suffix.casefold() in {".html", ".htm", ".xhtml"} else "text/plain"
    return {
        "url": local_url,
        "title": title or path.name,
        "text": text,
        "links": links,
        "retrieved_at": now_iso(),
        "source_bytes": len(raw),
        "source_page_bytes": len(raw),
        "source_encoding": source_encoding,
        "content_type": content_type,
        "http_status": 200,
        "content_kind": content_kind,
        "training": dict(assess_training_text(local_url, title or path.name, text, content_type, content_kind), transform=transform),
        "source_verification": assess_source(decoded, text, title or path.name),
    }


def load_source(source: str, timeout: float, training_mode: bool = False) -> dict[str, Any] | None:
    if re.match(r"^https?://", source, flags=re.IGNORECASE):
        return fetch_full_page(source, timeout, training_mode=training_mode)
    return load_local_source(source, training_mode=training_mode)


def command_add(args: argparse.Namespace) -> int:
    try:
        page = load_source(args.source, args.timeout, training_mode=args.training_mode)
        if page is None:
            print("Could not extract enough text from the source.", file=sys.stderr)
            return 1
        archive = LosslessArchive(args.archive)
        record = make_record(page)
        result = archive.add(record)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Add failed: {exc}", file=sys.stderr)
        return 1
    if result["duplicate"]:
        reason = result.get("duplicate_reason", "text")
        print(f"Skipped duplicate {reason}: {args.source}")
    else:
        print(f"Stored {result['entry']} using {result['method']} ({result['member_bytes']:,} compressed bytes)")
    print(f"Archive: {Path(args.archive).resolve()}")
    return 0


def command_crawl(args: argparse.Namespace) -> int:
    def report(event: dict[str, Any]) -> None:
        status = str(event.get("status", ""))
        url = str(event.get("url", ""))
        if status == "discovery":
            print(f"Discovered {int(event.get('count', 0)):,} source seed(s) for topic: {event.get('topic', '')}")
        elif status == "discovery_failed":
            print(f"Topic discovery failed: {event.get('error', 'unknown error')}", file=sys.stderr)
        elif status == "training_filtered":
            training = event.get("training") or {}
            tree = (event.get("page") or {}).get("archive_tree") or {}
            tree_note = f"; archive tree {tree.get('files', 0):,} file link(s)" if tree else ""
            print(
                f"Skipped training-low-value page {url}: "
                + ", ".join(str(value) for value in training.get("reasons", []))
                + tree_note
            )
        elif status == "filtered":
            verification = event.get("verification") or {}
            print(f"Skipped {url}: {verification.get('reason', 'source check failed')}")
        elif status == "fetch_failed":
            print(f"Fetch failed {url}: {event.get('error', 'request failed')}", file=sys.stderr)
        elif status == "blocked":
            print(
                f"Blocked {url} ({event.get('block_type', 'filter')}: "
                f"{event.get('block_value', '')})"
            )
        elif status == "code_filtered":
            print(f"Skipped non-code page {url}")

    try:
        stats = run_parallel_crawl(
            args.seeds,
            topic=args.topic,
            archive_path=args.archive,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            timeout=args.timeout,
            delay=args.delay,
            workers=args.workers,
            discover_limit=args.discover_limit,
            follow_external=args.follow_external,
            respect_robots=args.respect_robots,
            human_only=args.human_only,
            training_mode=args.training_mode,
            blocked_words=args.blocked_words,
            blocked_sites=args.blocked_sites,
            code_only=args.code_only,
            on_event=report,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Crawl failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Attempted: {stats.get('attempted', 0):,}; checked: {stats.get('checked', 0):,}; "
        f"stored unique pages: {stats.get('stored', 0):,}; "
        f"training-filtered: {stats.get('training_filtered', 0):,}; "
        f"code-filtered: {stats.get('code_filtered', 0):,}; "
        f"blocked: {stats.get('blocked', 0):,}; "
        f"duplicates skipped: {stats.get('duplicates', 0):,}"
    )
    print(f"Archive: {Path(args.archive).resolve()}")
    if stats.get("methods"):
        print("Compression methods: " + ", ".join(f"{key}={value}" for key, value in sorted(stats["methods"].items())))
    if stats.get("topics"):
        print("Topics: " + ", ".join(f"{key}={value}" for key, value in Counter(stats["topics"]).most_common()))
    print(f"Throughput: {stats.get('pages_per_second', 0):.2f} attempted page(s)/second with {args.workers} worker(s)")
    if args.blocked_words or args.blocked_sites:
        print(
            "Block lists: "
            f"{len(args.blocked_words or []):,} word(s), "
            f"{len(args.blocked_sites or []):,} site(s)"
        )
    if args.code_only:
        print("Code-only mode: enabled (navigation pages were traversed but not stored)")
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    try:
        with ZipFile(args.archive, "r") as archive:
            infos = list(iter_record_infos(archive))
            methods = Counter(METHOD_NAMES.get(info.compress_type, str(info.compress_type)) for info in infos)
            topics = Counter(info.filename.split("/", 2)[1] for info in infos if "/" in info.filename)
            content_kinds: Counter[str] = Counter()
            languages: Counter[str] = Counter()
            training_included = training_filtered = 0
            samples: list[dict[str, Any]] = []
            for info in infos:
                record = json.loads(archive.read(info).decode("utf-8"))
                assessment = record_training_assessment(record)
                content_kinds[str(record.get("content_kind") or assessment.get("content_kind") or "html")] += 1
                languages[str(record.get("language") or assessment.get("language") or "und")] += 1
                if assessment.get("include", True):
                    training_included += 1
                else:
                    training_filtered += 1
                if len(samples) < args.show:
                    samples.append(record)
            print(json.dumps({
                "archive": str(Path(args.archive).resolve()),
                "records": len(infos),
                "compressed_member_bytes": sum(info.compress_size for info in infos),
                "uncompressed_member_bytes": sum(info.file_size for info in infos),
                "methods": dict(methods),
                "topics": dict(topics),
                "content_kinds": dict(content_kinds),
                "languages": dict(languages),
                "training_included": training_included,
                "training_filtered": training_filtered,
            }, indent=2, ensure_ascii=False))
            for record in samples:
                print(f"\n{record.get('title') or '(untitled)'}")
                print(f"{record.get('url')} [{record.get('topic')}/{record.get('document_type')}; {record.get('compression')}]")
                print(record.get("text", "")[: args.chars])
    except (OSError, BadZipFile, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"Could not inspect archive: {exc}", file=sys.stderr)
        return 1
    return 0


def command_query(args: argparse.Namespace) -> int:
    wanted = [term.casefold() for term in re.findall(r"[^\W_]+", args.query, flags=re.UNICODE)]
    if not wanted:
        print("Query is empty.", file=sys.stderr)
        return 1
    ranked: list[tuple[int, dict[str, Any]]] = []
    try:
        with ZipFile(args.archive, "r") as archive:
            for info in iter_record_infos(archive):
                record = json.loads(archive.read(info).decode("utf-8"))
                haystack = f"{record.get('title', '')} {record.get('text', '')}".casefold()
                score = sum(haystack.count(term) for term in wanted)
                if score:
                    ranked.append((score, record))
    except (OSError, BadZipFile, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"Could not query archive: {exc}", file=sys.stderr)
        return 1
    ranked.sort(key=lambda item: (-item[0], item[1].get("title", "")))
    print(f"Query: {args.query}")
    for index, (score, record) in enumerate(ranked[: args.limit], start=1):
        print(f"\n{index}. [{score}] {record.get('title') or '(untitled)'}")
        print(f"{record.get('url')} [{record.get('topic')}/{record.get('document_type')}]\n")
        print(record.get("text", "")[: args.chars])
    if not ranked:
        print("No matching pages.")
    return 0


def command_export(args: argparse.Namespace) -> int:
    output = Path(args.output)
    try:
        if output.resolve() == Path(args.archive).resolve():
            print("Export output must be different from the input archive.", file=sys.stderr)
            return 1
    except OSError:
        pass
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    training_skipped = 0
    try:
        with ZipFile(args.archive, "r") as archive, output.open("w", encoding="utf-8", newline="\n") as handle:
            for info in iter_record_infos(archive):
                record = json.loads(archive.read(info).decode("utf-8"))
                if args.topic and record.get("topic") != args.topic:
                    continue
                if args.document_type and record.get("document_type") != args.document_type:
                    continue
                if args.training_only:
                    assessment = record_training_assessment(record)
                    if not assessment.get("include", True):
                        training_skipped += 1
                        continue
                    # Preserve the upgraded metadata when exporting a legacy
                    # archive that predates training-mode records.
                    record = dict(record)
                    record["training"] = assessment
                    record.setdefault("content_kind", assessment.get("content_kind", "html"))
                    record.setdefault("language", assessment.get("language", "und"))
                if args.format == "jsonl":
                    handle.write(json.dumps(record, **JSON_KWARGS) + "\n")
                else:
                    handle.write(record.get("text", ""))
                    handle.write("\n\n")
                written += 1
    except (OSError, BadZipFile, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    suffix = f"; skipped {training_skipped:,} low-value record(s)" if args.training_only else ""
    print(f"Exported {written:,} full-text record(s) to {output.resolve()}{suffix}")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    checked = 0
    try:
        with ZipFile(args.archive, "r") as archive:
            for info in iter_record_infos(archive):
                record = json.loads(archive.read(info).decode("utf-8"))
                text = str(record.get("text", ""))
                encoded = text.encode("utf-8")
                expected_hash = str(record.get("text_sha256", ""))
                expected_bytes = int(record.get("text_bytes", -1))
                if hashlib.sha256(encoded).hexdigest() != expected_hash or len(encoded) != expected_bytes:
                    print(f"Integrity failure: {info.filename}", file=sys.stderr)
                    return 1
                checked += 1
    except (OSError, BadZipFile, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        print(f"Verify failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verified {checked:,} full-text record(s); hashes and byte counts match.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Appendable lossless web-text ZIP archive")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="extract and append one URL or local text/HTML file")
    add.add_argument("source")
    add.add_argument("--archive", default="knowledge.zip")
    add.add_argument("--timeout", type=float, default=20.0)
    add.add_argument(
        "--raw-visible",
        dest="training_mode",
        action="store_false",
        help="keep the complete visible DOM text instead of training-oriented extraction",
    )
    add.set_defaults(training_mode=True)
    add.set_defaults(func=command_add)

    crawl = sub.add_parser("crawl", help="crawl seeds or discover sources from a topic")
    crawl.add_argument("seeds", nargs="*", help="one or more starting URLs (optional when --topic is used)")
    crawl.add_argument("--topic", help="short gathering goal; discover and crawl matching public sources")
    crawl.add_argument("--discover-limit", type=int, default=20, help="maximum topic-discovery seeds (default: 20)")
    crawl.add_argument("--archive", default="knowledge.zip")
    crawl.add_argument("--max-pages", type=int, default=500)
    crawl.add_argument("--max-depth", type=int, default=2)
    crawl.add_argument("--timeout", type=float, default=20.0)
    crawl.add_argument("--workers", type=int, default=5, help="concurrent fetch workers (default: 5)")
    crawl.add_argument("--delay", type=float, default=0.05, help="minimum seconds between requests to the same host")
    crawl.add_argument(
        "--block-word",
        dest="blocked_words",
        action="append",
        default=[],
        help="skip URLs/pages containing this word or phrase; repeat or separate with commas",
    )
    crawl.add_argument(
        "--block-site",
        dest="blocked_sites",
        action="append",
        default=[],
        help="skip this domain (or domain/path); repeat or separate with commas",
    )
    crawl.add_argument(
        "--code-only",
        action="store_true",
        help="store only recognized source/code pages while traversing navigation pages for links",
    )
    crawl.add_argument("--follow-external", action="store_true")
    crawl.add_argument(
        "--raw-visible",
        dest="training_mode",
        action="store_false",
        help="keep the complete visible DOM text instead of training-oriented extraction",
    )
    crawl.add_argument(
        "--human-only",
        action="store_true",
        help="store only pages with human or known archive provenance and no AI signals",
    )
    crawl.add_argument(
        "--ignore-robots",
        dest="respect_robots",
        action="store_false",
        help="do not consult robots.txt",
    )
    crawl.set_defaults(func=command_crawl, respect_robots=True, training_mode=True)

    inspect = sub.add_parser("inspect", help="show archive statistics and sample full text")
    inspect.add_argument("archive")
    inspect.add_argument("--show", type=int, default=3)
    inspect.add_argument("--chars", type=int, default=500)
    inspect.set_defaults(func=command_inspect)

    query = sub.add_parser("query", help="search complete stored text")
    query.add_argument("archive")
    query.add_argument("query")
    query.add_argument("--limit", type=int, default=10)
    query.add_argument("--chars", type=int, default=800)
    query.set_defaults(func=command_query)

    export = sub.add_parser("export", help="export complete text for training")
    export.add_argument("archive")
    export.add_argument("output")
    export.add_argument("--format", choices=("jsonl", "text"), default="jsonl")
    export.add_argument("--topic")
    export.add_argument("--document-type")
    export.add_argument(
        "--training-only",
        action="store_true",
        help="export only records that pass the training quality gate",
    )
    export.set_defaults(func=command_export)

    verify = sub.add_parser("verify", help="check every record's exact text hash")
    verify.add_argument("archive")
    verify.set_defaults(func=command_verify)
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

