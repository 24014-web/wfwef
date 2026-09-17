#!/usr/bin/env python3
"""Local browser UI for searching, checking, and crawling web text.

The server uses only the Python standard library.  Open the printed localhost
URL in a browser, search for a subject, review the source signals, select
results, and start an appendable full-text crawl.

The source check is deliberately conservative and explainable.  It can find
explicit AI-generation disclosures, authorship/publication metadata, and a
small set of known public-archive provenance signals, but no web page can
prove that every sentence was written by a person.  A crawl can also start
from a short topic, discover seeds automatically, and fetch several sites in
parallel.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import urllib.parse
from collections import Counter, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from local_opinion_ai import search_web
from lossless_web_archive import (
    LosslessArchive,
    fetch_full_page,
    crawl_children,
    canonicalize_url,
    is_software_heritage_search_url,
    make_record,
    normalize_block_sites,
    normalize_filter_values,
    normalize_url,
    robots_allowed,
    run_parallel_crawl,
    same_host,
    swh_frontload,
    software_heritage_search,
)


MAX_LOG_ENTRIES = 500
MAX_SEARCH_RESULTS = 20


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def domain_for(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower() or "(local)"


def bytes_label(value: int | float) -> str:
    amount = float(value)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if abs(amount) < 1024 or unit == units[-1]:
            return f"{amount:,.1f} {unit}" if unit != "B" else f"{int(amount):,} B"
        amount /= 1024
    return f"{amount:,.1f} TB"


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.casefold() in {"1", "true", "yes", "on"}
    return default


def bounded_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def bounded_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


class CrawlJob:
    """One background crawl with a thread-safe status snapshot."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.state = "idle"
        self.started_at = ""
        self.finished_at = ""
        self.current = ""
        self.archive_path = ""
        self.config: dict[str, Any] = {}
        self.counters: Counter[str] = Counter()
        self.logs: list[dict[str, Any]] = []
        self.sites: dict[str, dict[str, Any]] = {}

    def _log(self, entry: dict[str, Any]) -> None:
        entry.setdefault("time", utc_now())
        with self.lock:
            self.logs.append(entry)
            if len(self.logs) > MAX_LOG_ENTRIES:
                del self.logs[: len(self.logs) - MAX_LOG_ENTRIES]

    def _site_update(self, host: str, **values: int) -> None:
        with self.lock:
            site = self.sites.setdefault(
                host,
                {
                    "pages": 0,
                    "stored": 0,
                    "filtered": 0,
                    "code_filtered": 0,
                    "duplicates": 0,
                    "failed": 0,
                    "source_bytes": 0,
                    "text_bytes": 0,
                    "compressed_bytes": 0,
                },
            )
            for key, value in values.items():
                site[key] = int(site.get(key, 0)) + int(value)

    def start(self, config: dict[str, Any]) -> None:
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("A crawl is already running")
            self.stop_event = threading.Event()
            self.state = "running"
            self.started_at = utc_now()
            self.finished_at = ""
            self.current = ""
            self.archive_path = str(Path(config["archive"]).expanduser().resolve())
            self.config = dict(config)
            self.counters = Counter()
            self.logs = []
            self.sites = {}
            self.thread = threading.Thread(target=self._run, args=(dict(config),), daemon=True)
            self.thread.start()

    def stop(self) -> bool:
        with self.lock:
            active = bool(self.thread and self.thread.is_alive())
            if active:
                self.stop_event.set()
            return active

    def _set_current(self, value: str) -> None:
        with self.lock:
            self.current = value

    @staticmethod
    def _enqueue(
        pending: deque[tuple[str, int]],
        parent_url: str,
        links: list[str],
        depth: int,
        config: dict[str, Any],
        seed_hosts: set[str],
    ) -> None:
        children = crawl_children(
            parent_url,
            links,
            depth,
            config["max_depth"],
            seed_hosts,
            config["follow_external"],
        )
        if swh_frontload(parent_url):
            for child in reversed(children):
                pending.appendleft(child)
        else:
            pending.extend(children)

    def _count(self, key: str, amount: int = 1) -> None:
        with self.lock:
            self.counters[key] += amount

    def _handle_event(self, event: dict[str, Any]) -> None:
        """Translate coordinator events into the live UI counters and log."""
        status = str(event.get("status", ""))
        url = str(event.get("url", ""))
        if url:
            self._set_current(url)
        if status == "discovery":
            self._count("discovered", int(event.get("count", 0)))
            self._log({
                "url": "",
                "site": "topic discovery",
                "status": "discovery",
                "source_bytes": 0,
                "text_bytes": 0,
                "compressed_bytes": 0,
                "details": f"{event.get('count', 0)} seed(s) for {event.get('topic', '')}",
            })
            return
        if status == "discovery_failed":
            self._log({
                "url": "",
                "site": "topic discovery",
                "status": "discovery_failed",
                "source_bytes": 0,
                "text_bytes": 0,
                "compressed_bytes": 0,
                "details": str(event.get("error", "topic discovery failed")),
            })
            return
        if status == "out_of_scope":
            self._count("out_of_scope")
            return
        if status == "robots_blocked":
            self._count("robots_blocked")
            self._log({
                "url": url,
                "site": domain_for(url),
                "status": status,
                "source_bytes": 0,
                "text_bytes": 0,
                "compressed_bytes": 0,
                "details": "robots.txt disallowed this URL",
            })
            return
        if status == "blocked":
            page = event.get("page") or {}
            page_url = str(page.get("url") or url)
            host = str(event.get("host") or domain_for(page_url))
            source_bytes = int(event.get("source_bytes", page.get("source_bytes", 0)) or 0)
            text_bytes = int(event.get("text_bytes", len(str(page.get("text", "")).encode("utf-8"))) or 0)
            if page:
                self._count("checked")
                self._site_update(host, pages=1, filtered=1, source_bytes=source_bytes, text_bytes=text_bytes)
            self._count("blocked")
            self._log({
                "url": page_url,
                "site": host,
                "status": status,
                "source_bytes": source_bytes,
                "text_bytes": text_bytes,
                "compressed_bytes": 0,
                "details": f"blocked {event.get('block_type', 'filter')}: {event.get('block_value', '')}",
            })
            return
        if status == "fetch_failed":
            self._count("failed")
            self._site_update(domain_for(url), pages=1, failed=1)
            self._log({
                "url": url,
                "site": domain_for(url),
                "status": status,
                "source_bytes": 0,
                "text_bytes": 0,
                "compressed_bytes": 0,
                "details": str(event.get("error", "request failed, unsupported content, or too little text")),
            })
            return
        page = event.get("page") or {}
        host = str(event.get("host") or domain_for(str(page.get("url") or url)))
        source_bytes = int(event.get("source_bytes", page.get("source_bytes", 0)) or 0)
        text_bytes = int(event.get("text_bytes", len(str(page.get("text", "")).encode("utf-8"))) or 0)
        training = dict(event.get("training") or page.get("training") or {})
        if status == "training_filtered":
            self._count("checked")
            self._count("training_filtered")
            self._site_update(host, pages=1, filtered=1, source_bytes=source_bytes, text_bytes=text_bytes)
            reasons = ", ".join(str(value) for value in training.get("reasons", [])) or "training quality gate"
            tree = page.get("archive_tree") or {}
            tree_note = f"; archive tree {tree.get('files', 0):,} file link(s)" if tree else ""
            self._log({
                "url": str(page.get("url") or url),
                "site": host,
                "status": status,
                "verdict": str((page.get("source_verification") or {}).get("verdict", "unknown")),
                "source_bytes": source_bytes,
                "text_bytes": text_bytes,
                "compressed_bytes": 0,
                "details": f"{reasons}; score {training.get('score', 0)}/100{tree_note}",
            })
            return
        verification = dict(event.get("verification") or page.get("source_verification") or {})
        verdict = str(verification.get("verdict", "unknown"))
        if status == "filtered":
            self._count("checked")
            self._count("filtered")
            self._site_update(host, pages=1, filtered=1, source_bytes=source_bytes, text_bytes=text_bytes)
            self._log({
                "url": str(page.get("url") or url),
                "site": host,
                "status": status,
                "verdict": verdict,
                "source_bytes": source_bytes,
                "text_bytes": text_bytes,
                "compressed_bytes": 0,
                "details": str(verification.get("reason", "source check failed")),
            })
            return
        if status == "code_filtered":
            self._count("checked")
            self._count("filtered")
            self._count("code_filtered")
            self._site_update(
                host,
                pages=1,
                filtered=1,
                code_filtered=1,
                source_bytes=source_bytes,
                text_bytes=text_bytes,
            )
            self._log({
                "url": str(page.get("url") or url),
                "site": host,
                "status": status,
                "verdict": verdict,
                "source_bytes": source_bytes,
                "text_bytes": text_bytes,
                "compressed_bytes": 0,
                "details": str(event.get("reason", "page is not a recognized code resource")),
            })
            return
        record = event.get("record") or {}
        result = event.get("archive_result") or {}
        self._count("checked")
        compressed_bytes = int(event.get("compressed_bytes", result.get("member_bytes", 0)) or 0)
        if status == "duplicate":
            self._count("duplicates")
            self._site_update(host, pages=1, duplicates=1, source_bytes=source_bytes, text_bytes=text_bytes)
        elif status == "stored":
            self._count("stored")
            self._count("source_bytes", source_bytes)
            self._count("text_bytes", text_bytes)
            self._count("compressed_bytes", compressed_bytes)
            self._site_update(
                host,
                pages=1,
                stored=1,
                source_bytes=source_bytes,
                text_bytes=text_bytes,
                compressed_bytes=compressed_bytes,
            )
        else:
            return
        self._log({
            "url": str(page.get("url") or url),
            "site": host,
            "status": status,
            "verdict": verdict,
            "source_bytes": source_bytes,
            "text_bytes": text_bytes,
            "compressed_bytes": compressed_bytes,
            "method": result.get("method", ""),
            "topic": record.get("topic", "other"),
            "document_type": record.get("document_type", "article"),
            "details": "; ".join(
                item for item in (
                    str(verification.get("reason", "")),
                    f"training score {training.get('score', 0)}/100" if training else "",
                    f"duplicate by {result.get('duplicate_reason')}" if result.get("duplicate") else "",
                ) if item
            ),
        })

    def _run(self, config: dict[str, Any]) -> None:
        try:
            stats = run_parallel_crawl(
                config.get("seeds", []),
                topic=str(config.get("topic", "")),
                archive_path=config["archive"],
                max_pages=config["max_pages"],
                max_depth=config["max_depth"],
                timeout=config["timeout"],
                delay=config["delay"],
                workers=config["workers"],
                discover_limit=config["discover_limit"],
                follow_external=config["follow_external"],
                respect_robots=config["respect_robots"],
                human_only=config["human_only"],
                training_mode=config["training_mode"],
                blocked_words=config["blocked_words"],
                blocked_sites=config["blocked_sites"],
                code_only=config["code_only"],
                stop_event=self.stop_event,
                on_event=self._handle_event,
            )
            with self.lock:
                self.state = "stopped" if self.stop_event.is_set() else "done"
            self._log({
                "url": "",
                "site": "crawl",
                "status": "complete" if not self.stop_event.is_set() else "stopped",
                "source_bytes": 0,
                "text_bytes": 0,
                "compressed_bytes": 0,
                "details": f"{stats.get('attempted', 0):,} attempted at {stats.get('pages_per_second', 0):.2f} page(s)/second",
            })
        except Exception as exc:  # expose a useful UI error and stop cleanly
            with self.lock:
                self.state = "error"
            self._log({
                "url": self.current,
                "site": domain_for(self.current) if self.current else "",
                "status": "error",
                "source_bytes": 0,
                "text_bytes": 0,
                "compressed_bytes": 0,
                "details": f"{type(exc).__name__}: {exc}",
            })
        finally:
            with self.lock:
                self.current = ""
                self.finished_at = utc_now()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            counters = dict(self.counters)
            sites = {key: dict(value) for key, value in self.sites.items()}
            logs = list(self.logs)
            state = self.state
            archive_path = self.archive_path
            config = dict(self.config)
            started_at = self.started_at
            finished_at = self.finished_at
            current = self.current
        archive_bytes = 0
        if archive_path:
            try:
                archive_bytes = Path(archive_path).stat().st_size
            except OSError:
                pass
        return {
            "state": state,
            "started_at": started_at,
            "finished_at": finished_at,
            "current": current,
            "archive": archive_path,
            "archive_bytes": archive_bytes,
            "config": config,
            "counters": counters,
            "sites": sites,
            "logs": logs,
        }


JOB = CrawlJob()


def source_check_result(
    result: dict[str, str],
    timeout: float,
    robots_cache: dict[str, Any] | None = None,
    training_mode: bool = True,
) -> dict[str, Any]:
    url = result.get("url", "")
    if robots_cache is not None and not robots_allowed(
        url, robots_cache, "lossless-web-archive-ui", timeout
    ):
        return {
            **result,
            "site": domain_for(url),
            "status": "robots_blocked",
            "verdict": "unknown",
            "source_bytes": 0,
            "text_bytes": 0,
            "http_status": 0,
            "content_type": "",
            "verification": {
                "verdict": "unknown",
                "reason": "robots.txt disallowed this URL.",
                "ai_signals": [],
                "human_signals": [],
            },
        }
    # Search checks need only inspect the selected landing page.  The crawl
    # itself performs the recursive Software Heritage directory walk after a
    # user starts it, avoiding a full repository download for every result.
    page = fetch_full_page(url, timeout, training_mode=training_mode, expand_archives=False)
    if page is None:
        return {
            **result,
            "site": domain_for(url),
            "status": "unavailable",
            "verdict": "unknown",
            "source_bytes": 0,
            "text_bytes": 0,
            "http_status": 0,
            "content_type": "",
            "verification": {
                "verdict": "unknown",
                "reason": "The page could not be fetched or did not contain enough supported text.",
                "ai_signals": [],
                "human_signals": [],
            },
        }
    verification = dict(page.get("source_verification", {}))
    return {
        "title": page.get("title") or result.get("title", ""),
        "url": page.get("url", url),
        "origin_url": result.get("origin_url", ""),
        "provider": result.get("provider", ""),
        "snapshot_id": result.get("snapshot_id"),
        "site": domain_for(page.get("url", url)),
        "status": "checked",
        "verdict": verification.get("verdict", "unknown"),
        "source_bytes": int(page.get("source_bytes", 0)),
        "text_bytes": len(str(page.get("text", "")).encode("utf-8")),
        "http_status": int(page.get("http_status", 0)),
        "content_type": page.get("content_type", ""),
        "content_kind": page.get("content_kind", ""),
        "language": (page.get("training") or {}).get("language", "und"),
        "training": page.get("training", {}),
        "verification": verification,
    }


def run_software_heritage_search(url: str, limit: int, timeout: float) -> dict[str, Any]:
    """Search SWH's public API, then check the archive browser pages."""
    info = software_heritage_search(url, timeout, limit=limit)
    if info is None:
        raise ValueError("Not a Software Heritage origin-search URL")
    if info.get("error"):
        raise RuntimeError(str(info["error"]))
    discovered = [
        {
            "title": str(row.get("origin_url", row.get("url", ""))),
            "url": str(row.get("browse_url", "")),
            "origin_url": str(row.get("origin_url", row.get("url", ""))),
            "provider": "software_heritage",
            "snapshot_id": row.get("snapshot_id"),
            "status": "discovered",
        }
        for row in info.get("results", [])
        if row.get("browse_url")
    ]
    checked: list[dict[str, Any] | None] = [None] * len(discovered)

    def check_one(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        index, result = item
        return index, source_check_result(result, timeout, None, True)

    with ThreadPoolExecutor(max_workers=min(5, max(1, len(discovered)))) as executor:
        for future in executor.map(check_one, enumerate(discovered)):
            index, checked_result = future
            checked[index] = checked_result
    return {
        "query": url,
        "provider": "software_heritage",
        "deep_search": {
            "query": info.get("query", ""),
            "returned": info.get("returned", 0),
            "total_count": info.get("total_count", 0),
            "next_url": info.get("next_url", ""),
            "api_url": info.get("api_url", ""),
        },
        "results": [item for item in checked if item is not None],
        "checked_at": utc_now(),
    }


def run_search(query: str, limit: int, timeout: float) -> dict[str, Any]:
    normalized_query = normalize_url(query)
    if normalized_query and is_software_heritage_search_url(normalized_query):
        return run_software_heritage_search(normalized_query, limit, timeout)
    results = search_web(query, limit=max(1, min(limit, MAX_SEARCH_RESULTS)), timeout=timeout)
    checked: list[dict[str, Any] | None] = [None] * len(results)
    robots_cache: dict[str, Any] = {}
    to_check: list[tuple[int, dict[str, str]]] = []
    for index, result in enumerate(results):
        try:
            url = result.get("url", "")
            if not robots_allowed(url, robots_cache, "lossless-web-archive-ui", timeout):
                checked[index] = {
                    **result,
                    "site": domain_for(url),
                    "status": "robots_blocked",
                    "verdict": "unknown",
                    "source_bytes": 0,
                    "text_bytes": 0,
                    "http_status": 0,
                    "content_type": "",
                    "verification": {
                        "verdict": "unknown",
                        "reason": "robots.txt disallowed this URL.",
                        "ai_signals": [],
                        "human_signals": [],
                    },
                }
            else:
                to_check.append((index, result))
        except Exception as exc:
            checked[index] = {
                **result,
                "site": domain_for(result.get("url", "")),
                "status": "check_error",
                "verdict": "unknown",
                "source_bytes": 0,
                "text_bytes": 0,
                "http_status": 0,
                "content_type": "",
                "verification": {
                    "verdict": "unknown",
                    "reason": f"Source check error: {type(exc).__name__}: {exc}",
                    "ai_signals": [],
                    "human_signals": [],
                },
            }
    def check_one(item: tuple[int, dict[str, str]]) -> tuple[int, dict[str, Any]]:
        index, result = item
        return index, source_check_result(result, timeout, None, True)

    # Checking result pages is network-bound.  A small pool keeps the UI
    # responsive while avoiding an unbounded burst of requests.
    with ThreadPoolExecutor(max_workers=min(5, max(1, len(to_check)))) as executor:
        for future in executor.map(check_one, to_check):
            index, checked_result = future
            checked[index] = checked_result
    return {"query": query, "results": [item for item in checked if item is not None], "checked_at": utc_now()}


def parse_start_config(payload: dict[str, Any]) -> dict[str, Any]:
    raw_seeds = payload.get("seeds", [])
    if isinstance(raw_seeds, str):
        raw_seeds = raw_seeds.splitlines()
    seeds: list[str] = []
    for value in raw_seeds if isinstance(raw_seeds, list) else []:
        normalized = normalize_url(str(value).strip())
        if normalized and normalized not in seeds:
            seeds.append(normalized)
    topic = str(payload.get("topic", "")).strip()
    if not seeds and not topic:
        raise ValueError("Add a seed URL or a topic to discover sources")
    archive = str(payload.get("archive", "knowledge.zip")).strip() or "knowledge.zip"
    return {
        "seeds": seeds,
        "topic": topic,
        "archive": archive,
        "max_pages": bounded_int(payload.get("max_pages"), 500, 1, 100_000),
        "max_depth": bounded_int(payload.get("max_depth"), 2, 0, 20),
        "timeout": bounded_float(payload.get("timeout"), 20.0, 1.0, 120.0),
        "delay": bounded_float(payload.get("delay"), 0.05, 0.0, 60.0),
        "workers": bounded_int(payload.get("workers"), 5, 1, 64),
        "discover_limit": bounded_int(payload.get("discover_limit"), 20, 1, 100),
        "follow_external": as_bool(payload.get("follow_external"), False),
        "respect_robots": not as_bool(payload.get("ignore_robots"), False),
        "human_only": as_bool(payload.get("human_only"), True),
        "training_mode": as_bool(payload.get("training_mode"), True),
        "blocked_words": normalize_filter_values(payload.get("blocked_words")),
        "blocked_sites": normalize_block_sites(payload.get("blocked_sites")),
        "code_only": as_bool(payload.get("code_only"), False),
    }


def page_html() -> bytes:
    return UI_HTML.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "LosslessWebArchiveUI/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        # The browser log is the useful log; keep the terminal quiet.
        return

    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            data = page_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/state":
            self.send_json(JOB.snapshot())
            return
        if parsed.path == "/api/search":
            query = urllib.parse.parse_qs(parsed.query).get("q", [""])[0].strip()
            if not query:
                self.send_json({"error": "Enter a search query"}, 400)
                return
            try:
                limit = bounded_int(urllib.parse.parse_qs(parsed.query).get("limit", [8])[0], 8, 1, MAX_SEARCH_RESULTS)
                timeout = bounded_float(urllib.parse.parse_qs(parsed.query).get("timeout", [20])[0], 20, 1, 120)
                self.send_json(run_search(query, limit, timeout))
            except Exception as exc:
                self.send_json({"error": f"Search failed: {type(exc).__name__}: {exc}"}, 502)
            return
        self.send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length > 1_000_000:
            self.send_json({"error": "Request is too large"}, 413)
            return
        try:
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"error": "Request body must be JSON"}, 400)
            return
        if not isinstance(payload, dict):
            self.send_json({"error": "Request body must be a JSON object"}, 400)
            return
        if parsed.path == "/api/start":
            try:
                config = parse_start_config(payload)
                JOB.start(config)
            except (ValueError, RuntimeError) as exc:
                self.send_json({"error": str(exc)}, 409 if isinstance(exc, RuntimeError) else 400)
                return
            self.send_json({"ok": True, "state": JOB.snapshot()})
            return
        if parsed.path == "/api/stop":
            stopped = JOB.stop()
            self.send_json({"ok": True, "stop_requested": stopped})
            return
        self.send_json({"error": "Not found"}, 404)


UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lossless Web Archive</title>
<style>
:root { color-scheme: dark; --bg:#0d1117; --panel:#161b22; --line:#30363d; --muted:#9da7b3; --text:#e6edf3; --good:#3fb950; --warn:#d29922; --bad:#f85149; --blue:#58a6ff; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; }
.wrap { max-width:1280px; margin:0 auto; padding:24px; }
h1 { margin:0 0 4px; font-size:27px; }
h2 { margin:0 0 14px; font-size:18px; }
p { color:var(--muted); margin:5px 0 0; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:18px; margin:16px 0; }
.row { display:flex; flex-wrap:wrap; gap:12px; align-items:end; }
.grow { flex:1 1 420px; }
label { display:block; color:var(--muted); font-size:12px; }
input, textarea, button { font:inherit; }
input[type=text], input[type=number], textarea { width:100%; color:var(--text); background:#0d1117; border:1px solid var(--line); border-radius:6px; padding:9px 10px; margin-top:5px; }
textarea { min-height:75px; resize:vertical; }
input[type=checkbox] { accent-color:var(--blue); margin-right:7px; }
button { border:1px solid var(--line); background:#21262d; color:var(--text); border-radius:6px; padding:9px 13px; cursor:pointer; }
button.primary { background:#238636; border-color:#2ea043; }
button.danger { background:#8b1e24; border-color:#f85149; }
button:disabled { opacity:.5; cursor:wait; }
.checks { display:flex; flex-wrap:wrap; gap:16px; margin-top:12px; }
.checks label { font-size:13px; color:var(--text); }
.notice { color:var(--muted); border-left:3px solid var(--warn); padding:8px 10px; margin:12px 0; background:#1c1910; }
.error { color:var(--bad); }
.search-result { border-top:1px solid var(--line); padding:12px 0; display:grid; grid-template-columns:auto 1fr auto; gap:10px; align-items:start; }
.search-result:first-child { border-top:0; }
.search-result a { color:var(--blue); word-break:break-all; }
.small { color:var(--muted); font-size:12px; }
.pill { display:inline-block; border-radius:999px; padding:2px 8px; margin-left:5px; font-size:11px; border:1px solid var(--line); }
.pill.good { color:var(--good); border-color:#238636; }
.pill.warn { color:var(--warn); border-color:#9e6a03; }
.pill.bad { color:var(--bad); border-color:#da3633; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:10px; }
.card { border:1px solid var(--line); border-radius:8px; padding:11px; }
.card .value { font-size:20px; font-weight:650; margin-top:3px; }
.card .name { color:var(--muted); font-size:12px; }
.table-wrap { overflow:auto; max-height:430px; border:1px solid var(--line); border-radius:7px; }
table { width:100%; border-collapse:collapse; min-width:760px; }
th,td { text-align:left; padding:8px 9px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--muted); font-size:12px; position:sticky; top:0; background:var(--panel); }
td.url { max-width:360px; word-break:break-all; }
.status { margin-left:8px; color:var(--muted); }
@media (max-width:700px) { .wrap { padding:14px; } .search-result { grid-template-columns:auto 1fr; } .search-result .right { grid-column:2; } }
</style>
</head>
<body>
<main class="wrap">
  <header>
    <h1>Lossless Web Archive <span id="state" class="status">idle</span></h1>
    <p>Search, review source signals, and append complete extracted text to a local ZIP archive.</p>
  </header>

  <section class="panel">
    <h2>1. Search and check sources</h2>
    <div class="row">
      <label class="grow">Search query or deep-site URL
        <input id="query" type="text" placeholder="e.g. history of genocide, or a Software Heritage search URL">
      </label>
      <button id="search">Search web</button>
    </div>
    <div class="notice">Source screening is a heuristic. It rejects explicit AI-generation disclosures and accepts authorship/publication signals or a known public-archive provenance signal; no webpage can prove that AI was never used. Enter a topic to discover sources automatically, or paste seeds directly. Fetch workers run concurrently while the host delay spaces requests to the same site. Software Heritage directories are expanded through their API, so nested repository files count as one logical depth.</div>
    <div id="searchMessage" class="small"></div>
    <div id="results"></div>
  </section>

  <section class="panel">
    <h2>2. Crawl selected pages</h2>
    <label class="grow">Topic / gathering goal (optional; sources are discovered automatically)
      <input id="topic" type="text" placeholder="e.g. causes of ocean pollution, or Python reinforcement learning">
    </label>
    <label>Seeds (one http(s) URL per line; checked search results are added automatically)
      <textarea id="seeds" placeholder="https://example.org/article"></textarea>
    </label>
    <div class="row" style="margin-top:10px;align-items:start">
      <label class="grow">Block words or phrases (one per line or comma-separated)
        <textarea id="blockedWords" placeholder="login\nsign up\nsubscribe"></textarea>
      </label>
      <label class="grow">Block sites or domain paths (one per line or comma-separated)
        <textarea id="blockedSites" placeholder="facebook.com\nexample.org/login"></textarea>
      </label>
    </div>
    <div class="row" style="margin-top:12px">
      <label class="grow">Archive path<input id="archive" type="text" value="knowledge.zip"></label>
      <label>Max pages<input id="maxPages" type="number" min="1" value="500"></label>
      <label>Max depth<input id="maxDepth" type="number" min="0" value="2"></label>
      <label>Workers<input id="workers" type="number" min="1" max="64" value="5"></label>
      <label>Host delay (seconds)<input id="delay" type="number" min="0" step="0.01" value="0.05"></label>
      <label>Timeout (seconds)<input id="timeout" type="number" min="1" value="20"></label>
    </div>
    <div class="checks">
      <label><input id="humanOnly" type="checkbox" checked> Require human or archive provenance</label>
      <label><input id="trainingMode" type="checkbox" checked> Training mode (clean text and skip low-value files)</label>
      <label><input id="codeOnly" type="checkbox"> Code-bearing pages only</label>
      <label><input id="followExternal" type="checkbox"> Follow external domains</label>
    </div>
    <div class="notice">Block lists are applied before requests when a URL matches, and again to the fetched title/text. Code-only mode stores recognized source files and raw/blob pages; repository directory and login pages may be traversed for links but are never stored.</div>
    <div style="margin-top:14px">
      <button id="start" class="primary">Gather automatically</button>
      <button id="stop" class="danger">Stop</button>
      <span id="startMessage" class="status"></span>
    </div>
  </section>

  <section class="panel">
    <h2>Live status</h2>
    <div class="cards">
      <div class="card"><div class="name">Pages checked</div><div id="checked" class="value">0</div></div>
      <div class="card"><div class="name">Stored</div><div id="stored" class="value">0</div></div>
      <div class="card"><div class="name">Filtered</div><div id="filtered" class="value">0</div></div>
      <div class="card"><div class="name">Training filtered</div><div id="trainingFiltered" class="value">0</div></div>
      <div class="card"><div class="name">Code filtered</div><div id="codeFiltered" class="value">0</div></div>
      <div class="card"><div class="name">Blocked</div><div id="blocked" class="value">0</div></div>
      <div class="card"><div class="name">Duplicates</div><div id="duplicates" class="value">0</div></div>
      <div class="card"><div class="name">Source bytes</div><div id="sourceBytes" class="value">0 B</div></div>
      <div class="card"><div class="name">Archive size</div><div id="archiveBytes" class="value">0 B</div></div>
    </div>
    <p id="current" style="word-break:break-all;margin-top:12px"></p>
  </section>

  <section class="panel">
    <h2>Site totals</h2>
    <div class="table-wrap"><table><thead><tr><th>Site</th><th>Pages</th><th>Stored</th><th>Filtered</th><th>Source bytes</th><th>Text bytes</th><th>Compressed bytes</th></tr></thead><tbody id="sites"></tbody></table></div>
  </section>

  <section class="panel">
    <h2>Fetch log</h2>
    <div class="table-wrap"><table><thead><tr><th>Time</th><th>Site</th><th>Status</th><th>Source</th><th>Text</th><th>Compressed</th><th>Details</th><th>URL</th></tr></thead><tbody id="logs"></tbody></table></div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = value => { const n=Number(value||0); if(n<1024)return `${Math.round(n)} B`; const u=['KB','MB','GB','TB']; let x=n/1024,i=0; while(x>=1024&&i<u.length-1){x/=1024;i++;} return `${x.toFixed(1)} ${u[i]}`; };
let lastResults = [];

async function api(path, options={}) {
  const response = await fetch(path, {cache:'no-store', ...options, headers:{'Content-Type':'application/json', ...(options.headers||{})}});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function verdictPill(verdict) {
    const cls = (verdict === 'human_signals' || verdict === 'archive_signals') ? 'good' : verdict === 'ai_flagged' ? 'bad' : 'warn';
  return `<span class="pill ${cls}">${esc(verdict || 'unknown')}</span>`;
}

function renderResults(data) {
  lastResults = data.results || [];
    const deep = data.provider === 'software_heritage' ? `Software Heritage API: ${data.deep_search?.returned || 0} of ${data.deep_search?.total_count || 0} origin(s) returned. ` : '';
    $('searchMessage').textContent = `${deep}Checked ${lastResults.length} result(s) at ${data.checked_at || ''}. Select pages to use as seeds.`;
  if (!lastResults.length) { $('results').innerHTML = '<div class="small">No results returned.</div>'; return; }
  $('results').innerHTML = lastResults.map((r,i) => {
    const v = r.verdict || 'unknown';
     const verification = r.verification || {};
     const training = r.training || {};
     const ai = (verification.ai_signals || []).join(', ') || 'none detected';
     const human = (verification.human_signals || []).join(', ') || 'none detected';
     const archive = (verification.archive_signals || []).join(', ') || 'none detected';
     const origin = r.origin_url ? `<br><span class="small">Origin: ${esc(r.origin_url)}</span>` : '';
     const snapshot = r.snapshot_id ? ` · snapshot ${esc(String(r.snapshot_id).slice(0, 12))}` : '';
     const trainingState = training.include === false ? `training filter: ${(training.reasons || []).join(', ') || 'low value'}` : `training score: ${training.score ?? 'n/a'}/100`;
     return `<div class="search-result">
       <input class="result-check" data-index="${i}" type="checkbox" ${(v === 'human_signals' || v === 'archive_signals') && training.include !== false ? 'checked' : ''}>
       <div><strong>${esc(r.title || '(untitled)')}</strong> ${verdictPill(v)}${snapshot}<br><a href="${esc(r.url)}" target="_blank" rel="noreferrer">${esc(r.url)}</a>${origin}<div class="small">${esc(verification.reason || '')}<br>AI signals: ${esc(ai)} · Human signals: ${esc(human)} · Archive signals: ${esc(archive)}</div></div>
      <div class="small right">${esc(r.status)}<br>${esc(trainingState)}<br>${esc(r.content_kind || 'html')} · ${esc(r.language || 'und')}<br>${fmt(r.source_bytes)} source<br>${fmt(r.text_bytes)} text</div>
    </div>`;
  }).join('');
}

$('search').onclick = async () => {
  const query = $('query').value.trim();
  if (!query) return;
  $('search').disabled = true; $('searchMessage').textContent = 'Searching and checking sources…';
  try { renderResults(await api(`/api/search?q=${encodeURIComponent(query)}&limit=10`)); }
  catch (e) { $('searchMessage').innerHTML = `<span class="error">${esc(e.message)}</span>`; }
  finally { $('search').disabled = false; }
};

$('start').onclick = async () => {
  const chosen = [...document.querySelectorAll('.result-check:checked')].map(box => lastResults[Number(box.dataset.index)]?.url).filter(Boolean);
  const manual = $('seeds').value.split(/\r?\n/).map(x=>x.trim()).filter(Boolean);
  const seeds = [...new Set([...manual, ...chosen])];
  const topic = $('topic').value.trim();
  if (!seeds.length && !topic) { $('startMessage').innerHTML = '<span class="error">Enter a topic, seed URL, or select a search result.</span>'; return; }
  const body = {topic, seeds, blocked_words:$('blockedWords').value, blocked_sites:$('blockedSites').value, code_only:$('codeOnly').checked, discover_limit:20, workers:$('workers').value, archive:$('archive').value.trim(), max_pages:$('maxPages').value, max_depth:$('maxDepth').value, delay:$('delay').value, timeout:$('timeout').value, human_only:$('humanOnly').checked, follow_external:$('followExternal').checked, training_mode:$('trainingMode').checked};
  $('start').disabled = true; $('startMessage').textContent = 'Starting…';
  try { await api('/api/start',{method:'POST',body:JSON.stringify(body)}); $('startMessage').textContent='Running'; }
  catch (e) { $('startMessage').innerHTML = `<span class="error">${esc(e.message)}</span>`; }
  finally { $('start').disabled = false; }
};
$('stop').onclick = async () => { try { const r=await api('/api/stop',{method:'POST',body:'{}'}); $('startMessage').textContent=r.stop_requested?'Stop requested':'No crawl is running'; } catch(e) { $('startMessage').textContent=e.message; } };

function renderState(s) {
  $('state').textContent = s.state || 'idle';
  const c=s.counters||{};
  $('checked').textContent=(c.checked||0).toLocaleString(); $('stored').textContent=(c.stored||0).toLocaleString(); $('filtered').textContent=(c.filtered||0).toLocaleString(); $('trainingFiltered').textContent=(c.training_filtered||0).toLocaleString(); $('codeFiltered').textContent=(c.code_filtered||0).toLocaleString(); $('blocked').textContent=(c.blocked||0).toLocaleString(); $('duplicates').textContent=(c.duplicates||0).toLocaleString(); $('sourceBytes').textContent=fmt(c.source_bytes); $('archiveBytes').textContent=fmt(s.archive_bytes);
  $('current').textContent=s.current ? `Current: ${s.current}` : (s.archive ? `Archive: ${s.archive}` : '');
  const sites=Object.entries(s.sites||{}).sort((a,b)=>(b[1].source_bytes||0)-(a[1].source_bytes||0));
  $('sites').innerHTML=sites.map(([name,v])=>`<tr><td>${esc(name)}</td><td>${v.pages||0}</td><td>${v.stored||0}</td><td>${v.filtered||0}</td><td>${fmt(v.source_bytes)}</td><td>${fmt(v.text_bytes)}</td><td>${fmt(v.compressed_bytes)}</td></tr>`).join('') || '<tr><td colspan="7" class="small">No pages checked yet.</td></tr>';
  const logs=(s.logs||[]).slice().reverse();
  $('logs').innerHTML=logs.map(l=>`<tr><td>${esc(l.time||'')}</td><td>${esc(l.site||'')}</td><td>${verdictPill(l.status||'')}</td><td>${fmt(l.source_bytes)}</td><td>${fmt(l.text_bytes)}</td><td>${fmt(l.compressed_bytes)}</td><td>${esc([l.method,l.topic,l.details].filter(Boolean).join(' · '))}</td><td class="url">${esc(l.url||'')}</td></tr>`).join('') || '<tr><td colspan="8" class="small">No fetches yet.</td></tr>';
}

async function poll() { try { renderState(await api('/api/state')); } catch(e) { $('state').textContent='offline'; } }
poll(); setInterval(poll, 1000);
</script>
</body>
</html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local UI for the lossless web archive")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Lossless Web Archive UI: http://{args.host}:{args.port}/")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping UI server.")
    finally:
        JOB.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

