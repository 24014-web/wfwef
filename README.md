# Local Opinion AI prototype

This is a small, dependency-free prototype for the idea discussed in this
conversation: a local process that reads selected public webpages, stores a
private corpus, retrieves evidence, and records a working position about a
topic.

It is intentionally **not** a language model. A language model needs learned
weights; this prototype uses a transparent extractive algorithm so it can run
with only Python's standard library. It does not contain a moral worldview or
an Asimov-style rule set. It selects a representative sentence from the
evidence that has been stored and shows the sources used.

The prototype has technical boundaries: it only performs explicit read-only
GET requests, never executes text as code, never posts or sends messages, and
does not modify its own source. These boundaries keep an experiment from
becoming an uncontrolled program while leaving its recorded conclusions open
to whatever the evidence supports.

## Run it

```text
python local_opinion_ai.py init
python local_opinion_ai.py ingest https://example.com/article
python local_opinion_ai.py search "wars genocide" --limit 8
python local_opinion_ai.py review humanity --save
python local_opinion_ai.py beliefs
python local_opinion_ai.py ask "humanity and technology"
python local_opinion_ai.py chat
```

The `search` command uses a simple public HTML search endpoint from the
standard library, prints the result URLs, and ingests them by default. Add
`--no-ingest` when you only want discovery. You can ingest a local `.txt` or
`.html` file too. The journal is stored in `.local_opinion_ai/` by default. Set
`LOCAL_OPINION_HOME` or pass `--home` to put it somewhere else.

## Compact web archives

For broad information gathering, `compact_web_scraper.py` follows links from
explicit seed URLs, extracts readable text, selects high-information sentences,
removes repeated sentences across pages, and stores the result in a single
LZMA-compressed `.lca` archive. It keeps summaries and provenance rather than
the original HTML, which is what makes a strict storage budget practical.

```text
python compact_web_scraper.py scrape https://example.com/start \
  --max-pages 1000 --max-depth 2 --max-bytes 10000000 --output knowledge.lca
python compact_web_scraper.py inspect knowledge.lca
python compact_web_scraper.py query knowledge.lca "war genocide"
```

The crawler is bounded by page count, depth, response size, delay, and archive
size. It performs read-only GET requests and honors `robots.txt` by default.
These are crawler controls, not an opinion or moral policy. A seed list can
contain multiple domains; add `--follow-external` when links should cross from
one seed domain to another.

## Full-text training archive

When summaries are not wanted, `lossless_web_archive.py` keeps the extracted
source text without rewriting sentences. Training mode is on by default: it
selects semantic article bodies, unwraps archived source files, extracts
notebook markdown/code/output cells, fixes common legacy encodings, and skips
obvious page chrome, binary assets, generated lock files, and archive directory
listings. Every record gets a quality score, content kind, language hint,
source byte count, and the reasons a low-value page was skipped. Exact URL and
text duplicates are stored once to avoid wasting training storage.

```text
python lossless_web_archive.py add https://example.com/article --archive knowledge.zip
python lossless_web_archive.py crawl https://example.com --archive knowledge.zip \
  --max-pages 1000 --max-depth 2
python lossless_web_archive.py crawl --topic "causes of ocean pollution" \
  --archive ocean.zip --workers 5 --max-pages 500
python lossless_web_archive.py inspect knowledge.zip
python lossless_web_archive.py verify knowledge.zip
python lossless_web_archive.py export knowledge.zip training.jsonl
python lossless_web_archive.py export knowledge.zip training.jsonl --training-only
```

`training.jsonl` contains the full extracted text plus URL, title, labels,
retrieval time, encoding, quality metadata, and integrity hash. The ZIP is
lossless relative to the selected extraction transform: training mode removes
navigation and embedded media by design, while `--raw-visible` preserves the
complete visible DOM text for a compatibility/archive copy. It does not
summarize or invent sentences. `--training-only` also upgrades and filters
older archives that predate the training metadata.
Appending opens and closes the ZIP for each page so a successfully added page
is immediately readable. Pre-compressing text before putting it in ZIP would
usually save less space because ZIP would no longer see the original patterns.

### Autonomous, concurrent gathering

`crawl --topic` turns a short goal into ranked, diverse search seeds, then
continues through each page's links. The crawler uses a bounded worker pool
(`--workers`, default 5), reserves request slots per host, and writes completed
records as soon as they finish. `--delay` is the minimum spacing between
requests to one host; different sites proceed concurrently. Throughput is
reported at the end and depends on network latency, robots rules, and the
servers being contacted, so five workers is a concurrency target rather than a
guaranteed five pages every second. Crawls use fast DEFLATE by default; the
older `LosslessArchive(..., compression="auto")` path remains available when
minimum archive size matters more than speed.

### Deep sites and Software Heritage

The lossless crawler follows ordinary HTML links and has a built-in adapter for
Software Heritage origin-search URLs such as:

```text
python lossless_web_archive.py crawl \
  "https://archive.softwareheritage.org/browse/search/?q=AI&with_visit=true&with_content=true" \
  --archive software-heritage.zip --max-pages 500 --max-depth 3
```

That page renders its result table with JavaScript. The adapter calls the
site's public JSON origin-search endpoint, puts the returned archive-browser
links first, and records the exact API fields as `deep_search` metadata. When a
crawl opens an origin directory, it resolves the directory identifier through
Software Heritage's JSON directory API and walks nested directories to produce
direct links for every archived file. Source pages are then fetched through the
raw content endpoint, so the ZIP contains code/data text rather than the
archive's directory names and navigation. If that raw endpoint is unavailable,
training mode records the failure and skips the browser shell rather than
mistaking it for source. Paste the same URL into the UI's
**Search query or deep-site URL** field to review and select archive origins.
Archived files whose MIME type is `text/html` are treated as source files when
they come from that raw endpoint, so their tags and templates are preserved.

The archive tree counts as one logical crawl level, so the default **Max depth**
of 2 reaches files even when they are several folders deep. Keep **Follow
external domains** off to stay inside the Software Heritage archive. Turn it on
only when you intentionally want to fetch the live origin hosts too. The
crawler still honors `robots.txt`, page and response-size limits, and the
optional human-source screening heuristic.
JavaScript-only sites without a known adapter are identified as shells; the
standard-library crawler does not execute arbitrary page scripts, bypass
logins/CAPTCHAs, or turn binary downloads into text. Images are therefore
skipped in training mode; extracting code that exists only inside pixels would
require adding an OCR engine separately.

## Browser UI, search, and source checks

`crawl_ui.py` provides a local browser interface for the full-text crawler. It
shows search results, checks each result before crawling it, and keeps a live
log with the URL, site, downloaded bytes, extracted text bytes, compression
method, stored size, content kind, and training score. The **Training mode**
checkbox controls the cleaning and quality gate; it is enabled by default.

```text
python crawl_ui.py
```

Open `http://127.0.0.1:8765/` and use **Search web** to discover candidate
pages. Checked results can be selected as crawl seeds. **Require human or
archive provenance** is enabled by default: pages with explicit AI-generation disclosures
are rejected, and pages without author/publication signals or a known public
archive provenance are marked unknown and skipped. Software Heritage pages
carry a separate `archive_signals` label because the archive can be identified
without claiming that archived code was human-written. This is a screening
heuristic, not proof that a page was written entirely by a person; the UI
displays the signals it found so the choice is reviewable. Uncheck it when you
want to collect unknown sources too.

## What it can and cannot do

It can build a local, inspectable evidence journal and revise the journal when
you run a later review. It cannot understand meaning like a trained LLM,
verify that a webpage is true, crawl the entire internet, or experience
gratitude, hatred, or any other subjective feeling. Those capabilities would
require a trained model, a source-verification strategy, and a much larger
agent design.

