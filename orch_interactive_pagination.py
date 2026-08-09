"""Page-by-page pagination orchestrator (interactive, multi-mode).

Processes articles page-by-page instead of collecting all links upfront.
Page 1 establishes clusters and generates extraction code via LLM.
Subsequent pages reuse all generated code — new LLM calls only happen
for genuinely novel article structures.

Pagination is user-driven: the user is asked how the listing is paginated
and picks one of four modes —
  1. Numbered pagination — provide page 1 and page 2 URLs; the URL pattern
     is derived by diffing them (string diff first, LLM fallback if needed).
  2. No pagination — a single page only.
  3. Infinite scroll — content loads as the page is scrolled.
  4. Load more button — content loads when a button is clicked.

LLM call budget:
  0-1  — pagination pattern (only if string diff fails)
  1    — link extraction code (Links Agent)
  N    — one per unique article-structure cluster (page 1 establishes most)
  +    — extra calls only if a later page surfaces a new structure
"""

import asyncio
import json
import os
import subprocess
import sys
import hashlib
import time
import tempfile
import traceback
from collections import defaultdict
from datetime import datetime
import re as _re
from urllib.parse import urljoin, urlparse

import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# Reuse utilities from Agent_for_single_page_gemma
from Agent_for_single_page_gemma import (
    fetch_page_structure,
    execute_extraction_code,
    create_structural_map as _sp_create_structural_map,
)

# Shared core: N/A + token helpers, model listing, and the challenge detector
# used so the fast requests path rejects the same bot/challenge pages the
# browser path does.
from utils import (
    is_na_value, token_usage_from_response, list_available_models,
    _is_challenge_page as _cf_is_challenge_page,
)

# --- Configuration ---
LINKS_AGENT_SCRIPT = "Links_Agent_gemma_cloudflare.py" #"Links_Agent_gemma.py"
AGENT_SCRIPT = "Agent_for_single_page_gemma.py"

# Input-token budget for the Links agent's single per-run code-generation call.
# Kept just under the gemma free-tier per-minute input cap (16,000 tokens for (gemma-4-31b) so a large listing map doesn't trigger 429 quota errors. Raise this only on a paid tier / higher-quota key.
LINKS_INPUT_TOKEN_BUDGET = 15000

# Gemma 4 is currently free within this call allowance. The orchestrator tracks
# calls per run, not account-wide quota consumption.
GEMMA_FREE_TIER_CALL_LIMIT = 1000
GEMMA_FREE_TIER_MODEL_PREFIXES = ("gemma-4",)

# Fallback price per 1M tokens (USD) for models outside the Gemma 4 allowance.
LLM_PRICE_PER_1M_INPUT = 0.075
LLM_PRICE_PER_1M_OUTPUT = 0.30


def _uses_gemma_free_tier(model, llm_calls):
    """Return whether this run fits the configured Gemma 4 free allowance."""
    normalized_model = str(model or "").removeprefix("models/").lower()
    return (
        normalized_model.startswith(GEMMA_FREE_TIER_MODEL_PREFIXES)
        and 0 <= int(llm_calls or 0) <= GEMMA_FREE_TIER_CALL_LIMIT
    )

# Running count of LLM generation calls made during a run (links agent,
# per-cluster article agent, and pagination-pattern LLM fallback). Used for
# the stats written to results.md. Reset at the start of each main() run.
_LLM_CALLS = 0

# Breakdown of LLM calls by which agent / purpose triggered them, so results.md
# can show e.g. "4 LLM calls: 3 Links agent, 1 Article agent".
_LLM_CALLS_BY_AGENT = {}


def _reset_llm_calls():
    global _LLM_CALLS, _LLM_CALLS_BY_AGENT
    _LLM_CALLS = 0
    _LLM_CALLS_BY_AGENT = {}


def _bump_llm_calls(n=1, agent="Other"):
    global _LLM_CALLS
    _LLM_CALLS += n
    _LLM_CALLS_BY_AGENT[agent] = _LLM_CALLS_BY_AGENT.get(agent, 0) + n


# Token usage aggregated across every agent/LLM call in a run, broken down by
# which agent spent them. Reset at the start of each main() run.
_TOKENS_BY_AGENT = {}


def _reset_tokens():
    global _TOKENS_BY_AGENT
    _TOKENS_BY_AGENT = {}


def _bump_tokens(agent, usage):
    """Add a {prompt, candidates, total} usage dict to *agent*'s running total."""
    if not usage:
        return
    slot = _TOKENS_BY_AGENT.setdefault(
        agent, {"prompt": 0, "candidates": 0, "total": 0})
    slot["prompt"] += usage.get("prompt", 0) or 0
    slot["candidates"] += usage.get("candidates", 0) or 0
    slot["total"] += usage.get("total", 0) or 0


# How each article's HTML was ultimately fetched. Reset at the start of each
# main() run and reported in results.md so we can see how often the fast
# plain-requests path was used vs. the slower browser fallback.
_FETCH_VIA = {"requests": 0, "browser": 0}


def _reset_fetch_via():
    global _FETCH_VIA
    _FETCH_VIA = {"requests": 0, "browser": 0}


def _bump_fetch_via(via):
    if via in _FETCH_VIA:
        _FETCH_VIA[via] += 1


# Full structural-map depth an agent uses before any token-budget shrinking.
# Mirrors MAX_DEPTH in the agents; used only to flag reduced-depth runs.
_FULL_MAP_DEPTH = 10

# Per-agent-call record of how the prompt fit the token budget: the map depth
# actually sent and the measured input-token size. Reported in results.md and
# reset at the start of each main() run.
_MAP_FITS = []

# Per-run audit trail for generated Article Agent code. This catches the case
# where one bad extractor would otherwise be reused across a whole cluster.
_ARTICLE_VALIDATION_EVENTS = []


def _reset_map_fits():
    global _MAP_FITS
    _MAP_FITS = []


def _reset_article_validation_events():
    global _ARTICLE_VALIDATION_EVENTS
    _ARTICLE_VALIDATION_EVENTS = []


def _record_article_validation_event(event):
    if isinstance(event, dict):
        _ARTICLE_VALIDATION_EVENTS.append(event)


def _bump_map_fit(agent, result):
    """Record the map depth + input-token size an agent reported (if any)."""
    if not isinstance(result, dict):
        return
    depth = result.get("map_depth")
    if depth is None:
        return
    _MAP_FITS.append({
        "agent": agent,
        "depth": depth,
        "input_tokens": result.get("input_tokens"),
    })


# ═══════════════════════════════════════════════════════════════
# Utilities carried over from orch.py
# ═══════════════════════════════════════════════════════════════

# How deep the cluster signature inspects the structural map. Going a few levels
# deep lets genuinely different content templates (e.g. paragraph-based vs
# list-based article bodies) form separate clusters instead of colliding.
_SIG_MAX_DEPTH = 5


def _struct_signature(smap, depth=0, max_depth=_SIG_MAX_DEPTH):
    """Build a hashable string representing the skeleton of a structural map.

    Repeated sibling patterns are collapsed (count-insensitive) so pages that
    differ only in how many times a child repeats — e.g. 5 vs 20 paragraphs, or
    a different number of list items — stay in the same cluster, while a
    different *kind* of child (a <p> body vs a <ul>/<li> body) still separates
    them.
    """
    if depth >= max_depth or not isinstance(smap, list):
        return ""
    parts = []
    for node in smap:
        tag = node.get("tag", "")
        cls = node.get("attributes", {}).get("class", "")
        children_sig = _struct_signature(node.get("children", []), depth + 1, max_depth)
        parts.append(f"{tag}.{cls}({children_sig})")
    # Collapse duplicate sibling patterns so the signature ignores repeat counts.
    unique = list(dict.fromkeys(parts))
    return "|".join(unique)


def _sig_hash(smap):
    """Return the 12-char MD5 cluster key for a structural map."""
    sig = _struct_signature(smap)
    return hashlib.md5(sig.encode()).hexdigest()[:12]


def cluster_by_structure(articles_with_maps):
    """Group articles by structural-map similarity."""
    clusters = defaultdict(list)
    for item in articles_with_maps:
        clusters[_sig_hash(item["structural_map"])].append(item)
    return dict(clusters)


def _call_agent_subprocess(cmd, timeout=300):
    """Run a CLI agent subprocess, streaming its stdout live and parsing the
    final ORCH_RESULT line.

    stdout is echoed line-by-line (prefixed with '│ ') so token counts, map-depth
    shrink attempts and progress are visible in real time instead of only after
    the subprocess finishes. stderr is drained on a background thread to avoid a
    full-pipe deadlock, and a watchdog timer enforces *timeout*.
    """
    import threading

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n❌ Subprocess exception:\n{tb}")
        return False, {"status": "error", "error": f"{e}\n{tb}"}

    timed_out = {"flag": False}
    watchdog = threading.Timer(
        timeout, lambda: (timed_out.__setitem__("flag", True), proc.kill()))
    watchdog.start()

    stderr_chunks = []

    def _drain_stderr():
        try:
            for line in proc.stderr:
                stderr_chunks.append(line)
        except Exception:
            pass

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    stdout_lines = []
    result = None
    ok = False
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            stdout_lines.append(line)
            if line.startswith("ORCH_RESULT:"):
                try:
                    result = json.loads(line[len("ORCH_RESULT:"):])
                    ok = result.get("status") == "ok"
                except Exception:
                    result = None
            else:
                print(f"    │ {line}", flush=True)  # live echo from the agent
    finally:
        proc.wait()
        watchdog.cancel()
        stderr_thread.join(timeout=1)

    full_stderr = "".join(stderr_chunks)
    full_stdout = "\n".join(stdout_lines)

    if timed_out["flag"]:
        return False, {"status": "error", "error": f"Subprocess timed out ({timeout}s)"}

    if result is not None:
        if not ok:
            # Surface stderr so the real cause is visible (stdout was streamed above)
            print(f"\n{'─' * 40} SUBPROCESS STDERR {'─' * 40}")
            print(full_stderr or "(empty)")
            print(f"{'─' * 98}")
        return ok, result

    print(f"\n{'─' * 40} SUBPROCESS STDERR {'─' * 40}")
    print(full_stderr or "(empty)")
    print(f"{'─' * 98}")
    return False, {
        "status": "error",
        "error": f"No ORCH_RESULT in output. stderr: {full_stderr}  stdout(tail): {full_stdout[-3000:]}",
    }

def call_links_agent_cli(url, api_key, model="gemma-3-27b-it", html_file=None):
    """Call Links_Agent_gemma_cloudflare.py via subprocess in CLI mode.

    When *html_file* is given, the agent analyzes that pre-fetched HTML instead
    of fetching *url* live — used for infinite-scroll / load-more listings so it
    sees the fully-loaded grid rather than the pre-scroll skeleton.
    """
    cmd = [
        sys.executable, LINKS_AGENT_SCRIPT,
        "--url", url,
        "--api-key", api_key,
        "--model", model,
        "--max-input-tokens", str(LINKS_INPUT_TOKEN_BUDGET),
    ]
    if html_file:
        cmd += ["--html-file", html_file]
    print(f"  🔧 Calling: python {LINKS_AGENT_SCRIPT} --url {url[:80]}...")
    _bump_llm_calls(agent="Links agent")
    # The Links agent can make up to 3 generation attempts (initial + 2 retries),
    # each a slow free-tier Gemma call with rate-limit backoff, so give it more
    # headroom than the article agent to avoid killing it mid-retry.
    ok, result = _call_agent_subprocess(cmd, timeout=600)
    _bump_tokens("Links agent", result.get("token_usage"))
    _bump_map_fit("Links agent", result)
    return ok, result


def call_agent_cli(url, api_key, requirements, model="gemma-3-27b-it", html_content=None):
    """Call Agent_for_single_page_gemma.py via subprocess in CLI mode.

    When *html_content* is provided, it is written to a temp file and passed to
    the agent so its structural map is built from the exact HTML the extraction
    code will run against. This avoids the map/extraction DOM mismatch where a
    JS-rendered map exposes dynamic classes (e.g. ``active``) that the static
    fetched HTML does not have, making the LLM pick selectors that match nothing.
    """
    cmd = [
        sys.executable, AGENT_SCRIPT,
        "--url", url,
        "--api-key", api_key,
        "--requirements", requirements,
        "--model", model,
    ]
    html_tmp = None
    if html_content:
        fd, html_tmp = tempfile.mkstemp(suffix=".html", prefix="agent_html_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(html_content)
        cmd += ["--html-file", html_tmp]
    print(f"  🔧 Calling: python {AGENT_SCRIPT} --url {url[:80]}...")
    _bump_llm_calls(agent="Article agent")
    try:
        ok, result = _call_agent_subprocess(cmd, timeout=300)
    finally:
        if html_tmp:
            try:
                os.remove(html_tmp)
            except OSError:
                pass
    _bump_tokens("Article agent", result.get("token_usage"))
    _bump_map_fit("Article agent", result)
    return ok, result


def _requested_field_names(requirements):
    """Best-effort parse of user-requested output fields."""
    text = str(requirements or "")
    text = _re.sub(r"^[\s'\"]*(extract|scrape|get|collect)\s+", "", text, flags=_re.I)
    text = text.strip(" .'\"")
    if not text:
        return []
    parts = _re.split(r",|;|\band\b|\n", text, flags=_re.I)
    fields = []
    for part in parts:
        field = part.strip(" -:'\"\t\r\n")
        field = _re.sub(r"^(the|a|an)\s+", "", field, flags=_re.I)
        if field:
            fields.append(field)
    return fields


def _canonical_field_name(name):
    norm = _re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()
    tokens = set(norm.split())
    if "title" in tokens or "headline" in tokens:
        return "title"
    if "author" in tokens or "writer" in tokens or "byline" in tokens:
        return "author"
    if "date" in tokens or "time" in tokens or "published" in tokens:
        return "date"
    if "body" in tokens or "content" in tokens or "text" in tokens or "article" in tokens:
        return "article body text"
    return norm


def _field_value_by_canonical(data, canonical):
    if not isinstance(data, dict):
        return None, None
    for key, value in data.items():
        if _canonical_field_name(key) == canonical:
            return key, value
    return None, None


def _canonicalize_output_keys(data, requirements):
    """Rename returned keys to the exact requested field names when they match
    by canonical name (e.g. `article_body_text` -> `article body text`).

    Keeps the extractor's original key for anything the user did not request,
    and prefers a non-N/A value when two keys collapse to the same field.
    """
    if not isinstance(data, dict):
        return data
    canon_to_requested = {}
    for field in _requested_field_names(requirements):
        canon = _canonical_field_name(field)
        canon_to_requested.setdefault(canon, field)
    renamed = {}
    for key, value in data.items():
        target = canon_to_requested.get(_canonical_field_name(key), key)
        if target in renamed:
            if is_na_value(renamed[target]) and not is_na_value(value):
                renamed[target] = value
        else:
            renamed[target] = value
    return renamed


def _iter_json_ld_objects(value):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_json_ld_objects(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_json_ld_objects(nested)


def _json_ld_author(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("name") or "").strip()
    if isinstance(value, list):
        names = [_json_ld_author(item) for item in value]
        return ", ".join(name for name in names if name)
    return ""


def _standard_article_metadata(html_content):
    """Extract domain-independent article fields from standard metadata."""
    soup = BeautifulSoup(html_content or "", "html.parser")
    article_objects = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(script.string or script.get_text(" ", strip=True))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        for obj in _iter_json_ld_objects(payload):
            raw_types = obj.get("@type", [])
            types = raw_types if isinstance(raw_types, list) else [raw_types]
            if any("article" in str(item).lower() or str(item).lower() == "blogposting"
                   for item in types):
                article_objects.append(obj)

    def meta_content(*selectors):
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                value = element.get("content") or element.get("datetime")
                if not value:
                    value = element.get_text(" ", strip=True)
                if value and str(value).strip():
                    return str(value).strip()
        return ""

    metadata = {
        "title": meta_content(
            'meta[property="og:title"]', 'meta[name="twitter:title"]',
            'meta[itemprop="headline"]',
        ),
        "date": meta_content(
            'meta[property="article:published_time"]',
            'meta[itemprop="datePublished"]', 'meta[name="date"]',
            'time[itemprop="datePublished"]', 'time[datetime]',
        ),
        "author": meta_content(
            'meta[name="author"]', 'meta[property="article:author"]',
            'meta[itemprop="author"]',
        ),
        "article body text": "",
    }
    for obj in article_objects:
        metadata["title"] = metadata["title"] or str(
            obj.get("headline") or obj.get("name") or ""
        ).strip()
        metadata["date"] = metadata["date"] or str(
            obj.get("datePublished") or obj.get("dateCreated") or ""
        ).strip()
        metadata["author"] = metadata["author"] or _json_ld_author(obj.get("author"))
        metadata["article body text"] = metadata["article body text"] or str(
            obj.get("articleBody") or ""
        ).strip()
    return metadata


def _apply_metadata_fallbacks(data, html_content, requirements):
    recovered = dict(data) if isinstance(data, dict) else {}
    metadata = None
    for requested in _requested_field_names(requirements):
        canonical = _canonical_field_name(requested)
        if canonical not in {"title", "date", "author", "article body text"}:
            continue
        actual_key, value = _field_value_by_canonical(recovered, canonical)
        if actual_key is not None and not is_na_value(value):
            continue
        if metadata is None:
            metadata = _standard_article_metadata(html_content)
        fallback = metadata.get(canonical)
        if fallback and not is_na_value(fallback):
            recovered[actual_key or requested] = fallback
    return _canonicalize_output_keys(recovered, requirements)


def _extract_article_fields(code, html_content, requirements):
    """Run generated extractor code, then canonicalize its output keys to the
    requested field names so naming drift never looks like missing data."""
    ok, result = execute_extraction_code(code, html_content)
    if ok and isinstance(result, dict):
        result = _canonicalize_output_keys(result, requirements)
        result = _apply_metadata_fallbacks(result, html_content, requirements)
    return ok, result


def _short_value(value, limit=120):
    if value is None:
        return "N/A"
    text = str(value).replace("\n", " ").strip()
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _validate_article_extraction(data, article, requirements):
    """Return validation details for one generated extractor output."""
    requested = _requested_field_names(requirements)
    requested_canon = []
    for field in requested:
        canon = _canonical_field_name(field)
        if canon and canon not in requested_canon:
            requested_canon.append(canon)

    issues = []
    missing = []
    wrong_key = []
    present = []
    for field in requested:
        canon = _canonical_field_name(field)
        actual_key, value = _field_value_by_canonical(data, canon)
        if actual_key is None:
            issues.append(f"missing requested field `{field}`")
            missing.append(field)
            continue
        if actual_key != field:
            wrong_key.append(f"`{field}` returned as `{actual_key}`")
        if is_na_value(value):
            issues.append(f"`{actual_key}` is N/A")
            missing.append(field)
        else:
            present.append(field)

    critical = bool(missing)

    title_hint = str((article or {}).get("title") or "").strip()
    if title_hint and not is_na_value(title_hint) and "title" in requested_canon:
        _key, value = _field_value_by_canonical(data, "title")
        if is_na_value(value):
            issues.append(f"listing title exists but extractor returned N/A: {_short_value(title_hint)}")
            critical = True

    return {
        # Key-name drift alone (schema_warnings) is not a failure: the value is
        # matched by canonical name and the keys are canonicalized before saving.
        "ok": not issues,
        "critical": critical,
        "issues": issues,
        "schema_warnings": wrong_key,
        "missing_fields": missing,
        "present_fields": present,
    }


def _validation_feedback(requirements, code_file, data, validation, attempt_label):
    pieces = [
        str(requirements or ""),
        "",
        "ARTICLE EXTRACTOR RETRY CONTEXT:",
        f"The previous generated code ({code_file or 'unknown code file'}) failed validation on {attempt_label}.",
        "The user requested these fields and does not want N/A for fields that are visible in the page.",
    ]
    problems = validation.get("issues") or []
    schema = validation.get("schema_warnings") or []
    if problems:
        pieces.append("Validation issues: " + "; ".join(problems))
    if schema:
        pieces.append("Schema warnings: " + "; ".join(schema))
    pieces.append("Previous output: " + json.dumps(data or {}, ensure_ascii=False)[:1200])
    pieces.append("Regenerate the extractor. Do not hard-code N/A unless the field is truly absent. Use only selectors present in the structural map, and return keys that match the requested field names exactly.")
    return "\n".join(pieces)


MAX_ARTICLE_EXTRACTOR_ATTEMPTS = 3


def _automatic_validation_decision(validation, attempts):
    """Return retry/skip for critical failures, or None for manual warnings."""
    if not validation.get("critical"):
        return None
    if attempts < MAX_ARTICLE_EXTRACTOR_ATTEMPTS:
        return "retry"
    return "skip"


def _prompt_article_validation_decision(sig, article, validation, code_file, sample=False):
    label = "sample article" if sample else "representative article"
    print(f"\n  ⚠ Article extractor validation warning for cluster {sig} ({label})")
    print(f"     URL: {article.get('url', 'N/A')}")
    print(f"     Title hint: {_short_value(article.get('title'))}")
    if code_file:
        print(f"     Code: {code_file}")
    for issue in validation.get("issues") or []:
        print(f"     - {issue}")
    for warning in validation.get("schema_warnings") or []:
        print(f"     - schema: {warning}")

    allow_piped_prompt = os.environ.get("ORCH_ALLOW_STDIN_PROMPTS") == "1"
    if not sys.stdin or (not sys.stdin.isatty() and not allow_piped_prompt):
        decision = "skip" if validation.get("critical") else "accept"
        print(f"     Non-interactive run: defaulting to {decision}.")
        return decision

    print("     Choose: [a] accept anyway, [r] retry/regenerate, [s] skip this cluster")
    choice = input("     → ").strip().lower()
    if choice in {"r", "retry"}:
        return "retry"
    if choice in {"s", "skip"}:
        return "skip"
    return "accept"


def _load_generated_code(code_file):
    if code_file and os.path.exists(code_file):
        with open(code_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# ═══════════════════════════════════════════════════════════════
# Pagination: derive URL pattern from two example URLs
# ═══════════════════════════════════════════════════════════════

def derive_pagination_pattern(url1, url2):
    """Compare page-1 and page-2 URLs to find the varying page number.

    Returns (pattern_with_{page}_placeholder, page1_num, page2_num)
    or (None, None, None) on failure.

    Tested examples:
        (".../regionAll", ".../regionAll?page=2")
            → (".../regionAll?page={page}", None, 2)
        (".../65/1", ".../65/2")
            → (".../65/{page}", 1, 2)
        (".../testimonies-from-the-war-ar/", ".../testimonies-from-the-war-ar/page/2/")
            → (".../testimonies-from-the-war-ar/page/{page}/", None, 2)
        ("...sectionid=10", "...sectionid=10&page=2")
            → ("...sectionid=10&page={page}", None, 2)
    """
    if url1 == url2:
        return None, None, None

    # ── Strategy 1: character-level diff ─────────────────────
    min_len = min(len(url1), len(url2))

    # Find first differing position
    prefix_end = 0
    while prefix_end < min_len and url1[prefix_end] == url2[prefix_end]:
        prefix_end += 1

    # Find last differing position (from the end)
    suffix_start_1 = len(url1)
    suffix_start_2 = len(url2)
    while (suffix_start_1 > prefix_end and suffix_start_2 > prefix_end
           and url1[suffix_start_1 - 1] == url2[suffix_start_2 - 1]):
        suffix_start_1 -= 1
        suffix_start_2 -= 1

    diff1 = url1[prefix_end:suffix_start_1]
    diff2 = url2[prefix_end:suffix_start_2]

    # Both differing segments are pure numbers — perfect match
    if diff1.isdigit() and diff2.isdigit():
        pattern = url2[:prefix_end] + "{page}" + url2[suffix_start_2:]
        return pattern, int(diff1), int(diff2)

    # url1 is a prefix of url2 (page 1 has no page indicator)
    if diff1 == "" and diff2 != "":
        nums = _re.findall(r'\d+', diff2)
        if nums:
            page_num = nums[-1]  # last number in the appended part
            # Replace only the page number, not other numbers in the diff
            idx = diff2.rfind(page_num)
            replaced = diff2[:idx] + "{page}" + diff2[idx + len(page_num):]
            pattern = url2[:prefix_end] + replaced + url2[suffix_start_2:]
            return pattern, None, int(page_num)

    # ── Strategy 2: regex — find rightmost number that differs ──
    nums_in_2 = list(_re.finditer(r'\d+', url2))
    nums_in_1 = list(_re.finditer(r'\d+', url1))

    for m2 in reversed(nums_in_2):
        val2 = m2.group()
        start2, end2 = m2.start(), m2.end()

        for m1 in nums_in_1:
            if m1.start() == start2 and m1.group() != val2:
                pattern = url2[:start2] + "{page}" + url2[end2:]
                return pattern, int(m1.group()), int(val2)

        if start2 >= len(url1) or url1[start2:end2] != val2:
            pattern = url2[:start2] + "{page}" + url2[end2:]
            return pattern, None, int(val2)

    return None, None, None


def _pagination_values(page1_num, page2_num, total_pages):
    """Return numeric values for pages 2..N, preserving an observed offset."""
    if total_pages < 2:
        return []
    if page2_num is None:
        return list(range(2, total_pages + 1))
    if page1_num is not None:
        step = page2_num - page1_num
    else:
        step = page2_num if page2_num > 2 else 1
    if step <= 0:
        step = 1
    return [page2_num + step * index for index in range(total_pages - 1)]


def derive_pagination_pattern_llm(url1, url2, api_key, model="gemma-3-27b-it"):
    """LLM fallback: ask the model to derive the pagination pattern.

    Only called when derive_pagination_pattern() returns None.
    Returns (pattern, page1_num, page2_num) or (None, None, None).
    """
    prompt = f"""<start_of_turn>user
Given two URLs from consecutive pages of the same website, derive the URL pattern.

Page 1 URL: {url1}
Page 2 URL: {url2}

Return a JSON object with:
- "pattern": the URL with the page number replaced by {{page}} (literal curly braces)
- "page1_num": the page number in URL 1 (integer, or null if page 1 has no number)
- "page2_num": the page number in URL 2 (integer)

Example:
Page 1: https://example.com/news/page/1/
Page 2: https://example.com/news/page/2/
Answer: {{"pattern": "https://example.com/news/page/{{page}}/", "page1_num": 1, "page2_num": 2}}

Example:
Page 1: https://example.com/articles
Page 2: https://example.com/articles?page=2
Answer: {{"pattern": "https://example.com/articles?page={{page}}", "page1_num": null, "page2_num": 2}}

Respond with ONLY the JSON object.
<end_of_turn>
<start_of_turn>model
"""

    try:
        genai.configure(api_key=api_key)
        llm = genai.GenerativeModel(model)
        gen_config = genai.GenerationConfig(
            max_output_tokens=256,
            temperature=0.0,
            response_mime_type="application/json",
        )
        response = llm.generate_content(prompt, generation_config=gen_config)
        _bump_llm_calls(agent="Pagination pattern")
        _bump_tokens("Pagination pattern", token_usage_from_response(response))
        raw = response.text
        if not raw:
            print(f"    ⚠️  LLM returned empty response.")
            return None, None, None

        text = raw.strip()
        print(f"    ℹ  LLM pattern response ({len(text)} chars): {text[:500]}")

        parsed = json.loads(text)
        pattern = parsed.get("pattern")
        p1 = parsed.get("page1_num")
        p2 = parsed.get("page2_num")

        if not pattern or "{page}" not in pattern:
            print(f"    ⚠️  LLM returned pattern without {{page}}: {pattern}")
            return None, None, None

        return pattern, p1, int(p2) if p2 is not None else None

    except (json.JSONDecodeError, ValueError) as e:
        print(f"    ⚠️  Could not parse LLM JSON: {e}")
        return None, None, None
    except Exception as e:
        tb = traceback.format_exc()
        print(f"    ⚠️  LLM pagination pattern failed: {e}\n{tb}")
        return None, None, None


# ═══════════════════════════════════════════════════════════════
# Incremental save (atomic writes)
# ═══════════════════════════════════════════════════════════════

def _atomic_json_write(path, data):
    """Write JSON to *path* via a temp file to avoid corruption on crash.

    On OneDrive/Windows the destination file can be transiently locked by the
    sync client or antivirus, making the atomic ``os.replace`` fail with
    PermissionError (WinError 5). We retry the rename a few times with a short
    backoff, and if it still won't budge we fall back to writing in place so the
    run's data is never lost (losing only the atomicity guarantee for that one
    write).
    """
    dir_name = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        last_err = None
        for attempt in range(5):
            try:
                os.replace(tmp, path)   # atomic on same filesystem
                return
            except PermissionError as e:  # transient OneDrive/AV lock
                last_err = e
                time.sleep(0.4 * (attempt + 1))
        # Rename kept failing — write directly in place as a last resort.
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  ⚠️  Atomic rename blocked (likely OneDrive/AV lock); "
                  f"wrote {os.path.basename(path)} in place instead.")
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        if not os.path.exists(path):
            raise last_err
    except Exception:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def save_incremental(run_dir, all_extracted, all_failures, progress_info,
                     all_dropped=None, link_coverage=None):
    """Persist current state so nothing is lost on crash."""
    _atomic_json_write(
        os.path.join(run_dir, "extracted_data_all.json"), all_extracted,
    )
    _atomic_json_write(
        os.path.join(run_dir, "failed_links.json"), all_failures,
    )
    if all_dropped is not None:
        _atomic_json_write(
            os.path.join(run_dir, "dropped_links.json"), all_dropped,
        )
    if link_coverage is not None:
        _atomic_json_write(
            os.path.join(run_dir, "link_coverage.json"), link_coverage,
        )
    _atomic_json_write(
        os.path.join(run_dir, "progress.json"), progress_info,
    )


def _save_listing_audit(run_dir, page_num, page_url, page_html,
                        raw_links, accepted_links, dropped_links):
    """Persist evidence for manual link-coverage audit.

    Each listing page gets a small JSON file with raw/accepted/dropped links,
    and, when available, the listing HTML used for extraction.
    """
    audit_dir = os.path.join(run_dir, "listing_audit")
    os.makedirs(audit_dir, exist_ok=True)
    stem = f"page_{int(page_num):04d}"
    html_rel = None
    if page_html:
        html_path = os.path.join(audit_dir, f"{stem}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(page_html)
        html_rel = os.path.relpath(html_path, run_dir).replace(os.sep, "/")

    links_rel = f"listing_audit/{stem}_links.json"
    _atomic_json_write(os.path.join(run_dir, links_rel), {
        "page_num": page_num,
        "page_url": page_url,
        "raw_links": raw_links or [],
        "accepted_links": accepted_links or [],
        "dropped_links": dropped_links or [],
        "listing_html_file": html_rel,
    })
    return {"listing_html_file": html_rel, "link_audit_file": links_rel}


def _make_link_coverage_record(page_num, page_url, raw_links, accepted_links,
                               dropped_links, duplicate_count=0,
                               new_count=None, page_html=None,
                               fetch_error=None, evidence=None):
    """Small audit row for validating listing-page link recall.

    This cannot prove every article link exists, but it catches likely misses:
    empty pages, heavy filtering, duplicate-only pages, small HTML responses,
    and sudden link-count drops compared with neighbouring pages.
    """
    raw_count = len(raw_links or [])
    accepted_count = len(accepted_links or [])
    dropped_count = len(dropped_links or [])
    html_bytes = len(page_html or "") if page_html is not None else None
    if new_count is None:
        new_count = accepted_count
    record = {
        "page_num": page_num,
        "page_url": page_url,
        "raw_links": raw_count,
        "accepted_links": accepted_count,
        "new_unique_links": new_count,
        "duplicates": duplicate_count,
        "dropped_links": dropped_count,
        "html_bytes": html_bytes,
        "flags": [],
    }
    duplicate_ratio = duplicate_count / accepted_count if accepted_count else 0.0
    record["duplicate_ratio"] = round(duplicate_ratio, 4)
    if fetch_error:
        record["fetch_error"] = str(fetch_error)
        record["flags"].append("listing_fetch_failed")
    if evidence:
        record.update({k: v for k, v in evidence.items() if v})
    if raw_count == 0:
        record["flags"].append("zero_raw_links")
    if accepted_count == 0:
        record["flags"].append("zero_accepted_links")
    if raw_count and dropped_count / raw_count >= 0.5:
        record["flags"].append("many_links_filtered")
    if accepted_count and new_count == 0:
        record["flags"].append("all_links_duplicate")
    if duplicate_ratio >= 0.5:
        record["flags"].append("high_page_overlap")
    if duplicate_ratio >= 0.8:
        record["flags"].append("severe_page_overlap")
    if html_bytes is not None and html_bytes < 2000:
        record["flags"].append("small_listing_html")
    return record


def _refresh_link_coverage_flags(link_coverage):
    """Add cross-page anomaly flags such as sudden drops vs neighbouring pages."""
    counts = [r.get("accepted_links", 0) for r in link_coverage if not r.get("fetch_error")]
    nonzero = sorted(c for c in counts if c > 0)
    median = nonzero[len(nonzero) // 2] if nonzero else 0
    prev_count = None
    for rec in link_coverage:
        flags = set(rec.get("flags") or [])
        count = rec.get("accepted_links", 0) or 0
        if median and count > 0 and count < median * 0.4:
            flags.add("low_vs_run_median")
        if prev_count and count < prev_count * 0.4:
            flags.add("sudden_drop_from_previous_page")
        rec["flags"] = sorted(flags)
        prev_count = count
    return link_coverage


# ═══════════════════════════════════════════════════════════════
# Process a batch of articles against the cluster registry
# ═══════════════════════════════════════════════════════════════

def process_page_articles(articles_with_maps, cluster_registry,
                          api_key, model, requirements):
    """Match articles to known clusters; call the agent only for new ones.

    Returns (extracted, failures, updated_cluster_registry).
    """
    extracted = []
    failures = []

    batched = defaultdict(list)
    for art in articles_with_maps:
        batched[_sig_hash(art["structural_map"])].append(art)

    for sig, members in batched.items():
        if sig in cluster_registry:
            code = cluster_registry[sig]["code"]
            for m in members:
                ok, result = _extract_article_fields(code, m["html_content"], requirements)
                if ok and isinstance(result, dict):
                    validation = _validate_article_extraction(result, m, requirements)
                    if validation.get("critical"):
                        reason = "Critical field validation failed: " + "; ".join(
                            validation.get("issues") or ["unknown validation failure"]
                        )
                        failures.append({
                            "url": m["url"], "title": m["title"],
                            "reason": reason,
                        })
                        print(f"    ❌ {m['title'][:50]} (reused extractor rejected)")
                        print(f"       Error: {reason}")
                        continue
                    extracted.append({
                        "url": m["url"], "title": m["title"], "data": result,
                    })
                    print(f"    ✓ {m['title'][:50]} (reused cluster {sig})")
                else:
                    reason = str(result)
                    failures.append({
                        "url": m["url"], "title": m["title"],
                        "reason": reason,
                    })
                    print(f"    ❌ {m['title'][:50]} (cluster {sig} code failed)")
                    print(f"       Error: {reason}")
        else:
            rep = members[0]
            rest = members[1:]
            print(f"\n  🆕 New cluster {sig} ({len(members)} article(s))")
            print(f"     Representative: {rep['title'][:60]}")

            active_requirements = requirements
            success = False
            agent_result = {}
            rep_data = {}
            code_file = ""
            code_text = ""
            validation = {}
            attempts = 0

            while attempts < MAX_ARTICLE_EXTRACTOR_ATTEMPTS:
                attempts += 1
                success, agent_result = call_agent_cli(
                    rep["url"], api_key, active_requirements, model,
                    html_content=rep.get("html_content"),
                )

                if not success:
                    break

                code_file = agent_result.get("code_file", "")
                code_text = _load_generated_code(code_file)
                rep_data = _canonicalize_output_keys(agent_result.get("data", {}), requirements)

                if code_text:
                    ok, rerun_result = _extract_article_fields(code_text, rep["html_content"], requirements)
                    if ok and isinstance(rerun_result, dict):
                        rep_data = rerun_result
                    else:
                        rep_data = {}
                        validation = {
                            "ok": False,
                            "critical": True,
                            "issues": [f"generated code failed on representative HTML: {rerun_result}"],
                            "schema_warnings": [],
                            "missing_fields": _requested_field_names(requirements),
                            "present_fields": [],
                        }

                if not validation:
                    validation = _validate_article_extraction(rep_data, rep, requirements)

                if validation.get("ok"):
                    break

                decision = _automatic_validation_decision(validation, attempts)
                if decision:
                    print(
                        f"     Critical validation failure: automatically {decision}ing "
                        f"(attempt {attempts}/{MAX_ARTICLE_EXTRACTOR_ATTEMPTS})."
                    )
                else:
                    decision = _prompt_article_validation_decision(
                        sig, rep, validation, code_file, sample=False,
                    )
                _record_article_validation_event({
                    "cluster": sig,
                    "stage": "representative",
                    "attempt": attempts,
                    "url": rep.get("url"),
                    "title": rep.get("title"),
                    "code_file": code_file,
                    "decision": decision,
                    "issues": validation.get("issues") or [],
                    "schema_warnings": validation.get("schema_warnings") or [],
                })
                if decision == "retry" and attempts < MAX_ARTICLE_EXTRACTOR_ATTEMPTS:
                    active_requirements = _validation_feedback(
                        requirements, code_file, rep_data, validation, "the representative article",
                    )
                    validation = {}
                    continue
                if decision == "accept":
                    break
                success = False
                agent_result = {"error": "Article extractor rejected by validation"}
                break

            if not success:
                err = agent_result.get("error", "Unknown agent error")
                print(f"     ❌ Agent failed:\n{err}")
                for m in members:
                    failures.append({
                        "url": m["url"], "title": m["title"],
                        "reason": f"Agent failed on new cluster representative: {err}",
                    })
                continue

            print(f"     ✓ Agent succeeded! Code: {code_file}")

            extracted.append({
                "url": rep["url"], "title": rep["title"], "data": rep_data,
            })

            cluster_registry[sig] = {"code_file": code_file, "code": code_text}

            if rest and code_text:
                for sample in rest[:4]:
                    ok, sample_result = _extract_article_fields(code_text, sample["html_content"], requirements)
                    if not ok or not isinstance(sample_result, dict):
                        sample_warning = {
                            "ok": False,
                            "critical": True,
                            "issues": [f"generated code failed on sample article: {sample_result}"],
                            "schema_warnings": [],
                        }
                    else:
                        sample_warning = _validate_article_extraction(sample_result, sample, requirements)
                    if sample_warning and not sample_warning.get("ok") and sample_warning.get("critical"):
                        decision = _automatic_validation_decision(sample_warning, attempts)
                        if not decision:
                            decision = _prompt_article_validation_decision(
                                sig, sample, sample_warning, code_file, sample=True,
                            )
                        _record_article_validation_event({
                            "cluster": sig,
                            "stage": "sample",
                            "attempt": attempts,
                            "url": sample.get("url"),
                            "title": sample.get("title"),
                            "code_file": code_file,
                            "decision": decision,
                            "issues": sample_warning.get("issues") or [],
                            "schema_warnings": sample_warning.get("schema_warnings") or [],
                        })
                        if decision == "retry":
                            retry_requirements = _validation_feedback(
                                requirements,
                                code_file,
                                sample_result if isinstance(sample_result, dict) else {},
                                sample_warning,
                                "a same-cluster sample article",
                            )
                            success, retry_result = call_agent_cli(
                                rep["url"], api_key, retry_requirements, model,
                                html_content=rep.get("html_content"),
                            )
                            if success:
                                code_file = retry_result.get("code_file", "")
                                code_text = _load_generated_code(code_file)
                                ok, rerun_result = _extract_article_fields(code_text, rep["html_content"], requirements)
                                if ok and isinstance(rerun_result, dict):
                                    rep_data = rerun_result
                                    extracted[-1]["data"] = rep_data
                                    cluster_registry[sig] = {"code_file": code_file, "code": code_text}
                                    print(f"     ✓ Retry succeeded! Code: {code_file}")
                                    break
                            decision = "skip"
                        if decision == "skip":
                            failures.append({
                                "url": sample["url"], "title": sample["title"],
                                "reason": "Article extractor rejected by sample validation",
                            })
                            rest = [m for m in rest if m is not sample]
                        break

            if rest and code_text:
                for m in rest:
                    ok, result = _extract_article_fields(code_text, m["html_content"], requirements)
                    if ok and isinstance(result, dict):
                        validation = _validate_article_extraction(result, m, requirements)
                        if validation.get("critical"):
                            reason = "Critical field validation failed: " + "; ".join(
                                validation.get("issues") or ["unknown validation failure"]
                            )
                            failures.append({
                                "url": m["url"], "title": m["title"],
                                "reason": reason,
                            })
                            print(f"    ❌ {m['title'][:50]} (extractor rejected)")
                            print(f"       Error: {reason}")
                            continue
                        extracted.append({
                            "url": m["url"], "title": m["title"], "data": result,
                        })
                        print(f"    ✓ {m['title'][:50]}")
                    else:
                        reason = str(result)
                        failures.append({
                            "url": m["url"], "title": m["title"],
                            "reason": reason,
                        })
                        print(f"    ❌ {m['title'][:50]}")
                        print(f"       Error: {reason}")
            elif rest:
                for m in rest:
                    failures.append({
                        "url": m["url"], "title": m["title"],
                        "reason": "Code file from representative not found",
                    })

    return extracted, failures, cluster_registry


# ═══════════════════════════════════════════════════════════════
# Helper: fetch articles, build structural maps
# ═══════════════════════════════════════════════════════════════

# How many articles to fetch at the same time. Kept low (2) to stay polite and
# avoid tripping rate-limiting / Cloudflare on the target site.
ARTICLE_FETCH_CONCURRENCY = 2

# Validate a representative sample before fetching and clustering an entire
# listing. Structure diversity alone never trips the breaker: it must coincide
# with weak article-page evidence across the sample.
LINK_PREFLIGHT_SAMPLE_SIZE = 10
LINK_PREFLIGHT_SINGLETON_RATIO = 0.80
LINK_PREFLIGHT_MIN_ARTICLE_RATIO = 0.50

# Minimum HTML size for a plain-requests result to be trusted. Smaller responses
# are usually JS shells or block pages, so we fall back to the browser.
_MIN_REQUESTS_HTML_BYTES = 2000

_REQUESTS_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'ar,en-US;q=0.9,en;q=0.8',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}


def _quick_requests_fetch(url):
    """Single fast plain-HTTP attempt. Returns HTML or None.

    Returns None (so the caller falls back to the browser) when the request
    fails, returns a Cloudflare/bot challenge, or doesn't look like real HTML.
    """
    try:
        resp = requests.get(url, headers=_REQUESTS_HEADERS, timeout=20)
        resp.raise_for_status()
        html = resp.text
    except Exception:
        return None

    if not html or len(html) < _MIN_REQUESTS_HTML_BYTES:
        return None
    if _cf_is_challenge_page(html):
        return None
    lowered = html.lower()
    if not any(tag in lowered for tag in ('<html', '<body', '<article', '<div')):
        return None
    return html


async def _fetch_one_article(link):
    """Fetch a single article: try plain HTTP first, fall back to the browser.

    Returns (item_dict_or_None, failure_dict_or_None).
    """
    url = link["url"]
    title = link.get("title", "")

    # ── Fast path: plain requests (runs in a thread; it's blocking) ──
    html_content = await asyncio.to_thread(_quick_requests_fetch, url)
    structural_map = None
    via = "requests"

    if html_content:
        soup = BeautifulSoup(html_content, "lxml")
        structural_map = _sp_create_structural_map(soup.body if soup.body else soup)

    # ── Fallback: full Cloudflare-resistant browser fetch ──
    if not html_content or not structural_map:
        via = "browser"
        html_content, structural_map = await fetch_page_structure(url)

    if html_content and structural_map:
        return {
            "url": url,
            "title": title,
            "html_content": html_content,
            "structural_map": structural_map,
            "_via": via,
        }, None

    return None, {"url": url, "title": title, "reason": "Empty response"}


async def fetch_articles(article_links, label="", concurrency=ARTICLE_FETCH_CONCURRENCY):
    """Fetch HTML + structural maps for a list of {url, title} dicts.

    Uses a plain-requests-first strategy with a browser fallback, run with
    bounded concurrency (default 2) to speed up large batches while staying
    polite to the target site.
    """
    total = len(article_links)
    sem = asyncio.Semaphore(concurrency)
    results = [None] * total
    done = {"n": 0}

    async def worker(idx, link):
        async with sem:
            try:
                item, failure = await _fetch_one_article(link)
            except Exception as e:
                tb = traceback.format_exc()
                item, failure = None, {
                    "url": link["url"], "title": link.get("title", ""),
                    "reason": f"{e}\n{tb}",
                }
            done["n"] += 1
            n = done["n"]
            ttl = link.get("title", "")[:50]
            if item:
                _bump_fetch_via(item["_via"])
                print(f"  [{n}/{total}]{label} ✓ {ttl} "
                      f"({len(item['html_content'])} bytes via {item['_via']})")
            else:
                print(f"  [{n}/{total}]{label} ❌ {ttl}: "
                      f"{failure['reason'].splitlines()[0][:80]}")
            results[idx] = (item, failure)

    await asyncio.gather(*(worker(i, l) for i, l in enumerate(article_links)))

    articles_with_maps = []
    fetch_failures = []
    for item, failure in results:
        if item:
            item.pop("_via", None)
            articles_with_maps.append(item)
        elif failure:
            fetch_failures.append(failure)

    return articles_with_maps, fetch_failures


def _looks_like_article_html(html):
    """Return whether HTML has generic metadata or content evidence of an article."""
    soup = BeautifulSoup(html or "", "lxml")
    json_ld = " ".join(
        script.get_text(" ", strip=True)
        for script in soup.find_all("script", attrs={"type": "application/ld+json"})
    )
    og_type = soup.find("meta", attrs={"property": "og:type"})
    explicit_article = (
        bool(_re.search(r'"@type"\s*:\s*"[^"\n]*article', json_ld, _re.I))
        or bool(og_type and "article" in (og_type.get("content") or "").lower())
        or bool(soup.find(attrs={"itemtype": _re.compile(r"article", _re.I)}))
    )
    if explicit_article:
        return True

    published = soup.find("meta", attrs={
        "property": _re.compile(r"article:(published|modified)_time", _re.I),
    }) or soup.find("meta", attrs={
        "name": _re.compile(r"(date|publish|publication)", _re.I),
    })
    root = soup.find("article") or soup.find("main")
    prose_length = sum(
        len(node.get_text(" ", strip=True)) for node in root.find_all("p")
    ) if root else 0
    return bool(published and soup.find("h1") and prose_length >= 600)


def _select_preflight_links(article_links, sample_size=LINK_PREFLIGHT_SAMPLE_SIZE):
    """Select an evenly distributed, order-preserving candidate sample."""
    total = len(article_links)
    if total <= sample_size:
        return list(article_links)
    indices = {
        round(position * (total - 1) / (sample_size - 1))
        for position in range(sample_size)
    }
    return [article_links[index] for index in sorted(indices)]


def _assess_link_preflight(articles):
    """Assess sampled page semantics and structural fragmentation."""
    signatures = [
        _sig_hash(article.get("structural_map", [])) for article in articles
    ]
    counts = {signature: signatures.count(signature) for signature in signatures}
    sampled = len(articles)
    singleton_pages = sum(counts[signature] == 1 for signature in signatures)
    article_like = sum(
        _looks_like_article_html(article.get("html_content", ""))
        for article in articles
    )
    should_stop = (
        sampled >= 8
        and singleton_pages / sampled >= LINK_PREFLIGHT_SINGLETON_RATIO
        and article_like / sampled < LINK_PREFLIGHT_MIN_ARTICLE_RATIO
    )
    return {
        "sampled_pages": sampled,
        "article_like_pages": article_like,
        "singleton_pages": singleton_pages,
        "status": "rejected" if should_stop else "passed",
    }


# ═══════════════════════════════════════════════════════════════
# Helper: extract links from a page using saved link-extraction code
# ═══════════════════════════════════════════════════════════════

def _pagination_url_regex(pattern):
    """Compile a regex matching the derived pagination URLs (…/page/{page}/…)."""
    if not pattern or "{page}" not in pattern:
        return None
    try:
        esc = _re.escape(pattern).replace(_re.escape("{page}"), r"\d+")
        return _re.compile("^" + esc + "$")
    except Exception:
        return None


def _filter_article_links(links, listing_url, pattern=None):
    """Drop pagination / category / navigation links the LLM may have grabbed
    by mistake, keeping only plausible article links.

    Safety net behind the Links Agent's own validation, and also applied to the
    reused link code on pages 2..N. Returns (kept_links, dropped_links), where
    each dropped entry is the original link dict augmented with a ``reason`` so
    a human can later audit whether the filter made a mistake.
    """
    pag_re = _pagination_url_regex(pattern)
    listing = (listing_url or "").rstrip("/")
    cat_root = None
    if "/category/" in (listing_url or ""):
        cat_root = listing_url.split("/category/")[0] + "/category/"

    def _drop(link, reason):
        url = (link.get("url") or "").strip() if isinstance(link, dict) else ""
        title = (link.get("title") or "").strip() if isinstance(link, dict) else ""
        dropped.append({"url": url, "title": title, "reason": reason})

    kept, dropped = [], []
    for l in links:
        if not isinstance(l, dict):
            _drop(l, "malformed link object")
            continue
        url = (l.get("url") or "").strip()
        title = (l.get("title") or "").strip()
        if not url or title == "" or title.isdigit():
            _drop(l, "empty or pagination-number title")
            continue
        if url.rstrip("/") == listing:
            _drop(l, "link back to the listing page")
            continue
        if pag_re and pag_re.match(url):
            _drop(l, "matches the derived pagination URL pattern")
            continue
        if _re.search(r"/page/\d+/?$", url):
            _drop(l, "generic /page/N/ pagination link")
            continue
        try:
            path = urlparse(url).path.lower()
        except Exception:
            path = ""
        if _re.search(r"/(taxonomy/term|writer|author|home/page)(/|$)", path):
            _drop(l, "author/profile/archive link")
            continue
        if cat_root and url.startswith(cat_root):
            _drop(l, "category/archive page")
            continue
        kept.append(l)
    return kept, dropped


def _looks_like_article_url(url, listing_url):
    """Return whether a URL can represent a same-site article document."""
    try:
        parsed = urlparse(url)
        listing = urlparse(listing_url or "")
    except Exception:
        return False
    if not parsed.scheme.startswith("http") or not parsed.netloc:
        return False
    if listing.netloc and parsed.netloc != listing.netloc:
        return False
    path = parsed.path.rstrip("/")
    if not path or path == (listing.path or "").rstrip("/"):
        return False
    if _re.search(
        r"\.(jpg|jpeg|png|gif|webp|svg|pdf|mp4|mp3|css|js|xml|json)$",
        path.lower(),
    ):
        return False
    return True


def _article_anchor_evidence(a_tag, title):
    """Score article intent using semantic and local card evidence."""
    ancestors = list(a_tag.parents)[:4]
    if any(
        getattr(node, "name", None) in {"nav", "header", "footer"}
        or (node.get("role") if hasattr(node, "get") else None)
        in {"menu", "menubar", "navigation"}
        for node in ancestors
    ):
        return -100

    score = 0
    visible_title = a_tag.get_text(" ", strip=True)
    accessible_title = (
        (a_tag.get("aria-label") or "").strip()
        or (a_tag.get("title") or "").strip()
    )
    if len(title) >= 8 and not title.isdigit():
        score += 1
    if accessible_title:
        score += 2
    if any(getattr(node, "name", None) == "article" for node in ancestors):
        score += 3

    local_nodes = [a_tag, *ancestors[:3]]
    semantic_text = " ".join(
        " ".join(
            [
                getattr(node, "name", "") or "",
                " ".join(node.get("class") or []),
                node.get("id") or "",
                node.get("role") or "",
                node.get("data-link-name") or "",
            ]
        ).lower()
        for node in local_nodes
        if hasattr(node, "get")
    )
    if _re.search(r"\b(article|story|post|card|teaser)\b", semantic_text):
        score += 2
    if _re.search(r"\b(nav|menu|pagination|pager|breadcrumb|section heading)\b", semantic_text):
        score -= 3
    if _re.search(r"\b(author|writer|byline|profile|print|edition|subscribe)\b", semantic_text):
        score -= 3

    local_root = ancestors[0] if ancestors else a_tag
    heading = (
        local_root
        if getattr(local_root, "name", None) in {"h1", "h2", "h3", "h4"}
        else local_root.find(["h1", "h2", "h3", "h4"])
    )
    if heading and heading.get_text(" ", strip=True) == title:
        score += 2
    if local_root.find("time"):
        score += 1
    if local_root.find("img") or local_root.find("picture"):
        score += 1
    if not visible_title and not accessible_title:
        score -= 2
    return score


def _fallback_article_links_from_html(html, listing_url):
    """Extract likely article links from article/card containers.

    This does not replace the generated Links Agent code. It is a recovery guard
    for cases where the generated code picks a navigation/category anchor from
    each card while a later anchor in the same card is the real article URL.
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    base = "/".join((listing_url or "").split("/")[:3])
    class_fragments = ("article", "story", "post", "card", "item", "teaser")
    containers = soup.find_all(
        lambda tag: tag.name == "article" or any(
            fragment in " ".join(tag.get("class") or []).lower()
            for fragment in class_fragments
        )
    )
    seen = set()
    links = []

    def add_anchor(a_tag, container=None, require_text=False):
        href = a_tag.get("href", "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            return False
        parsed_url = urlparse(urljoin(base, href))
        url = parsed_url._replace(fragment="").geturl()
        if url in seen or not _looks_like_article_url(url, listing_url):
            return False
        title = (
            a_tag.get_text(" ", strip=True)
            or (a_tag.get("aria-label") or "").strip()
            or (a_tag.get("title") or "").strip()
        )
        if not title and container is not None:
            title_el = container.select_one(
                "[class*='title'], h1, h2, h3, h4, [class*='headline']"
            )
            title = title_el.get_text(" ", strip=True) if title_el else ""
        if require_text and not title:
            return False
        if _article_anchor_evidence(a_tag, title) < 3:
            return False
        seen.add(url)
        links.append({"url": url, "title": title or url})
        return True

    for container in containers:
        for a_tag in container.find_all("a", href=True):
            if add_anchor(a_tag, container=container):
                break
    if len(links) < 8:
        for a_tag in soup.find_all("a", href=True):
            add_anchor(a_tag, require_text=True)
    return links


MIN_LINK_RECALL_RATIO = 0.70
MIN_LINK_RECALL_BASELINE = 3


def recover_under_extracted_links(raw_links, accepted_links, html, listing_url, pattern=None):
    """Augment suspiciously small generated-link output from saved HTML.

    The deterministic DOM pass acts as a conservative recall baseline. When it
    finds at least three plausible article links and generated recall is below
    70%, merge the missed links and record the measured ratio for auditing.
    """
    fallback_raw = _fallback_article_links_from_html(html, listing_url)
    fallback_accepted, fallback_dropped = _filter_article_links(
        fallback_raw, listing_url, pattern,
    )
    before = len(accepted_links or [])
    fallback_count = len(fallback_accepted)
    recall_ratio = before / fallback_count if fallback_count else 1.0
    if (fallback_count < MIN_LINK_RECALL_BASELINE
            or recall_ratio >= MIN_LINK_RECALL_RATIO):
        return accepted_links, None

    merged = {l.get("url"): l for l in accepted_links or [] if l.get("url")}
    added = 0
    for link in fallback_accepted:
        url = link.get("url")
        if url and url not in merged:
            merged[url] = link
            added += 1
    recovered = list(merged.values())
    return recovered, {
        "generated_raw_links": len(raw_links or []),
        "generated_accepted_links": before,
        "fallback_raw_links": len(fallback_raw),
        "fallback_accepted_links": fallback_count,
        "fallback_dropped_links": len(fallback_dropped),
        "minimum_recall_ratio": MIN_LINK_RECALL_RATIO,
        "generated_recall_vs_dom": round(recall_ratio, 4),
        "validation": "low_dom_link_recall",
        "recovered_links_added": added,
        "accepted_links_after_recovery": len(recovered),
    }


def run_link_extraction_code(code, html):
    """Execute the link-extraction code on *html* and return article_links list."""
    ok, result = execute_extraction_code(code, html)
    if ok and isinstance(result, dict):
        return result.get("article_links", [])
    return []


def _clean_url_input(raw):
    """Sanitize a URL a user pasted into a prompt.

    Strips surrounding whitespace, a leading markdown bullet ("- ", "* ", "• "),
    and wrapping quotes or angle brackets, so a copy-pasted list item like
    "- https://site.com/x" becomes "https://site.com/x".
    """
    if not raw:
        return ""
    url = raw.strip()
    # Drop a leading markdown/list bullet (possibly repeated).
    url = _re.sub(r"^\s*(?:[-*•]\s+)+", "", url)
    # Drop wrapping quotes or angle brackets.
    url = url.strip().strip('\'"<>').strip()
    return url


# ═══════════════════════════════════════════════════════════════
# Dynamic pagination: infinite scroll & "load more" button
# ═══════════════════════════════════════════════════════════════

# Common "load more" button selectors/text used as fallbacks when the user
# does not supply an explicit selector.
_LOAD_MORE_HINTS = [
    "load more", "show more", "view more", "see more", "more articles",
    "load more articles", "المزيد", "تحميل المزيد", "عرض المزيد", "شاهد المزيد",
]


async def _click_load_more(page, selector):
    """Try to click a 'load more' button. Returns True if a click happened."""
    # 1) Explicit user-provided CSS selector
    if selector:
        try:
            el = await page.query_selector(selector)
            if el and await el.is_visible():
                await el.scroll_into_view_if_needed()
                await el.click()
                return True
        except Exception:
            pass

    # 2) Common buttons/links matched by visible text
    try:
        candidates = await page.query_selector_all(
            "button, a, span[role='button'], div[role='button']"
        )
        for el in candidates:
            try:
                if not await el.is_visible():
                    continue
                text = (await el.inner_text() or "").strip().lower()
                if text and any(hint in text for hint in _LOAD_MORE_HINTS):
                    await el.scroll_into_view_if_needed()
                    await el.click()
                    return True
            except Exception:
                continue
    except Exception:
        pass

    return False


async def collect_listing_html(url, mode="scroll", max_rounds=20,
                               load_more_selector=None, delay_ms=2000):
    """Load a JS listing page and accumulate all article content into one HTML.

    mode="scroll"     → repeatedly scroll to the bottom (infinite scroll).
    mode="load_more"  → repeatedly click a 'load more' button.

    Stops when the page height stops growing (scroll) or no 'load more'
    button remains (load_more), or after *max_rounds* iterations.
    Returns the final fully-loaded HTML (or None on failure).
    """
    label = "infinite scroll" if mode == "scroll" else "load-more button"
    print(f"\n⏳ Collecting links via {label} (up to {max_rounds} rounds)...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox'],
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
        )
        page = await context.new_page()
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            try:
                await page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass

            prev_height = 0
            stagnant = 0
            for i in range(max_rounds):
                if mode == "load_more":
                    clicked = await _click_load_more(page, load_more_selector)
                    if not clicked:
                        print(f"  ✓ No more 'load more' button (round {i + 1}) — stopping")
                        break
                else:
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

                await page.wait_for_timeout(delay_ms)
                height = await page.evaluate("document.body.scrollHeight")
                print(f"  round {i + 1}/{max_rounds}: page height = {height}")

                if height <= prev_height:
                    stagnant += 1
                    if stagnant >= 2:
                        print(f"  ✓ Content stopped growing — stopping after {i + 1} rounds")
                        break
                else:
                    stagnant = 0
                prev_height = height

            html_content = await page.content()
            if _cf_is_challenge_page(html_content):
                print("  ❌ Dynamic listing returned a CAPTCHA/access-denial page")
                return None
            print(f"  ✓ Collected HTML ({len(html_content)} bytes)")
            return html_content
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  ❌ Dynamic collection failed: {e}\n{tb}")
            return None
        finally:
            try:
                await browser.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
# Stats reporting → results.md (appended once per run for the paper)
# ═══════════════════════════════════════════════════════════════

def _field_na_stats(all_extracted):
    """Count missing/N/A values per field across all extracted articles.

    Returns (total_articles, [(field, na_count, total, percent), ...]).
    A field absent from an article's data counts as missing for that field.
    """
    total = len(all_extracted)
    if total == 0:
        return 0, []

    fields = []            # union of field names, in first-seen order
    seen = set()
    for item in all_extracted:
        data = item.get("data") or {}
        if isinstance(data, dict):
            for k in data.keys():
                if k not in seen:
                    seen.add(k)
                    fields.append(k)

    stats = []
    for f in fields:
        na = 0
        for item in all_extracted:
            data = item.get("data") or {}
            if (not isinstance(data, dict) or f not in data
                    or is_na_value(data.get(f))):
                na += 1
        stats.append((f, na, total, 100.0 * na / total))
    return total, stats


def write_results_md(stats, path="results.md"):
    """Append a per-run results section to results.md for later analysis."""
    header_needed = not os.path.exists(path)

    input_urls = stats.get("input_urls", [])
    errors = stats.get("errors", [])
    elapsed = stats.get("elapsed_seconds", 0.0)
    mins, secs = divmod(int(elapsed), 60)

    lines = []
    if header_needed:
        lines.append("# Scraping Run Results\n")
        lines.append("Auto-generated stats, one section per orchestrator run.\n")

    lines.append(f"\n## Run {stats.get('timestamp', '')}\n")
    lines.append(f"- **Run directory:** `{stats.get('run_dir', 'N/A')}`")
    lines.append(f"- **Model:** {stats.get('model', 'N/A')}")
    lines.append(f"- **Pagination type:** {stats.get('pagination_type', 'N/A')}")
    scroll_mode = stats.get("scroll_mode")
    scroll_rounds = stats.get("scroll_rounds_requested")
    load_more_selector = stats.get("load_more_selector")
    if scroll_mode == "scroll" and scroll_rounds is not None:
        lines.append(f"- **Scroll rounds requested:** {scroll_rounds}")
    elif scroll_mode == "load_more" and scroll_rounds is not None:
        lines.append(f"- **Load-more clicks requested:** {scroll_rounds}")
        lines.append(
            f"- **Load-more selector:** {load_more_selector or 'auto-detect'}"
        )

    if input_urls:
        lines.append("- **Input URLs:**")
        for u in input_urls:
            lines.append(f"  - {u}")

    lines.append(f"- **Requirements:** {stats.get('requirements', 'N/A')}")
    lines.append(f"- **Pages requested:** {stats.get('pages_requested', 'N/A')}")
    lines.append(f"- **Pages processed:** {stats.get('pages_processed', 'N/A')}")
    lines.append(f"- **Articles extracted:** {stats.get('articles_extracted', 0)}")
    lines.append(f"- **Articles failed:** {stats.get('articles_failed', 0)}")
    lines.append(f"- **Clusters (unique structures):** {stats.get('clusters', 0)}")

    # ── Links dropped by the non-article filter (audit trail) ──
    links_dropped = stats.get("links_dropped") or []
    lines.append(f"- **Links dropped (non-article filter):** {len(links_dropped)}")
    if links_dropped:
        by_reason = defaultdict(int)
        for d in links_dropped:
            by_reason[str((d or {}).get("reason", "unknown"))] += 1
        for reason, cnt in sorted(by_reason.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  - {reason}: {cnt}")
        lines.append(
            f"  - Full list saved to "
            f"`{stats.get('run_dir', 'N/A')}/dropped_links.json`"
        )

    # ── Listing-page link coverage audit ──
    link_coverage = stats.get("link_coverage") or []
    if link_coverage:
        flagged = [r for r in link_coverage if r.get("flags")]
        total_raw = sum(r.get("raw_links", 0) or 0 for r in link_coverage)
        total_accepted = sum(r.get("accepted_links", 0) or 0 for r in link_coverage)
        total_new = sum(r.get("new_unique_links", 0) or 0 for r in link_coverage)
        lines.append(
            f"- **Link coverage audit:** {len(link_coverage)} page(s), "
            f"{total_raw} raw links, {total_accepted} accepted article links, "
            f"{total_new} new unique links"
        )
        if flagged:
            lines.append(f"  - ⚠ {len(flagged)} page(s) flagged for review:")
            for rec in flagged[:10]:
                flags = ", ".join(rec.get("flags") or [])
                lines.append(
                    f"    - page {rec.get('page_num')}: {flags} "
                    f"({rec.get('accepted_links', 0)} accepted, "
                    f"{rec.get('new_unique_links', 0)} new)"
                )
            if len(flagged) > 10:
                lines.append(f"    - ...and {len(flagged) - 10} more")
        lines.append(
            f"  - Full audit saved to "
            f"`{stats.get('run_dir', 'N/A')}/link_coverage.json`"
        )

    link_recovery = stats.get("link_recovery") or []
    if link_recovery:
        lines.append(f"- **Link recovery guard:** triggered on {len(link_recovery)} page(s)")
        for rec in link_recovery[:10]:
            lines.append(
                f"  - page {rec.get('page_num', '?')}: "
                f"{rec.get('generated_accepted_links', 0)} generated accepted -> "
                f"{rec.get('accepted_links_after_recovery', 0)} after recovery "
                f"(+{rec.get('recovered_links_added', 0)})"
            )
        if len(link_recovery) > 10:
            lines.append(f"  - ...and {len(link_recovery) - 10} more")

    link_preflight = stats.get("link_preflight") or {}
    if link_preflight:
        lines.append(
            f"- **Link preflight:** {link_preflight.get('status', 'unknown')} - "
            f"{link_preflight.get('article_like_pages', 0)}/"
            f"{link_preflight.get('sampled_pages', 0)} article-like, "
            f"{link_preflight.get('singleton_pages', 0)}/"
            f"{link_preflight.get('sampled_pages', 0)} singleton structures"
        )

    article_validation = stats.get("article_validation") or []
    if article_validation:
        lines.append(
            f"- **Article extractor validation:** {len(article_validation)} warning(s)"
        )
        for event in article_validation[:10]:
            issues = "; ".join((event.get("issues") or [])[:3])
            schema = "; ".join((event.get("schema_warnings") or [])[:2])
            detail = issues or schema or "validation warning"
            lines.append(
                f"  - cluster {event.get('cluster', '?')} {event.get('stage', '?')} "
                f"attempt {event.get('attempt', '?')}: {event.get('decision', 'recorded')} - "
                f"{detail[:180]}"
            )
        if len(article_validation) > 10:
            lines.append(f"  - ...and {len(article_validation) - 10} more")

    lines.append(f"- **LLM calls:** {stats.get('llm_calls', 0)}")

    # ── Which agent made each LLM call ──
    llm_by_agent = stats.get("llm_calls_by_agent") or {}
    if llm_by_agent:
        for agent, cnt in sorted(llm_by_agent.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  - {agent}: {cnt}")

    # ── Cost-efficiency: LLM calls per article & code-reuse rate ──
    n_extracted = stats.get("articles_extracted", 0) or 0
    if n_extracted:
        per_article = stats.get("llm_calls", 0) / n_extracted
        lines.append(f"- **LLM calls per article:** {per_article:.2f}")
        # Each new cluster costs one Article-agent generation; the rest reuse it.
        article_gen = llm_by_agent.get("Article agent", 0)
        reused = max(n_extracted - article_gen, 0)
        reuse_pct = 100.0 * reused / n_extracted
        lines.append(
            f"- **Code reuse rate:** {reused}/{n_extracted} articles "
            f"reused cluster code ({reuse_pct:.0f}%)"
        )

    # ── Token usage & estimated cost ──
    tokens_by_agent = stats.get("tokens_by_agent") or {}
    tot_prompt = 0
    tot_out = 0
    if tokens_by_agent:
        tot_prompt = sum(v.get("prompt", 0) for v in tokens_by_agent.values())
        tot_out = sum(v.get("candidates", 0) for v in tokens_by_agent.values())
        tot_all = sum(v.get("total", 0) for v in tokens_by_agent.values())
        lines.append(
            f"- **Tokens:** {tot_all:,} total "
            f"({tot_prompt:,} prompt + {tot_out:,} output)"
        )
        for agent, v in sorted(tokens_by_agent.items(),
                               key=lambda kv: -kv[1].get("total", 0)):
            lines.append(f"  - {agent}: {v.get('total', 0):,} tokens")

    llm_calls = stats.get("llm_calls", 0) or 0
    cost = None
    if _uses_gemma_free_tier(stats.get("model"), llm_calls):
        cost = 0.0
        lines.append(
            f"- **Estimated cost:** $0.0000 "
            f"(Gemma 4 free tier; {llm_calls:,}/"
            f"{GEMMA_FREE_TIER_CALL_LIMIT:,} calls recorded in this run)"
        )
    elif tokens_by_agent:
        cost = (tot_prompt / 1_000_000 * LLM_PRICE_PER_1M_INPUT
                + tot_out / 1_000_000 * LLM_PRICE_PER_1M_OUTPUT)
        lines.append(
            f"- **Estimated cost:** ${cost:.4f} "
            f"(at ${LLM_PRICE_PER_1M_INPUT}/${LLM_PRICE_PER_1M_OUTPUT} "
            f"per 1M input/output tokens)"
        )
    if cost is not None and n_extracted:
        per_1k = cost / n_extracted * 1000
        lines.append(f"- **Estimated cost per 1,000 articles:** ${per_1k:.4f}")

    # ── Structural-map depth budgeting (prompt size & any depth shrink) ──
    if _MAP_FITS:
        reduced = [f for f in _MAP_FITS if isinstance(f.get("depth"), int)
                   and f["depth"] < _FULL_MAP_DEPTH]
        lines.append(
            f"- **Structural-map depth:** {len(_MAP_FITS)} agent call(s), "
            f"{len(reduced)} shrunk below full depth {_FULL_MAP_DEPTH} to fit the "
            f"input-token budget"
        )
        for f in _MAP_FITS:
            tok = f.get("input_tokens")
            tok_str = f"~{tok:,} input tokens" if isinstance(tok, int) else "input tokens n/a"
            note = "" if (isinstance(f.get("depth"), int)
                          and f["depth"] >= _FULL_MAP_DEPTH) else "  ⚠ reduced"
            lines.append(f"  - {f['agent']}: depth {f.get('depth')} ({tok_str}){note}")

    lines.append(f"- **Total time:** {mins}m {secs}s")

    # ── Fetch method breakdown (fast requests vs. browser fallback) ──
    fetch_via = stats.get("fetch_via") or {}
    req_n = fetch_via.get("requests", 0)
    br_n = fetch_via.get("browser", 0)
    if req_n or br_n:
        total_fetch = req_n + br_n
        req_pct = 100.0 * req_n / total_fetch
        br_pct = 100.0 * br_n / total_fetch
        lines.append(
            f"- **Fetch method:** {req_n} via requests ({req_pct:.0f}%), "
            f"{br_n} via browser ({br_pct:.0f}%)"
        )

    # ── Missing / N/A audit of the extracted data ──
    total_arts, na_stats = _field_na_stats(stats.get("extracted_data") or [])
    if na_stats:
        lines.append(f"- **Missing/N/A values (of {total_arts} extracted):**")
        for field, na, tot, pct in na_stats:
            lines.append(f"  - `{field}`: {na}/{tot} N/A ({pct:.0f}%)")

    if errors:
        lines.append(f"- **Errors ({len(errors)}):**")
        for err in errors[:10]:
            one_line = str(err).replace("\n", " ")[:200]
            lines.append(f"  - {one_line}")
        if len(errors) > 10:
            lines.append(f"  - ...and {len(errors) - 10} more")
    else:
        lines.append("- **Errors:** none")

    lines.append("")

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n📝 Results appended to {path}")


# ═══════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("🎯 ORCHESTRATOR")
    print("=" * 60)
    # defense, just in case we started calling main twice in the same process
    #ToDo: remove them cause when will that ever happen? the program is meant to be an application
    _reset_llm_calls()
    _reset_fetch_via()
    _reset_tokens()
    _reset_map_fits()
    _reset_article_validation_events()
    run_start = time.time()
    input_urls = []          # every URL the user supplied (for results.md)
    pagination_type = "single page"

    # ── Step 0: Setup ────────────────────────────────────────
    api_key = input("\n🔑 Enter your Gemini API key: ").strip()
    if not api_key:
        print("❌ API key is required!")
        return

    print("\n⏳ Checking available models...")
    available_models = await list_available_models(api_key)

    gemma_models = [m for m in available_models if "gemma" in m.lower()]
    other_models = [m for m in available_models if "gemma" not in m.lower()]
    sorted_models = gemma_models + other_models

    if sorted_models:
        print(f"\n Found {len(sorted_models)} available models:")
        if gemma_models:
            print("  ── Gemma models ──")
        for i, m in enumerate(sorted_models, 1):
            marker = " ★" if "gemma" in m.lower() else ""
            print(f"   {i}. {m}{marker}")

        print("\n Select a model:")
        print("   [Enter number] Choose from list above")
        print("   [Press Enter]  Use default (gemma-3-27b-it)")
        choice = input("   → ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(sorted_models):
            model = sorted_models[int(choice) - 1]
            if model.startswith("models/"):
                model = model[7:]
            print(f"✓ Selected: {model}")
        else:
            model = "gemma-3-27b-it"
            print(f"✓ Using default: {model}")
    else:
        model = input("\n🤖 Model name [Enter for gemma-3-27b-it]: ").strip() or "gemma-3-27b-it"

    run_dir = f"orch_runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n📂 Run directory: {run_dir}")

    # ── Listing-page flow ────────────────────────────────────
    listing_url = _clean_url_input(input("\n🌐 Enter the listing page URL: "))
    if not listing_url:
        print("❌ URL is required!")
        return
    input_urls.append(listing_url)

    requirements = input(
        "\n📝 What data to extract from each article?\n"
        "   (default: 'title, date, author, article body text')\n   → "
    ).strip() or "title, date, author, article body text"

    # ══════════════════════════════════════════════════════════
    # Phase 1: Pagination setup (user-driven, no structural map)
    # ══════════════════════════════════════════════════════════
    page_urls = []  # URLs for pages 2, 3, 4, ...  (numbered pagination only)
    pattern = None            # derived pagination URL pattern (…/page/{page}/…)
    scroll_mode = None        # None | "scroll" | "load_more"
    scroll_selector = None    # optional CSS selector for the 'load more' button
    scroll_rounds = 20        # max scroll/click iterations

    print("\n📄 How is this listing paginated?")
    print("   [1] Numbered pagination — I'll provide page 1 and page 2 URLs")
    print("   [2] No pagination — single page only")
    print("   [3] Infinite scroll — content loads as you scroll down")
    print("   [4] Load more button — content loads when a button is clicked")
    pag_choice = input("   → ").strip()

    if pag_choice == "3":
        pagination_type = "infinite scroll"
        scroll_mode = "scroll"
        rounds_str = input(
            "\n   How many scroll rounds at most? [Enter for 20] → "
        ).strip()
        if rounds_str.isdigit() and int(rounds_str) > 0:
            scroll_rounds = int(rounds_str)
        print(f"   ✓ Infinite scroll — up to {scroll_rounds} rounds.")
    elif pag_choice == "4":
        pagination_type = "load more button"
        scroll_mode = "load_more"
        scroll_selector = input(
            "\n   CSS selector for the 'load more' button "
            "[Enter to auto-detect by text] → "
        ).strip() or None
        rounds_str = input(
            "   How many clicks at most? [Enter for 20] → "
        ).strip()
        if rounds_str.isdigit() and int(rounds_str) > 0:
            scroll_rounds = int(rounds_str)
        print(f"   ✓ Load-more — up to {scroll_rounds} clicks"
              + (f" on '{scroll_selector}'." if scroll_selector else " (auto-detect)."))
    elif pag_choice == "1":
        pagination_type = "numbered pagination"
        print(f"\n   Page 1 URL [Enter to use the listing URL above]:")
        print(f"   ({listing_url})")
        url1 = _clean_url_input(input("   → ")) or listing_url

        url2 = _clean_url_input(input("\n   Page 2 URL: "))
        if not url2:
            print("   ❌ Page 2 URL is required for pagination!")
        else:
            input_urls.append(url2)
            # Try string diff first
            pattern, p1_num, p2_num = derive_pagination_pattern(url1, url2)

            if pattern:
                print(f"\n   ✓ Derived URL pattern (string diff): {pattern}")
            else:
                # Fall back to LLM
                print(f"\n   ⚠️  String diff couldn't derive pattern. Asking LLM...")
                pattern, p1_num, p2_num = derive_pagination_pattern_llm(
                    url1, url2, api_key, model,
                )
                if pattern:
                    print(f"   ✓ Derived URL pattern (LLM): {pattern}")

            if pattern:
                if p1_num is not None:
                    print(f"     (page 1 = {p1_num}, page 2 = {p2_num})")
                else:
                    print(f"     (page 2 = {p2_num})")

                print(f"\n   Is this pattern correct?")
                print(f"   [Enter] Yes, use it")
                print(f"   [type]  Enter corrected pattern (use {{page}} as placeholder)")
                pat_input = input("   → ").strip()
                if pat_input:
                    pattern = pat_input
                    print(f"   ✓ Using your pattern: {pattern}")

                total_str = input("\n   📄 How many pages to scrape (including page 1)? → ").strip()
                if total_str.isdigit() and int(total_str) >= 2:
                    total_pages = int(total_str)
                    page_urls = [
                        pattern.format(page=n)
                        for n in _pagination_values(p1_num, p2_num, total_pages)
                    ]
                    print(f"   ✓ Will scrape {total_pages} pages ({len(page_urls)} after page 1)")
                    print(f"     First: {page_urls[0]}")
                    if len(page_urls) > 1:
                        print(f"     Last:  {page_urls[-1]}")
                else:
                    print("   ⚠️  Need at least 2 pages. Proceeding with page 1 only.")
            else:
                # Both string diff and LLM failed — ask for manual pattern
                print(f"\n   ⚠️  Could not derive pattern automatically.")
                print(f"   Enter the URL pattern manually (use {{page}} as page number placeholder):")
                print(f"   Example: https://site.com/news/page/{{page}}/")
                manual_pat = input("   → ").strip()
                if manual_pat and "{page}" in manual_pat:
                    pattern = manual_pat
                    total_str = input("   📄 How many pages to scrape (including page 1)? → ").strip()
                    if total_str.isdigit() and int(total_str) >= 2:
                        total_pages = int(total_str)
                        page_urls = [
                            pattern.format(page=n)
                            for n in range(2, total_pages + 1)
                        ]
                        print(f"   ✓ Will scrape {total_pages} pages ({len(page_urls)} after page 1)")
                    else:
                        print("   ⚠️  Need at least 2 pages. Proceeding with page 1 only.")
                else:
                    print("   ⚠️  Invalid or missing pattern. Proceeding with page 1 only.")
    else:
        print("   ✓ Single page — no pagination.")

    # ══════════════════════════════════════════════════════════
    # Phase 1.5: For dynamic listings, load the fully-scrolled DOM first
    # ══════════════════════════════════════════════════════════
    # Infinite-scroll / load-more pages render only a skeleton (often a single
    # featured card) on the initial load; the article grid appears after
    # scrolling. Collect the fully-loaded HTML up front and hand it to the Links
    # agent so it generates its selector from the complete grid instead of a
    # one-card page (which produced 0 links before this change).
    scrolled_html_file = None
    full_html = None
    if scroll_mode:
        full_html = await collect_listing_html(
            listing_url, mode=scroll_mode,
            max_rounds=scroll_rounds, load_more_selector=scroll_selector,
        )
        if full_html:
            scrolled_html_file = os.path.join(run_dir, "listing_full.html")
            with open(scrolled_html_file, "w", encoding="utf-8") as f:
                f.write(full_html)
            print(f"  ✓ Saved fully-loaded listing HTML ({len(full_html)} bytes) "
                  f"→ {scrolled_html_file}")
        else:
            print("  ⚠️  Dynamic collection failed — the Links agent will fetch "
                  "the page live instead.")

    # ══════════════════════════════════════════════════════════
    # Phase 2: Page 1 — extract links (LLM call #1)
    # ══════════════════════════════════════════════════════════
    print(f"\n⏳ Extracting article links from page 1: {listing_url}...")
    link_success, link_result = call_links_agent_cli(
        listing_url, api_key, model, html_file=scrolled_html_file)

    all_dropped = []  # links the article-link filter removed, kept for auditing
    link_coverage = []  # per-listing-page link recall sanity checks
    link_recovery = []  # fallback repairs when generated selectors undercount

    if not link_success:
        error_msg = link_result.get("error", "Unknown error")
        page1_links_raw = []
        page1_links, recovery = recover_under_extracted_links(
            page1_links_raw, [], full_html, listing_url, pattern,
        )
        if recovery:
            recovery["page_num"] = 1
            recovery["agent_error"] = error_msg
            link_recovery.append(recovery)
            print(
                f"  ⚠ Links Agent failed, but recovery guard added "
                f"{recovery['recovered_links_added']} article link(s) "
                f"from the saved listing HTML"
            )
            link_result = {}
            dropped1 = []
        else:
            page1_audit_html = full_html or await asyncio.to_thread(_quick_requests_fetch, listing_url)
            page1_evidence = _save_listing_audit(
                run_dir, 1, listing_url, page1_audit_html,
                page1_links_raw, [], [],
            )
            link_coverage.append(_make_link_coverage_record(
                1, listing_url, page1_links_raw, [], [],
                page_html=page1_audit_html,
                evidence=page1_evidence,
            ))
            _refresh_link_coverage_flags(link_coverage)
            _atomic_json_write(os.path.join(run_dir, "link_coverage.json"), link_coverage)
            if link_recovery:
                _atomic_json_write(os.path.join(run_dir, "link_recovery.json"), link_recovery)
            print(f"\n❌ Links extraction failed:\n{error_msg}")
            write_results_md({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_dir": run_dir, "model": model,
                "pagination_type": pagination_type, "input_urls": input_urls,
                "scroll_mode": scroll_mode,
                "scroll_rounds_requested": scroll_rounds if scroll_mode else None,
                "load_more_selector": scroll_selector,
                "requirements": requirements,
                "pages_requested": len(page_urls) + 1, "pages_processed": 0,
                "articles_extracted": 0, "articles_failed": 0, "clusters": 0,
                "link_coverage": link_coverage,
                "link_recovery": link_recovery,
                "llm_calls": _LLM_CALLS,
                "llm_calls_by_agent": dict(_LLM_CALLS_BY_AGENT),
                "tokens_by_agent": dict(_TOKENS_BY_AGENT),
                "elapsed_seconds": time.time() - run_start,
                "errors": [f"Links extraction failed: {error_msg}"],
            })
            return

    if link_success:
        page1_links_raw = link_result.get("data", {}).get("article_links", [])
        page1_links, dropped1 = _filter_article_links(page1_links_raw, listing_url, pattern)
        page1_links, recovery = recover_under_extracted_links(
            page1_links_raw, page1_links, full_html, listing_url, pattern,
        )
        if recovery:
            recovery["page_num"] = 1
            link_recovery.append(recovery)
            print(
                f"  ⚠ Link recovery guard added {recovery['recovered_links_added']} "
                f"article link(s) from the saved listing HTML"
            )
    page1_dropped = list(dropped1)
    for d in dropped1:
        d["page_num"] = 1
    all_dropped.extend(dropped1)
    if dropped1:
        print(f"  🧹 Dropped {len(dropped1)} non-article (pagination/category) link(s)")
    link_code_file = link_result.get("code_file", "")
    print(f"✓ Extracted {len(page1_links)} article links from page 1")

    if not page1_links:
        recovery_html = full_html
        if not recovery_html:
            try:
                recovery_html, _ = await fetch_page_structure(listing_url)
            except Exception:
                recovery_html = None
        page1_links, recovery = recover_under_extracted_links(
            page1_links_raw, page1_links, recovery_html, listing_url, pattern,
        )
        if recovery:
            recovery["page_num"] = 1
            link_recovery.append(recovery)
            full_html = recovery_html
            print(
                f"  ⚠ Link recall validation recovered "
                f"{recovery['recovered_links_added']} article link(s) from the DOM"
            )

    if not page1_links:
        print("❌ No article links found on page 1!")
        page1_audit_html = full_html or await asyncio.to_thread(_quick_requests_fetch, listing_url)
        page1_evidence = _save_listing_audit(
            run_dir, 1, listing_url, page1_audit_html,
            page1_links_raw, page1_links, page1_dropped,
        )
        link_coverage.append(_make_link_coverage_record(
            1, listing_url, page1_links_raw, page1_links, page1_dropped,
            page_html=page1_audit_html,
            evidence=page1_evidence,
        ))
        _refresh_link_coverage_flags(link_coverage)
        if all_dropped:
            _atomic_json_write(
                os.path.join(run_dir, "dropped_links.json"), all_dropped,
            )
        _atomic_json_write(os.path.join(run_dir, "link_coverage.json"), link_coverage)
        write_results_md({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": run_dir, "model": model,
            "pagination_type": pagination_type, "input_urls": input_urls,
            "scroll_mode": scroll_mode,
            "scroll_rounds_requested": scroll_rounds if scroll_mode else None,
            "load_more_selector": scroll_selector,
            "requirements": requirements,
            "pages_requested": len(page_urls) + 1, "pages_processed": 0,
            "articles_extracted": 0, "articles_failed": 0, "clusters": 0,
            "links_dropped": all_dropped,
            "link_coverage": link_coverage,
            "link_recovery": link_recovery,
            "llm_calls": _LLM_CALLS,
            "llm_calls_by_agent": dict(_LLM_CALLS_BY_AGENT),
            "tokens_by_agent": dict(_TOKENS_BY_AGENT),
            "elapsed_seconds": time.time() - run_start,
            "errors": ["No article links found on page 1"],
        })
        return

    # Load link-extraction code for reuse on later pages
    link_extraction_code = ""
    if link_code_file and os.path.exists(link_code_file):
        with open(link_code_file, "r", encoding="utf-8") as f:
            link_extraction_code = f.read()

    # ── Dynamic pagination fallback: only if the up-front scroll failed ──
    # When Phase 1.5 succeeded, page1_links already come from the fully-loaded
    # DOM, so no second scroll is needed. This block only runs when the up-front
    # collection failed and the Links agent had to fetch the page live.
    if scroll_mode and scrolled_html_file is None:
        if not link_extraction_code:
            print("\n⚠️  No link-extraction code available — "
                  "cannot expand dynamic content. Using initial links only.")
        else:
            full_html = await collect_listing_html(
                listing_url, mode=scroll_mode,
                max_rounds=scroll_rounds, load_more_selector=scroll_selector,
            )
            if full_html:
                expanded = run_link_extraction_code(link_extraction_code, full_html)
                page1_links_raw = expanded
                expanded, dropped_dyn = _filter_article_links(expanded, listing_url, pattern)
                expanded, recovery = recover_under_extracted_links(
                    page1_links_raw, expanded, full_html, listing_url, pattern,
                )
                if recovery:
                    recovery["page_num"] = 1
                    link_recovery.append(recovery)
                    print(
                        f"  ⚠ Link recovery guard added "
                        f"{recovery['recovered_links_added']} article link(s)"
                    )
                for d in dropped_dyn:
                    d["page_num"] = 1
                all_dropped.extend(dropped_dyn)
                page1_dropped.extend(dropped_dyn)
                print(f"  ✓ {len(expanded)} links after {pagination_type} "
                      f"(was {len(page1_links)} on initial load)")
                merged = {l["url"]: l for l in page1_links}
                for l in expanded:
                    merged.setdefault(l["url"], l)
                page1_links = list(merged.values())
                print(f"  ✓ {len(page1_links)} unique links total")

    page1_audit_html = full_html or await asyncio.to_thread(_quick_requests_fetch, listing_url)
    page1_evidence = _save_listing_audit(
        run_dir, 1, listing_url, page1_audit_html,
        page1_links_raw, page1_links, page1_dropped,
    )
    link_coverage.append(_make_link_coverage_record(
        1, listing_url, page1_links_raw, page1_links, page1_dropped,
        duplicate_count=max(len(page1_links) - len({l.get("url") for l in page1_links}), 0),
        new_count=len(page1_links),
        page_html=page1_audit_html,
        evidence=page1_evidence,
    ))
    _refresh_link_coverage_flags(link_coverage)

    # ══════════════════════════════════════════════════════════
    # Phase 2 cont.: Process page 1 articles (LLM calls #2..N)
    # ══════════════════════════════════════════════════════════
    all_extracted = []
    all_failures = []
    cluster_registry = {}  # sig_hash -> {code_file, code}
    seen_urls = set()

    preflight_links = _select_preflight_links(page1_links)
    print(f"\n⏳ Preflight-checking {len(preflight_links)} of "
          f"{len(page1_links)} candidate article links...")
    preflight_arts, preflight_fails = await fetch_articles(
        preflight_links, label=" [preflight]",
    )
    link_preflight = _assess_link_preflight(preflight_arts)
    link_preflight["fetch_failures"] = preflight_fails
    _atomic_json_write(os.path.join(run_dir, "link_preflight.json"), link_preflight)
    print(
        f"  Article evidence: {link_preflight['article_like_pages']}/"
        f"{link_preflight['sampled_pages']}; singleton structures: "
        f"{link_preflight['singleton_pages']}/"
        f"{link_preflight['sampled_pages']}"
    )

    if link_preflight["status"] == "rejected":
        error_msg = (
            "Link preflight rejected the candidate set: "
            f"{link_preflight['article_like_pages']}/"
            f"{link_preflight['sampled_pages']} sampled pages had article evidence "
            f"and {link_preflight['singleton_pages']}/"
            f"{link_preflight['sampled_pages']} had singleton DOM structures. "
            "The link extractor likely selected navigation or section pages."
        )
        print(f"\n❌ {error_msg}")
        write_results_md({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": run_dir, "model": model, "input_urls": input_urls,
            "pagination_type": pagination_type, "scroll_mode": scroll_mode,
            "scroll_rounds_requested": scroll_rounds if scroll_mode else None,
            "load_more_selector": scroll_selector, "requirements": requirements,
            "pages_requested": len(page_urls) + 1, "pages_processed": 0,
            "articles_extracted": 0, "articles_failed": 0, "clusters": 0,
            "links_dropped": all_dropped, "link_coverage": link_coverage,
            "link_recovery": link_recovery, "link_preflight": link_preflight,
            "llm_calls": _LLM_CALLS, "errors": [error_msg],
            "llm_calls_by_agent": dict(_LLM_CALLS_BY_AGENT),
            "tokens_by_agent": dict(_TOKENS_BY_AGENT),
            "elapsed_seconds": time.time() - run_start,
        })
        raise SystemExit(2)

    sampled_urls = {link["url"] for link in preflight_links}
    remaining_links = [
        link for link in page1_links if link["url"] not in sampled_urls
    ]
    remaining_arts, remaining_fails = await fetch_articles(
        remaining_links, label=" [page 1]",
    )
    arts = preflight_arts + remaining_arts
    all_failures.extend(preflight_fails)
    all_failures.extend(remaining_fails)

    for a in page1_links:
        seen_urls.add(a["url"])

    if arts:
        print("\n⏳ Clustering page 1 articles and extracting data...")
        ext, fl, cluster_registry = process_page_articles(
            arts, cluster_registry, api_key, model, requirements,
        )
        all_extracted.extend(ext)
        all_failures.extend(fl)

    # Incremental save after page 1
    max_pages = len(page_urls) + 1  # +1 for page 1
    progress = {
        "current_page": 1,
        "total_pages": max_pages,
        "extracted_count": len(all_extracted),
        "failed_count": len(all_failures),
        "pagination_type": pagination_type,
        "scroll_mode": scroll_mode,
        "scroll_rounds_requested": scroll_rounds if scroll_mode else None,
        "load_more_selector": scroll_selector,
        "link_recovery": link_recovery,
        "link_preflight": link_preflight,
        "cluster_registry": {
            sig: {"code_file": v["code_file"]} for sig, v in cluster_registry.items()
        },
    }
    save_incremental(run_dir, all_extracted, all_failures, progress, all_dropped, link_coverage)
    _atomic_json_write(os.path.join(run_dir, "extracted_links.json"), {
        "article_links": page1_links,
    })
    print(f"\n💾 Page 1 saved ({len(all_extracted)} extracted, {len(all_failures)} failed)")

    # ══════════════════════════════════════════════════════════
    # Phase 3: Subsequent pages (pages 2..N)
    # ══════════════════════════════════════════════════════════

    if page_urls:
        if not link_extraction_code:
            print("\n⚠️  No link-extraction code available — cannot process further pages.")
        else:
            for idx, page_url in enumerate(page_urls):
                page_num = idx + 2  # pages 2, 3, 4, ...
                print(f"\n{'─' * 60}")
                print(f"📄 Page {page_num}/{max_pages}: {page_url[:80]}")
                print(f"{'─' * 60}")

                # Fetch listing page HTML
                try:
                    page_html, _ = await fetch_page_structure(page_url)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"  ❌ Failed to fetch page: {e}\n{tb}")
                    page_evidence = _save_listing_audit(
                        run_dir, page_num, page_url, None, [], [], [],
                    )
                    link_coverage.append(_make_link_coverage_record(
                        page_num, page_url, [], [], [], fetch_error=f"{e}\n{tb}",
                        evidence=page_evidence,
                    ))
                    _refresh_link_coverage_flags(link_coverage)
                    progress["current_page"] = page_num
                    save_incremental(run_dir, all_extracted, all_failures, progress, all_dropped, link_coverage)
                    continue

                if not page_html:
                    print("  ❌ Empty response for listing page")
                    page_evidence = _save_listing_audit(
                        run_dir, page_num, page_url, "", [], [], [],
                    )
                    link_coverage.append(_make_link_coverage_record(
                        page_num, page_url, [], [], [], page_html="",
                        fetch_error="Empty response for listing page",
                        evidence=page_evidence,
                    ))
                    _refresh_link_coverage_flags(link_coverage)
                    progress["current_page"] = page_num
                    save_incremental(run_dir, all_extracted, all_failures, progress, all_dropped, link_coverage)
                    continue

                # Extract links using saved code
                raw_links = run_link_extraction_code(
                    link_extraction_code, page_html,
                )
                accepted_links, dropped_n = _filter_article_links(
                    raw_links, listing_url, pattern,
                )
                accepted_links, recovery = recover_under_extracted_links(
                    raw_links, accepted_links, page_html, listing_url, pattern,
                )
                if recovery:
                    recovery["page_num"] = page_num
                    link_recovery.append(recovery)
                    print(
                        f"  ⚠ Link recovery guard added "
                        f"{recovery['recovered_links_added']} article link(s)"
                    )
                for d in dropped_n:
                    d["page_num"] = page_num
                all_dropped.extend(dropped_n)
                if dropped_n:
                    print(f"  🧹 Dropped {len(dropped_n)} non-article "
                          f"(pagination/category) link(s)")
                print(f"  Found {len(accepted_links)} links on this page")

                # Deduplicate
                new_links = []
                duplicate_count = 0
                for lnk in accepted_links:
                    if lnk["url"] not in seen_urls:
                        seen_urls.add(lnk["url"])
                        new_links.append(lnk)
                    else:
                        duplicate_count += 1

                print(f"  {len(new_links)} new (after dedup)")

                page_evidence = _save_listing_audit(
                    run_dir, page_num, page_url, page_html,
                    raw_links, accepted_links, dropped_n,
                )
                link_coverage.append(_make_link_coverage_record(
                    page_num, page_url, raw_links, accepted_links, dropped_n,
                    duplicate_count=duplicate_count,
                    new_count=len(new_links),
                    page_html=page_html,
                    evidence=page_evidence,
                ))
                _refresh_link_coverage_flags(link_coverage)

                if "severe_page_overlap" in link_coverage[-1]["flags"]:
                    print(
                        f"  ⚠ Stopping pagination: {link_coverage[-1]['duplicate_ratio']:.0%} "
                        "of this page's links were already seen"
                    )
                    progress["current_page"] = page_num
                    save_incremental(
                        run_dir, all_extracted, all_failures, progress,
                        all_dropped, link_coverage,
                    )
                    break

                if not new_links:
                    progress["current_page"] = page_num
                    save_incremental(run_dir, all_extracted, all_failures, progress, all_dropped, link_coverage)
                    continue

                # Fetch article HTML + structural maps
                arts, fails = await fetch_articles(
                    new_links, label=f" [page {page_num}]",
                )
                all_failures.extend(fails)

                if arts:
                    ext, fl, cluster_registry = process_page_articles(
                        arts, cluster_registry, api_key, model, requirements,
                    )
                    all_extracted.extend(ext)
                    all_failures.extend(fl)

                # Incremental save after each page
                progress["current_page"] = page_num
                progress["extracted_count"] = len(all_extracted)
                progress["failed_count"] = len(all_failures)
                progress["link_recovery"] = link_recovery
                progress["cluster_registry"] = {
                    sig: {"code_file": v["code_file"]}
                    for sig, v in cluster_registry.items()
                }
                save_incremental(run_dir, all_extracted, all_failures, progress, all_dropped, link_coverage)
                print(f"  💾 Saved (total: {len(all_extracted)} extracted, "
                      f"{len(all_failures)} failed)")

                # Polite delay between pages
                if idx < len(page_urls) - 1:
                    time.sleep(1.5)

    # ══════════════════════════════════════════════════════════
    # Phase 4: Final save
    # ══════════════════════════════════════════════════════════
    cluster_info = {}
    for sig, v in cluster_registry.items():
        cluster_info[sig] = {"code_file": v.get("code_file", "")}
    _atomic_json_write(os.path.join(run_dir, "clusters.json"), cluster_info)

    save_incremental(run_dir, all_extracted, all_failures, {
        "status": "finished",
        "extracted_count": len(all_extracted),
        "failed_count": len(all_failures),
        "pagination_type": pagination_type,
        "scroll_mode": scroll_mode,
        "scroll_rounds_requested": scroll_rounds if scroll_mode else None,
        "load_more_selector": scroll_selector,
        "link_recovery": link_recovery,
        "link_preflight": link_preflight,
        "article_validation": list(_ARTICLE_VALIDATION_EVENTS),
    }, all_dropped, link_coverage)

    _print_summary(run_dir, all_extracted, all_failures)

    # ── Per-run stats for the paper (results.md) ─────────────
    pages_processed = (len(page_urls) + 1) if page_urls else 1
    write_results_md({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": run_dir,
        "model": model,
        "pagination_type": pagination_type,
        "scroll_mode": scroll_mode,
        "scroll_rounds_requested": scroll_rounds if scroll_mode else None,
        "load_more_selector": scroll_selector,
        "input_urls": input_urls,
        "requirements": requirements,
        "pages_requested": pages_processed,
        "pages_processed": pages_processed,
        "articles_extracted": len(all_extracted),
        "articles_failed": len(all_failures),
        "clusters": len(cluster_registry),
        "links_dropped": all_dropped,
        "link_coverage": link_coverage,
        "link_recovery": link_recovery,
        "link_preflight": link_preflight,
        "article_validation": list(_ARTICLE_VALIDATION_EVENTS),
        "llm_calls": _LLM_CALLS,
        "llm_calls_by_agent": dict(_LLM_CALLS_BY_AGENT),
        "elapsed_seconds": time.time() - run_start,
        "fetch_via": dict(_FETCH_VIA),
        "tokens_by_agent": dict(_TOKENS_BY_AGENT),
        "extracted_data": all_extracted,
        "errors": [f.get("reason", "") for f in all_failures],
    })


def _print_summary(run_dir, all_extracted, all_failures):
    print(f"\n{'=' * 60}")
    print(f"📊 FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  ✓ Successfully extracted: {len(all_extracted)} articles")
    print(f"  ❌ Failed: {len(all_failures)} articles")
    print(f"\n💾 Results in: {run_dir}/")
    print(f"   extracted_data_all.json  — all extracted data")
    print(f"   failed_links.json        — failures with reasons")
    print(f"   dropped_links.json       — links filtered out as non-article")
    print(f"   link_coverage.json       — page-level link recall sanity checks")
    print(f"   progress.json            — run metadata")
    print(f"\n🏁 Orchestrator finished!")


if __name__ == "__main__":
    asyncio.run(main())
