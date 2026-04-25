"""Page-by-page pagination orchestrator.

Processes articles page-by-page instead of collecting all links upfront.
Page 1 establishes clusters and generates extraction code via LLM.
Subsequent pages reuse all generated code — new LLM calls only happen
for genuinely novel article structures.

LLM call budget:
  1  — pagination detection + loop code generation
  1  — link extraction code (Links Agent)
  N  — one per unique article-structure cluster (page 1 establishes most)
  +  — extra calls only if a later page surfaces a new structure
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
from bs4 import BeautifulSoup

# Reuse utilities from Agent_for_single_page_gemma
from Agent_for_single_page_gemma import (
    fetch_page_structure,
    execute_extraction_code,
    list_available_models,
)

# --- Configuration ---
LINKS_AGENT_SCRIPT = "Links_Agent_gemma.py"
AGENT_SCRIPT = "Agent_for_single_page_gemma.py"


# ═══════════════════════════════════════════════════════════════
# Utilities carried over from orch.py
# ═══════════════════════════════════════════════════════════════

def _struct_signature(smap, depth=0, max_depth=3):
    """Build a hashable string representing the skeleton of a structural map."""
    if depth >= max_depth or not isinstance(smap, list):
        return ""
    parts = []
    for node in smap:
        tag = node.get("tag", "")
        cls = node.get("attributes", {}).get("class", "")
        children_sig = _struct_signature(node.get("children", []), depth + 1, max_depth)
        parts.append(f"{tag}.{cls}({children_sig})")
    return "|".join(parts)


def _sig_hash(smap):
    """Return the 12-char MD5 cluster key for a structural map."""
    sig = _struct_signature(smap)
    return hashlib.md5(sig.encode()).hexdigest()[:12]


def cluster_by_structure(articles_with_maps):
    """Group articles by structural-map similarity.

    Returns dict: sig_hash -> list of article dicts.
    """
    clusters = defaultdict(list)
    for item in articles_with_maps:
        clusters[_sig_hash(item["structural_map"])].append(item)
    return dict(clusters)


def _call_agent_subprocess(cmd, timeout=300):
    """Run a CLI agent subprocess and parse ORCH_RESULT from stdout."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            env=env,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("ORCH_RESULT:"):
                payload = line[len("ORCH_RESULT:"):]
                result = json.loads(payload)
                return (result.get("status") == "ok"), result

        full_stderr = proc.stderr or ""
        full_stdout = proc.stdout or ""
        print(f"\n{'─' * 40} SUBPROCESS STDOUT {'─' * 40}")
        print(full_stdout)
        print(f"{'─' * 40} SUBPROCESS STDERR {'─' * 40}")
        print(full_stderr)
        print(f"{'─' * 98}")
        return False, {
            "status": "error",
            "error": f"No ORCH_RESULT in output. stderr: {full_stderr}  stdout(tail): {full_stdout[-3000:]}",
        }
    except subprocess.TimeoutExpired:
        return False, {"status": "error", "error": f"Subprocess timed out ({timeout}s)"}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n❌ Subprocess exception:\n{tb}")
        return False, {"status": "error", "error": f"{e}\n{tb}"}


def call_links_agent_cli(url, api_key, model="gemma-3-27b-it"):
    """Call Links_Agent_gemma.py via subprocess in CLI mode."""
    cmd = [
        sys.executable, LINKS_AGENT_SCRIPT,
        "--url", url,
        "--api-key", api_key,
        "--model", model,
    ]
    print(f"  🔧 Calling: python {LINKS_AGENT_SCRIPT} --url {url[:80]}...")
    return _call_agent_subprocess(cmd, timeout=300)


def call_agent_cli(url, api_key, requirements, model="gemma-3-27b-it"):
    """Call Agent_for_single_page_gemma.py via subprocess in CLI mode."""
    cmd = [
        sys.executable, AGENT_SCRIPT,
        "--url", url,
        "--api-key", api_key,
        "--requirements", requirements,
        "--model", model,
    ]
    print(f"  🔧 Calling: python {AGENT_SCRIPT} --url {url[:80]}...")
    return _call_agent_subprocess(cmd, timeout=300)


async def fetch_html_with_load_more(url):
    """Fetch a page and click 'Load More' until no more content loads."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(2000)

        clicks = 0
        while True:
            try:
                btn = await page.query_selector(
                    "button:has-text('Load More'), "
                    "button:has-text('load more'), "
                    "a:has-text('Load More'), "
                    "button:has-text('المزيد'), "
                    "a:has-text('المزيد'), "
                    ".load-more, .loadmore, [data-load-more]"
                )
                if not btn or not await btn.is_visible():
                    break
                await btn.click()
                clicks += 1
                print(f"    Clicked 'Load More' ({clicks})...")
                await page.wait_for_timeout(2000)
            except Exception as e:
                print(f"    ⚠️  Load-more click stopped: {e}")
                break

        html = await page.content()
        await browser.close()
        print(f"    ✓ Fully loaded after {clicks} click(s)")
        return html


async def fetch_html_with_infinite_scroll(url):
    """Fetch a page and scroll to the bottom until no more content loads."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(2000)

        scrolls = 0
        while True:
            previous_height = await page.evaluate("document.body.scrollHeight")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2000)
            new_height = await page.evaluate("document.body.scrollHeight")
            scrolls += 1
            if new_height == previous_height:
                break
            print(f"    Scrolled ({scrolls})... height: {previous_height} → {new_height}")

        html = await page.content()
        await browser.close()
        print(f"    ✓ Fully loaded after {scrolls} scroll(s)")
        return html


# ═══════════════════════════════════════════════════════════════
# NEW: Enhanced pagination detection — LLM generates code
# ═══════════════════════════════════════════════════════════════

_PAG_FEW_SHOT_MAP = """\
[
  {
    "tag": "nav",
    "attributes": {"class": "pagination"},
    "children": [
      {"tag": "a", "attributes": {"class": "page-numbers current", "href": "https://example.com/news/"}, "text_snippet": "1"},
      {"tag": "a", "attributes": {"class": "page-numbers", "href": "https://example.com/news/page/2/"}, "text_snippet": "2"},
      {"tag": "a", "attributes": {"class": "page-numbers", "href": "https://example.com/news/page/3/"}, "text_snippet": "3"},
      {"tag": "a", "attributes": {"class": "page-numbers dots"}, "text_snippet": "…"},
      {"tag": "a", "attributes": {"class": "page-numbers", "href": "https://example.com/news/page/12/"}, "text_snippet": "12"},
      {"tag": "a", "attributes": {"class": "next page-numbers", "href": "https://example.com/news/page/2/"}, "text_snippet": "Next →"}
    ]
  }
]"""

_PAG_FEW_SHOT_CODE = """\
def get_all_page_urls(first_page_url):
    # Pattern derived from href: https://example.com/news/page/{N}/
    # Highest page number visible in pagination: 12
    urls = []
    for page_num in range(2, 12 + 1):
        urls.append("https://example.com/news/page/" + str(page_num) + "/")
    return urls"""


async def detect_pagination_with_llm(structural_map_json, page_url, api_key,
                                      model="gemma-3-27b-it"):
    """Ask the LLM to identify pagination type and generate loop code.

    Returns a dict:
        {type, code}  where type is "none"|"numbered"|"load-more"|"infinite-scroll"
        and code is a Python function string (or None for non-numbered types).
    Returns None on failure.
    """
    import google.generativeai as genai

    # ── Trim the structural map to pagination-relevant parts ──
    # Full maps can be 100K+ chars, causing the LLM to ramble or hit limits.
    # We extract only nodes likely to contain pagination controls.
    trimmed_map = _extract_pagination_region(structural_map_json)
    map_size = len(trimmed_map)
    print(f"    ℹ  Structural map for pagination: {map_size:,} chars "
          f"(original: {len(structural_map_json):,} chars)")

    # ── Check if hrefs are present in pagination nodes ──
    has_hrefs = '"href"' in trimmed_map
    href_note = ""
    if not has_hrefs:
        href_note = (
            "\nIMPORTANT: The pagination links have NO href attributes (they are JavaScript-driven). "
            "You CANNOT derive a URL pattern. Return type 'numbered' with code set to null. "
            "Do NOT try to guess URLs.\n"
        )

    prompt = f"""<start_of_turn>user
You are a web scraping expert. Identify the pagination type from this structural map.

PAGE URL: {page_url}
{href_note}
STRUCTURAL MAP (pagination-relevant section):
{trimmed_map}

Pagination types:
- "numbered": nav/div/ul with page-number links
- "load-more": button/link with "Load More" / "Show More" / "المزيد"
- "infinite-scroll": lazy-loading, no visible pagination
- "none": single page

If "numbered" AND href attributes are present, also write:
  def get_all_page_urls(first_page_url):
      # Return list of page URLs for pages 2..N
      # Use ONLY string ops, no imports.
      return urls
If "numbered" but NO href attributes: set code to null.

FEW-SHOT EXAMPLE:
{{"type": "numbered", "code": {json.dumps(_PAG_FEW_SHOT_CODE)}}}

Respond with ONLY a JSON object. No explanation, no markdown.
{{"type": "...", "code": "..." or null}}
<end_of_turn>
<start_of_turn>model
"""

    try:
        genai.configure(api_key=api_key)
        llm = genai.GenerativeModel(model)
        gen_config = genai.GenerationConfig(
            max_output_tokens=512,
            temperature=0.0,
            response_mime_type="application/json",
        )
        t0 = time.time()
        response = llm.generate_content(prompt, generation_config=gen_config)
        elapsed = time.time() - t0
        print(f"    ℹ  LLM responded in {elapsed:.1f}s")

        raw_text = response.text
        if raw_text is None:
            print(f"    ⚠️  LLM returned None. Finish reason: {response.candidates}")
            return None

        text = raw_text.strip()
        print(f"    ℹ  Raw LLM response ({len(text)} chars):\n{text[:2000]}")

        # Try to parse the response as JSON (multiple strategies)
        parsed = _try_parse_json(text)
        if parsed is None:
            print("    ⚠️  Could not extract valid JSON from LLM response.")
            return None

        pag_type = parsed.get("type", "none")
        if pag_type not in ("none", "numbered", "load-more", "infinite-scroll"):
            print(f"    ⚠️  Unexpected pagination type from LLM: {pag_type!r}")
            return None

        return {
            "type": pag_type,
            "code": parsed.get("code"),
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"    ⚠️  LLM pagination detection failed: {e}\n{tb}")
        return None


def _try_parse_json(text):
    """Try multiple strategies to extract a JSON object from LLM output.

    1. Direct parse
    2. Strip markdown fences then parse
    3. Find first {...} block via brace matching
    """
    # Strategy 1: direct
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: strip markdown fences
    cleaned = text
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        cleaned = cleaned[first_nl + 1:] if first_nl != -1 else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:cleaned.rfind("```")]
    cleaned = cleaned.strip()
    if cleaned:
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: find the first { and match braces
    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        break
    return None


def _extract_pagination_region(structural_map_json):
    """Extract only pagination-relevant nodes from a structural map.

    Looks for nodes containing pagination keywords in their tag, class,
    or text_snippet. Falls back to the last 15K chars if nothing found.
    """
    PAG_KEYWORDS = {
        "pagination", "pager", "page-numbers", "page-item", "page-link",
        "nav-links", "wp-pagenavi", "next", "prev", "previous",
        "load-more", "loadmore", "show-more",
        "المزيد",  # Arabic "more"
    }

    try:
        full_map = json.loads(structural_map_json)
    except (json.JSONDecodeError, ValueError):
        # Can't parse — return truncated original
        return structural_map_json[-15000:]

    matches = []
    _find_pagination_nodes(full_map, matches, PAG_KEYWORDS)

    if matches:
        trimmed = json.dumps(matches, indent=2, ensure_ascii=False)
        # If still huge, cap it
        if len(trimmed) > 20000:
            trimmed = trimmed[:20000] + "\n... (truncated)"
        return trimmed

    # No pagination-specific nodes found — return last portion of the full map
    # (pagination is typically near the bottom of the page)
    if len(structural_map_json) > 15000:
        return "... (trimmed to last 15000 chars)\n" + structural_map_json[-15000:]
    return structural_map_json


def _find_pagination_nodes(nodes, results, keywords, depth=0, max_depth=15):
    """Recursively find nodes whose class/text/tag matches pagination keywords."""
    if depth > max_depth or not isinstance(nodes, list):
        return
    for node in nodes:
        if not isinstance(node, dict):
            continue
        cls = node.get("attributes", {}).get("class", "").lower()
        tag = node.get("tag", "").lower()
        text = node.get("text_snippet", "").lower()
        role = node.get("attributes", {}).get("role", "").lower()

        if any(kw in cls or kw in text or kw in role for kw in keywords) or tag == "nav":
            results.append(node)
        else:
            # Keep searching children
            _find_pagination_nodes(node.get("children", []), results, keywords, depth + 1, max_depth)


# ═══════════════════════════════════════════════════════════════
# NEW: HTML-based pagination fallback (from orch.py)
# ═══════════════════════════════════════════════════════════════

async def detect_pagination_html(url):
    """Analyze a listing page's HTML to find numbered pagination links.

    Returns (url_pattern_with_{page}, total_pages) or (None, None).
    """
    try:
        html, _ = await fetch_page_structure(url)
        if not html:
            return None, None
    except Exception as e:
        print(f"    ⚠️  HTML pagination detection fetch failed: {e}")
        return None, None

    soup = BeautifulSoup(html, "html.parser")

    pag_selectors = [
        "nav.pagination", "ul.pagination", "div.pagination",
        "nav.nav-links", "div.nav-links",
        ".wp-pagenavi", ".page-numbers", ".pager",
        "[role='navigation']",
    ]

    candidate_links = []
    for sel in pag_selectors:
        for container in soup.select(sel):
            candidate_links.extend(container.find_all("a", href=True))

    if not candidate_links:
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            if text.isdigit() and int(text) >= 2:
                candidate_links.append(a)

    if not candidate_links:
        return None, None

    base = urlparse(url).scheme + "://" + urlparse(url).netloc
    page_nums = []

    for a in candidate_links:
        href = urljoin(base, a["href"])
        text = a.get_text(strip=True)
        num = None
        if text.isdigit():
            num = int(text)
        else:
            m = _re.search(r'[/?&]page[=/](\d+)', href)
            if m:
                num = int(m.group(1))
            else:
                m = _re.search(r'[/?&]p[=/](\d+)', href)
                if m:
                    num = int(m.group(1))
        if num and num >= 2:
            page_nums.append((num, href))

    if not page_nums:
        return None, None

    page_nums.sort()
    page2_num, page2_url = page_nums[0]

    pattern = _re.sub(
        r'(/page/)' + str(page2_num) + r'(/|$)',
        r'\g<1>{page}\2', page2_url,
    )
    if pattern == page2_url:
        pattern = _re.sub(
            r'([?&]page=)' + str(page2_num) + r'(&|$)',
            r'\g<1>{page}\2', page2_url,
        )
    if pattern == page2_url:
        pattern = _re.sub(
            r'(/p/)' + str(page2_num) + r'(/|$)',
            r'\g<1>{page}\2', page2_url,
        )
    if pattern == page2_url:
        return None, None

    total_pages = max(num for num, _ in page_nums)
    return pattern, total_pages


# ═══════════════════════════════════════════════════════════════
# NEW: Execute LLM-generated pagination code in sandbox
# ═══════════════════════════════════════════════════════════════

def execute_pagination_code(code, first_page_url):
    """Run the LLM-generated get_all_page_urls() in a restricted sandbox.

    Returns (success: bool, result_or_error).
    result is a list of URL strings on success.
    """
    SAFE_BUILTINS = {
        "True": True, "False": False, "None": None,
        "int": int, "float": float, "str": str, "bool": bool,
        "list": list, "dict": dict, "tuple": tuple, "set": set,
        "range": range, "len": len, "enumerate": enumerate,
        "zip": zip, "sorted": sorted, "reversed": reversed,
        "min": min, "max": max, "sum": sum, "abs": abs,
        "isinstance": isinstance, "print": print,
    }

    namespace = {"__builtins__": SAFE_BUILTINS}

    try:
        exec(code, namespace)
    except Exception as e:
        return False, f"Code compilation failed: {e}"

    fn = namespace.get("get_all_page_urls")
    if not callable(fn):
        return False, "No callable get_all_page_urls() found in generated code"

    try:
        urls = fn(first_page_url)
    except Exception as e:
        return False, f"get_all_page_urls() raised: {e}"

    if not isinstance(urls, list):
        return False, f"Expected list, got {type(urls).__name__}"

    # Validate every element is a non-empty string
    clean = [u for u in urls if isinstance(u, str) and u.startswith("http")]
    if not clean:
        return False, "get_all_page_urls() returned no valid URLs"

    return True, clean


# ═══════════════════════════════════════════════════════════════
# NEW: Incremental save (atomic writes)
# ═══════════════════════════════════════════════════════════════

def _atomic_json_write(path, data):
    """Write JSON to *path* via a temp file to avoid corruption on crash."""
    dir_name = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)          # atomic on same filesystem
    except Exception:
        os.unlink(tmp)
        raise


def save_incremental(run_dir, all_extracted, all_failures, progress_info):
    """Persist current state so nothing is lost on crash."""
    _atomic_json_write(
        os.path.join(run_dir, "extracted_data_all.json"), all_extracted,
    )
    _atomic_json_write(
        os.path.join(run_dir, "failed_links.json"), all_failures,
    )
    _atomic_json_write(
        os.path.join(run_dir, "progress.json"), progress_info,
    )


# ═══════════════════════════════════════════════════════════════
# NEW: Process a batch of articles against the cluster registry
# ═══════════════════════════════════════════════════════════════

def process_page_articles(articles_with_maps, cluster_registry,
                          api_key, model, requirements):
    """Match articles to known clusters; call the agent only for new ones.

    Args:
        articles_with_maps: list of {url, title, html_content, structural_map}
        cluster_registry:   {sig_hash: {"code_file": str, "code": str}}
        api_key, model, requirements: forwarded to the agent

    Returns (extracted, failures, updated_cluster_registry).
    """
    extracted = []
    failures = []

    # Group this batch by structure
    batched = defaultdict(list)
    for art in articles_with_maps:
        batched[_sig_hash(art["structural_map"])].append(art)

    for sig, members in batched.items():
        if sig in cluster_registry:
            # ── Known cluster: reuse saved code ──────────────
            code = cluster_registry[sig]["code"]
            for m in members:
                ok, result = execute_extraction_code(code, m["html_content"])
                if ok and isinstance(result, dict):
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
            # ── New cluster: call agent on representative ────
            rep = members[0]
            rest = members[1:]
            print(f"\n  🆕 New cluster {sig} ({len(members)} article(s))")
            print(f"     Representative: {rep['title'][:60]}")

            success, agent_result = call_agent_cli(
                rep["url"], api_key, requirements, model,
            )

            if not success:
                err = agent_result.get("error", "Unknown agent error")
                print(f"     ❌ Agent failed:\n{err}")
                for m in members:
                    failures.append({
                        "url": m["url"], "title": m["title"],
                        "reason": f"Agent failed on new cluster representative: {err}",
                    })
                continue

            rep_data = agent_result.get("data", {})
            code_file = agent_result.get("code_file", "")
            print(f"     ✓ Agent succeeded! Code: {code_file}")

            extracted.append({
                "url": rep["url"], "title": rep["title"], "data": rep_data,
            })

            # Load code and register cluster
            code_text = ""
            if code_file and os.path.exists(code_file):
                with open(code_file, "r", encoding="utf-8") as f:
                    code_text = f.read()
            cluster_registry[sig] = {"code_file": code_file, "code": code_text}

            # Apply to remaining members
            if rest and code_text:
                for m in rest:
                    ok, result = execute_extraction_code(code_text, m["html_content"])
                    if ok and isinstance(result, dict):
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

async def fetch_articles(article_links, label=""):
    """Fetch HTML + structural maps for a list of {url, title} dicts.

    Returns (articles_with_maps, fetch_failures).
    """
    articles_with_maps = []
    fetch_failures = []
    total = len(article_links)

    for i, link in enumerate(article_links):
        url = link["url"]
        title = link.get("title", "")
        print(f"  [{i+1}/{total}]{label} Fetching: {title[:50]}...")

        try:
            html_content, structural_map = await fetch_page_structure(url)
            if html_content and structural_map:
                articles_with_maps.append({
                    "url": url,
                    "title": title,
                    "html_content": html_content,
                    "structural_map": structural_map,
                })
                print(f"    ✓ OK ({len(html_content)} bytes)")
            else:
                fetch_failures.append({"url": url, "title": title, "reason": "Empty response"})
                print(f"    ❌ Empty response")
        except Exception as e:
            tb = traceback.format_exc()
            fetch_failures.append({"url": url, "title": title, "reason": f"{e}\n{tb}"})
            print(f"    ❌ {e}\n{tb}")

    return articles_with_maps, fetch_failures


# ═══════════════════════════════════════════════════════════════
# Helper: extract links from a page using saved link-extraction code
# ═══════════════════════════════════════════════════════════════

def run_link_extraction_code(code, html):
    """Execute the link-extraction code on *html* and return article_links list."""
    ok, result = execute_extraction_code(code, html)
    if ok and isinstance(result, dict):
        return result.get("article_links", [])
    return []


# ═══════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("🎯 ORCHESTRATOR (Paginated) — Page-by-Page Extraction")
    print("=" * 60)

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

    # ── Step 1: Choose link source ───────────────────────────
    print("\n How do you want to get article links?")
    print("   [1] Extract from a listing page (calls Links_Agent_gemma.py)")
    print("   [2] Load from an existing JSON file")
    link_choice = input("   → ").strip()

    if link_choice == "2":
        # Load from file — no pagination, process as flat batch
        input_file = input(
            "\n📂 Enter JSON file path [Enter for extracted_data_pchrgaza_org.json]: "
        ).strip() or "extracted_data_pchrgaza_org.json"

        print(f"\n⏳ Loading links from {input_file}...")
        with open(input_file, "r", encoding="utf-8") as f:
            links_data = json.load(f)
        article_links = links_data.get("article_links", [])
        print(f"✓ Found {len(article_links)} article links")

        if not article_links:
            print("❌ No article links found!")
            return

        _atomic_json_write(os.path.join(run_dir, "input_links.json"), article_links)

        requirements = input(
            "\n📝 What data to extract from each article?\n"
            "   (e.g., 'title, date, author, article body text')\n   → "
        ).strip() or "title, date, author, article body text"

        # Process everything as a single "page"
        all_extracted = []
        all_failures = []
        cluster_registry = {}

        print(f"\n⏳ Fetching structural maps for {len(article_links)} articles...")
        arts, fails = await fetch_articles(article_links)
        all_failures.extend(fails)

        if arts:
            ext, fl, cluster_registry = process_page_articles(
                arts, cluster_registry, api_key, model, requirements,
            )
            all_extracted.extend(ext)
            all_failures.extend(fl)

        save_incremental(run_dir, all_extracted, all_failures, {
            "status": "finished",
            "extracted_count": len(all_extracted),
            "failed_count": len(all_failures),
        })
        _print_summary(run_dir, all_extracted, all_failures)
        return

    # ── From here: listing-page flow ─────────────────────────
    listing_url = input("\n🌐 Enter the listing page URL: ").strip()
    if not listing_url:
        print("❌ URL is required!")
        return

    requirements = input(
        "\n📝 What data to extract from each article?\n"
        "   (e.g., 'title, date, author, article body text')\n   → "
    ).strip() or "title, date, author, article body text"

    # ══════════════════════════════════════════════════════════
    # Phase 1: Pagination detection (LLM call #1)
    # ══════════════════════════════════════════════════════════
    print("\n📄 Does this listing page have multiple pages (pagination)?")
    print("   [1] Yes — detect pagination automatically (LLM call)")
    print("   [2] No  — single page, skip pagination")
    pag_choice = input("   → ").strip()

    pag_info = None
    pag_type = "none"

    if pag_choice == "1":
        print("\n⏳ Fetching listing page structure...")
        t0 = time.time()
        html_content, structural_map = await fetch_page_structure(listing_url)
        elapsed = time.time() - t0
        print(f"   ✓ Page fetched in {elapsed:.1f}s")

        structural_map_json = json.dumps(structural_map, indent=2) if structural_map else ""

        if structural_map_json:
            print("⏳ Asking LLM to detect pagination type...")
            pag_info = await detect_pagination_with_llm(
                structural_map_json, listing_url, api_key, model,
            )

        if pag_info:
            pag_type = pag_info["type"]
            print(f"\n   ✓ LLM detected pagination: {pag_type}")
            if pag_type == "numbered" and pag_info.get("code"):
                print(f"     (pagination code generated)")
            elif pag_type == "numbered" and not pag_info.get("code"):
                print(f"     ⚠️  Numbered pagination detected, but URLs are JavaScript-driven (no hrefs).")
                print(f"     Will use HTML-based detection or ask you for the URL pattern later.")
        else:
            pag_type = "none"
            print("\n   ⚠️  Pagination detection failed or returned nothing.")
            print("   Proceeding as single page. You can still enter pagination info manually later.")
    else:
        print("   ✓ Skipping pagination — will process page 1 only.")

    # ══════════════════════════════════════════════════════════
    # Phase 2: Page 1 — extract links (LLM call #2)
    # ══════════════════════════════════════════════════════════
    print(f"\n⏳ Extracting article links from page 1: {listing_url}...")
    link_success, link_result = call_links_agent_cli(listing_url, api_key, model)

    if not link_success:
        error_msg = link_result.get("error", "Unknown error")
        print(f"\n❌ Links extraction failed:\n{error_msg}")
        return

    page1_links = link_result.get("data", {}).get("article_links", [])
    link_code_file = link_result.get("code_file", "")
    print(f"✓ Extracted {len(page1_links)} article links from page 1")

    if not page1_links:
        print("❌ No article links found on page 1!")
        return

    # Load link-extraction code for reuse on later pages
    link_extraction_code = ""
    if link_code_file and os.path.exists(link_code_file):
        with open(link_code_file, "r", encoding="utf-8") as f:
            link_extraction_code = f.read()

    # ══════════════════════════════════════════════════════════
    # Phase 2 cont.: Process page 1 articles (LLM calls #3..N)
    # ══════════════════════════════════════════════════════════
    all_extracted = []
    all_failures = []
    cluster_registry = {}  # sig_hash -> {code_file, code}
    seen_urls = set()

    print(f"\n⏳ Fetching structural maps for page 1 ({len(page1_links)} articles)...")
    arts, fails = await fetch_articles(page1_links, label=" [page 1]")
    all_failures.extend(fails)

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
    progress = {
        "current_page": 1,
        "total_pages": 1,
        "extracted_count": len(all_extracted),
        "failed_count": len(all_failures),
        "cluster_registry": {
            sig: {"code_file": v["code_file"]} for sig, v in cluster_registry.items()
        },
    }
    save_incremental(run_dir, all_extracted, all_failures, progress)
    _atomic_json_write(os.path.join(run_dir, "extracted_links.json"), {
        "article_links": page1_links,
    })
    print(f"\n💾 Page 1 saved ({len(all_extracted)} extracted, {len(all_failures)} failed)")

    # ══════════════════════════════════════════════════════════
    # Phase 3: Subsequent pages
    # ══════════════════════════════════════════════════════════

    if pag_type == "numbered":
        # ── Numbered pagination ──────────────────────────────
        pagination_code = pag_info.get("code") if pag_info else None
        page_urls = []

        if pagination_code:
            ok, result = execute_pagination_code(pagination_code, listing_url)
            if ok:
                page_urls = result
                print(f"\n✓ Pagination code generated {len(page_urls)} page URL(s)")
            else:
                print(f"\n⚠️  Pagination code failed: {result}")
                print("   Falling back to HTML-based detection...")

        # Fallback: HTML-based pattern detection
        if not page_urls:
            pattern, total_pages = await detect_pagination_html(listing_url)
            if pattern and total_pages:
                page_urls = [pattern.format(page=p) for p in range(2, total_pages + 1)]
                print(f"   ✓ HTML detection: {total_pages} pages, pattern: {pattern}")
            else:
                print("   ⚠️  Could not auto-detect pagination.")
                total_str = input("   📄 How many pages total? [Enter to skip] → ").strip()
                if total_str.isdigit() and int(total_str) >= 2:
                    total_pages = int(total_str)
                    default_pat = listing_url.rstrip("/") + "/page/{page}/"
                    print(f"   URL pattern (use {{page}} as placeholder):")
                    print(f"   [Enter] Use: {default_pat}")
                    pat = input("   → ").strip() or default_pat
                    page_urls = [pat.format(page=p) for p in range(2, total_pages + 1)]

        if not page_urls:
            print("   ℹ  No extra pages to process.")
        else:
            total_available = len(page_urls) + 1  # +1 for page 1

            # Ask user how many pages to scrape
            print(f"\n📄 Found {total_available} total pages (page 1 already processed).")
            print(f"   How many pages do you want to scrape?")
            print(f"   [Enter] All {total_available} pages")
            print(f"   [number] Only scrape first N pages")
            limit_str = input("   → ").strip()

            if limit_str.isdigit() and int(limit_str) >= 1:
                max_pages = int(limit_str)
                if max_pages > total_available:
                    print(f"   ⚠️  You requested {max_pages} pages but only "
                          f"{total_available} exist. Extracting {total_available}.")
                    max_pages = total_available
                # page 1 already done, so we need max_pages - 1 more
                page_urls = page_urls[:max_pages - 1]
                print(f"   ✓ Will scrape pages 1–{max_pages} "
                      f"({len(page_urls)} remaining).")
            else:
                max_pages = total_available
                print(f"   ✓ Will scrape all {total_available} pages.")

            progress["total_pages"] = max_pages

            if not link_extraction_code:
                print("⚠️  No link-extraction code available — cannot process further pages.")
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
                        continue

                    if not page_html:
                        print("  ❌ Empty response for listing page")
                        continue

                    # Extract links using saved code
                    new_links_raw = run_link_extraction_code(
                        link_extraction_code, page_html,
                    )
                    print(f"  Found {len(new_links_raw)} links on this page")

                    # Deduplicate
                    new_links = []
                    for lnk in new_links_raw:
                        if lnk["url"] not in seen_urls:
                            seen_urls.add(lnk["url"])
                            new_links.append(lnk)

                    print(f"  {len(new_links)} new (after dedup)")

                    if not new_links:
                        # Update progress even for empty pages
                        progress["current_page"] = page_num
                        save_incremental(run_dir, all_extracted, all_failures, progress)
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
                    progress["cluster_registry"] = {
                        sig: {"code_file": v["code_file"]}
                        for sig, v in cluster_registry.items()
                    }
                    save_incremental(run_dir, all_extracted, all_failures, progress)
                    print(f"  💾 Saved (total: {len(all_extracted)} extracted, "
                          f"{len(all_failures)} failed)")

                    # Polite delay between pages
                    if idx < len(page_urls) - 1:
                        time.sleep(1.5)

    elif pag_type == "load-more":
        # ── Load-more: click until done, then process ────────
        print(f"\n⏳ Loading all content via 'Load More' from {listing_url}...")
        full_html = await fetch_html_with_load_more(listing_url)

        if link_extraction_code and full_html:
            extra_links = run_link_extraction_code(link_extraction_code, full_html)
            new_links = [l for l in extra_links if l["url"] not in seen_urls]
            for l in new_links:
                seen_urls.add(l["url"])
            print(f"✓ {len(new_links)} new links after load-more")

            if new_links:
                arts, fails = await fetch_articles(new_links, label=" [load-more]")
                all_failures.extend(fails)
                if arts:
                    ext, fl, cluster_registry = process_page_articles(
                        arts, cluster_registry, api_key, model, requirements,
                    )
                    all_extracted.extend(ext)
                    all_failures.extend(fl)
                save_incremental(run_dir, all_extracted, all_failures, {
                    "status": "load-more-done",
                    "extracted_count": len(all_extracted),
                    "failed_count": len(all_failures),
                })

    elif pag_type == "infinite-scroll":
        # ── Infinite scroll: scroll until done, then process ─
        print(f"\n⏳ Scrolling to load all content from {listing_url}...")
        full_html = await fetch_html_with_infinite_scroll(listing_url)

        if link_extraction_code and full_html:
            extra_links = run_link_extraction_code(link_extraction_code, full_html)
            new_links = [l for l in extra_links if l["url"] not in seen_urls]
            for l in new_links:
                seen_urls.add(l["url"])
            print(f"✓ {len(new_links)} new links after infinite scroll")

            if new_links:
                arts, fails = await fetch_articles(new_links, label=" [scroll]")
                all_failures.extend(fails)
                if arts:
                    ext, fl, cluster_registry = process_page_articles(
                        arts, cluster_registry, api_key, model, requirements,
                    )
                    all_extracted.extend(ext)
                    all_failures.extend(fl)
                save_incremental(run_dir, all_extracted, all_failures, {
                    "status": "infinite-scroll-done",
                    "extracted_count": len(all_extracted),
                    "failed_count": len(all_failures),
                })

    # ══════════════════════════════════════════════════════════
    # Phase 4: Final save
    # ══════════════════════════════════════════════════════════
    # Save cluster info
    cluster_info = {}
    for sig, v in cluster_registry.items():
        cluster_info[sig] = {"code_file": v.get("code_file", "")}
    _atomic_json_write(os.path.join(run_dir, "clusters.json"), cluster_info)

    save_incremental(run_dir, all_extracted, all_failures, {
        "status": "finished",
        "extracted_count": len(all_extracted),
        "failed_count": len(all_failures),
    })

    _print_summary(run_dir, all_extracted, all_failures)


def _print_summary(run_dir, all_extracted, all_failures):
    print(f"\n{'=' * 60}")
    print(f"📊 FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  ✓ Successfully extracted: {len(all_extracted)} articles")
    print(f"  ❌ Failed: {len(all_failures)} articles")
    print(f"\n💾 Results in: {run_dir}/")
    print(f"   extracted_data_all.json  — all extracted data")
    print(f"   failed_links.json        — failures with reasons")
    print(f"   progress.json            — run metadata")
    print(f"\n🏁 Orchestrator finished!")


if __name__ == "__main__":
    asyncio.run(main())
