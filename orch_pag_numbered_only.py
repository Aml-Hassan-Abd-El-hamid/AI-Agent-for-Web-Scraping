"""Page-by-page pagination orchestrator (numbered pagination only).

Processes articles page-by-page instead of collecting all links upfront.
Page 1 establishes clusters and generates extraction code via LLM.
Subsequent pages reuse all generated code — new LLM calls only happen
for genuinely novel article structures.

Pagination is user-driven: the user provides page 1 URL, page 2 URL,
and the number of pages to scrape. The URL pattern is derived by
diffing the two URLs (string diff first, LLM fallback if needed).

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
from playwright.async_api import async_playwright

# Reuse utilities from Agent_for_single_page_gemma
from Agent_for_single_page_gemma import (
    fetch_page_structure,
    execute_extraction_code,
    list_available_models,
)

# --- Configuration ---
LINKS_AGENT_SCRIPT = "Links_Agent_gemma_cloudflare.py" #"Links_Agent_gemma.py"
AGENT_SCRIPT = "Agent_for_single_page_gemma.py"

# Running count of LLM generation calls made during a run (links agent,
# per-cluster article agent, and pagination-pattern LLM fallback). Used for
# the stats written to results.md. Reset at the start of each main() run.
_LLM_CALLS = 0


def _reset_llm_calls():
    global _LLM_CALLS
    _LLM_CALLS = 0


def _bump_llm_calls(n=1):
    global _LLM_CALLS
    _LLM_CALLS += n


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
    """Group articles by structural-map similarity."""
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
                ok = result.get("status") == "ok"
                if not ok:
                    # Surface the full subprocess logs so the real cause is visible
                    print(f"\n{'─' * 40} SUBPROCESS STDOUT {'─' * 40}")
                    print(proc.stdout)
                    print(f"{'─' * 40} SUBPROCESS STDERR {'─' * 40}")
                    print(proc.stderr or "(empty)")
                    print(f"{'─' * 98}")
                return ok, result

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
    _bump_llm_calls()
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
    _bump_llm_calls()
    return _call_agent_subprocess(cmd, timeout=300)


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
        _bump_llm_calls()
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

            code_text = ""
            if code_file and os.path.exists(code_file):
                with open(code_file, "r", encoding="utf-8") as f:
                    code_text = f.read()
            cluster_registry[sig] = {"code_file": code_file, "code": code_text}

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
    """Fetch HTML + structural maps for a list of {url, title} dicts."""
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
    lines.append(f"- **LLM calls:** {stats.get('llm_calls', 0)}")
    lines.append(f"- **Total time:** {mins}m {secs}s")

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
    print("🎯 ORCHESTRATOR — Numbered Pagination, Page-by-Page")
    print("=" * 60)

    _reset_llm_calls()
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
        input_urls.append(f"(file) {input_file}")
        pagination_type = "from file"

        if not article_links:
            print("❌ No article links found!")
            return

        _atomic_json_write(os.path.join(run_dir, "input_links.json"), article_links)

        requirements = input(
            "\n📝 What data to extract from each article?\n"
            "   (e.g., 'title, date, author, article body text')\n   → "
        ).strip() or "title, date, author, article body text"

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
        write_results_md({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": run_dir,
            "model": model,
            "pagination_type": pagination_type,
            "input_urls": input_urls,
            "requirements": requirements,
            "pages_requested": 1,
            "pages_processed": 1,
            "articles_extracted": len(all_extracted),
            "articles_failed": len(all_failures),
            "clusters": len(cluster_registry),
            "llm_calls": _LLM_CALLS,
            "elapsed_seconds": time.time() - run_start,
            "errors": [f.get("reason", "") for f in all_failures],
        })
        return

    # ── From here: listing-page flow ─────────────────────────
    listing_url = input("\n🌐 Enter the listing page URL: ").strip()
    if not listing_url:
        print("❌ URL is required!")
        return
    input_urls.append(listing_url)

    requirements = input(
        "\n📝 What data to extract from each article?\n"
        "   (e.g., 'title, date, author, article body text')\n   → "
    ).strip() or "title, date, author, article body text"

    # ══════════════════════════════════════════════════════════
    # Phase 1: Pagination setup (user-driven, no structural map)
    # ══════════════════════════════════════════════════════════
    page_urls = []  # URLs for pages 2, 3, 4, ...  (numbered pagination only)
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
        url1 = input("   → ").strip() or listing_url

        url2 = input("\n   Page 2 URL: ").strip()
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
                    start_num = p2_num if p2_num is not None else 2
                    page_urls = [
                        pattern.format(page=n)
                        for n in range(start_num, start_num + total_pages - 1)
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
    # Phase 2: Page 1 — extract links (LLM call #1)
    # ══════════════════════════════════════════════════════════
    print(f"\n⏳ Extracting article links from page 1: {listing_url}...")
    link_success, link_result = call_links_agent_cli(listing_url, api_key, model)

    if not link_success:
        error_msg = link_result.get("error", "Unknown error")
        print(f"\n❌ Links extraction failed:\n{error_msg}")
        write_results_md({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": run_dir, "model": model,
            "pagination_type": pagination_type, "input_urls": input_urls,
            "requirements": requirements,
            "pages_requested": len(page_urls) + 1, "pages_processed": 0,
            "articles_extracted": 0, "articles_failed": 0, "clusters": 0,
            "llm_calls": _LLM_CALLS, "elapsed_seconds": time.time() - run_start,
            "errors": [f"Links extraction failed: {error_msg}"],
        })
        return

    page1_links = link_result.get("data", {}).get("article_links", [])
    link_code_file = link_result.get("code_file", "")
    print(f"✓ Extracted {len(page1_links)} article links from page 1")

    if not page1_links:
        print("❌ No article links found on page 1!")
        write_results_md({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": run_dir, "model": model,
            "pagination_type": pagination_type, "input_urls": input_urls,
            "requirements": requirements,
            "pages_requested": len(page_urls) + 1, "pages_processed": 0,
            "articles_extracted": 0, "articles_failed": 0, "clusters": 0,
            "llm_calls": _LLM_CALLS, "elapsed_seconds": time.time() - run_start,
            "errors": ["No article links found on page 1"],
        })
        return

    # Load link-extraction code for reuse on later pages
    link_extraction_code = ""
    if link_code_file and os.path.exists(link_code_file):
        with open(link_code_file, "r", encoding="utf-8") as f:
            link_extraction_code = f.read()

    # ── Dynamic pagination: expand page 1 with all scrolled/loaded links ──
    if scroll_mode:
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
                print(f"  ✓ {len(expanded)} links after {pagination_type} "
                      f"(was {len(page1_links)} on initial load)")
                merged = {l["url"]: l for l in page1_links}
                for l in expanded:
                    merged.setdefault(l["url"], l)
                page1_links = list(merged.values())
                print(f"  ✓ {len(page1_links)} unique links total")

    # ══════════════════════════════════════════════════════════
    # Phase 2 cont.: Process page 1 articles (LLM calls #2..N)
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
    max_pages = len(page_urls) + 1  # +1 for page 1
    progress = {
        "current_page": 1,
        "total_pages": max_pages,
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
    })

    _print_summary(run_dir, all_extracted, all_failures)

    # ── Per-run stats for the paper (results.md) ─────────────
    pages_processed = (len(page_urls) + 1) if page_urls else 1
    write_results_md({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": run_dir,
        "model": model,
        "pagination_type": pagination_type,
        "input_urls": input_urls,
        "requirements": requirements,
        "pages_requested": pages_processed,
        "pages_processed": pages_processed,
        "articles_extracted": len(all_extracted),
        "articles_failed": len(all_failures),
        "clusters": len(cluster_registry),
        "llm_calls": _LLM_CALLS,
        "elapsed_seconds": time.time() - run_start,
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
    print(f"   progress.json            — run metadata")
    print(f"\n🏁 Orchestrator finished!")


if __name__ == "__main__":
    asyncio.run(main())
