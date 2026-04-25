"""Take the output of Linkes_Agent_gemma.py and do the following:
1- Run the generated code to extract the article links and save them to a file.
2- For each extracted article link, get the structural map of the article.
3- Cluster the articles based on the similarity of their structural maps.
4- From each cluster, take a representative article link, call "Agent_for_single_page_gemma.py" on it, test if the the code work on that link.
5- For the rest of the cluster, take the same code that worked for the representative article, and test it on the rest of the links in the cluster, if it worked, save the extracted data, if not, save the failed link to a file for future analysis.
6- Finally, save the successfully extracted data to a file, and the failed links to another file for future analysis.
"""

import asyncio
import json
import os
import subprocess
import sys
import hashlib
import time
from collections import defaultdict
from datetime import datetime
import re as _re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup


async def detect_pagination(url):
    """Analyze a listing page's HTML to find pagination links.

    Looks for common pagination nav patterns (numbered page links) and
    extracts the URL pattern + total page count.

    Returns (url_pattern, total_pages) or (None, None) if not detected.
    url_pattern uses {page} as the placeholder, e.g.:
        https://example.com/articles/page/{page}/
    """
    try:
        html, _ = await fetch_page_structure(url)
        if not html:
            return None, None
    except Exception:
        return None, None

    soup = BeautifulSoup(html, "html.parser")

    # Common selectors where pagination lives
    pag_selectors = [
        "nav.pagination", "ul.pagination", "div.pagination",
        "nav.nav-links", "div.nav-links",
        ".wp-pagenavi", ".page-numbers", ".pager",
        "[role='navigation']",
    ]

    # Collect candidate <a> tags from pagination areas
    candidate_links = []
    for sel in pag_selectors:
        for container in soup.select(sel):
            candidate_links.extend(container.find_all("a", href=True))

    # Fallback: look for any <a> whose text is a number (2, 3, …)
    if not candidate_links:
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            if text.isdigit() and int(text) >= 2:
                candidate_links.append(a)

    if not candidate_links:
        return None, None

    # Parse hrefs to find a numbered pattern
    base = urlparse(url).scheme + "://" + urlparse(url).netloc
    page_nums = []  # (page_number, full_url)

    for a in candidate_links:
        href = urljoin(base, a["href"])
        text = a.get_text(strip=True)
        num = None

        # Try getting page number from link text first
        if text.isdigit():
            num = int(text)
        else:
            # Try extracting from href: /page/3/, ?page=3, &p=3, etc.
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

    # Use the link for page 2 to derive the pattern
    page_nums.sort()
    page2_num, page2_url = page_nums[0]

    # Replace the page number in the URL with {page}
    # Try path-based: /page/2/ → /page/{page}/
    pattern = _re.sub(
        r'(/page/)' + str(page2_num) + r'(/|$)',
        r'\g<1>{page}\2',
        page2_url
    )
    # Try query-based: ?page=2 or &page=2
    if pattern == page2_url:
        pattern = _re.sub(
            r'([?&]page=)' + str(page2_num) + r'(&|$)',
            r'\g<1>{page}\2',
            page2_url
        )
    # Try /p/2
    if pattern == page2_url:
        pattern = _re.sub(
            r'(/p/)' + str(page2_num) + r'(/|$)',
            r'\g<1>{page}\2',
            page2_url
        )

    if pattern == page2_url:
        # Could not find where to put {page} — give up
        return None, None

    # Total pages = highest page number found
    total_pages = max(num for num, _ in page_nums)

    return pattern, total_pages


async def detect_pagination_with_llm(structural_map_json, page_url, api_key, model="gemma-3-27b-it"):
    """Ask the LLM to identify pagination type, URL pattern, and page count
    from a listing page's structural map.

    Returns a dict: {type, url_pattern, total_pages} or None on failure.
    type is one of: "none", "numbered", "load-more", "infinite-scroll"
    """
    import google.generativeai as genai

    prompt = f"""<start_of_turn>user
You are a web scraping expert. Analyze this structural map of a listing page and identify its pagination mechanism.

PAGE URL: {page_url}

STRUCTURAL MAP:
{structural_map_json}

Look for:
- Numbered pagination: nav/div/ul with page number links (e.g., 1, 2, 3... or "next"/"prev" links with /page/N/ URLs)
- Load More button: a button or link with text like "Load More", "Show More", "المزيد" (Arabic for "more")
- Infinite scroll: no visible pagination controls but the page uses lazy-loading patterns
- No pagination: single page with all content visible

Respond with ONLY a JSON object (no explanation, no markdown):
{{"type": "none"|"numbered"|"load-more"|"infinite-scroll", "url_pattern": "full URL with {{page}} placeholder or null", "total_pages": number or null}}

For numbered pagination:
- url_pattern must be the full URL with {{page}} as the page number placeholder (e.g., "https://example.com/articles/page/{{page}}/")
- total_pages should be the highest page number you can find in the pagination links
- Derive the pattern from the actual href values in the structural map, not by guessing

For load-more or infinite-scroll: set url_pattern to null and total_pages to null.
For no pagination: set both to null.
<end_of_turn>
<start_of_turn>model
"""

    try:
        genai.configure(api_key=api_key)
        llm = genai.GenerativeModel(model)
        response = llm.generate_content(prompt)
        text = response.text.strip()

        # Clean up: remove markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:text.rfind("```")]
        text = text.strip()

        result = json.loads(text)

        # Validate the response
        pag_type = result.get("type", "none")
        if pag_type not in ("none", "numbered", "load-more", "infinite-scroll"):
            return None

        return {
            "type": pag_type,
            "url_pattern": result.get("url_pattern"),
            "total_pages": result.get("total_pages"),
        }

    except Exception as e:
        print(f"    ⚠️  LLM pagination detection failed: {e}")
        return None


# Reuse utilities from Agent_for_single_page_gemma (fetch, structural map, execute)
# rather than copying the logic here.
from Agent_for_single_page_gemma import (
    fetch_page_structure,
    execute_extraction_code,
    list_available_models,
)

# --- Configuration ---
LINKS_AGENT_SCRIPT = "Links_Agent_gemma.py"
AGENT_SCRIPT = "Agent_for_single_page_gemma.py"
RUN_DIR = f"orch_runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _struct_signature(smap, depth=0, max_depth=3):
    """Build a hashable string representing the skeleton of a structural map.
    
    Captures the tag+class tree up to max_depth so that articles sharing the
    same template end up with the same signature.
    """
    if depth >= max_depth or not isinstance(smap, list):
        return ""
    parts = []
    for node in smap:
        tag = node.get("tag", "")
        cls = node.get("attributes", {}).get("class", "")
        children_sig = _struct_signature(node.get("children", []), depth + 1, max_depth)
        parts.append(f"{tag}.{cls}({children_sig})")
    return "|".join(parts)


def cluster_by_structure(articles_with_maps):
    """Group articles by structural map similarity.
    
    Returns dict: signature_hash -> list of (url, title, html_content, structural_map)
    """
    clusters = defaultdict(list)
    for item in articles_with_maps:
        sig = _struct_signature(item["structural_map"])
        sig_hash = hashlib.md5(sig.encode()).hexdigest()[:12]
        clusters[sig_hash].append(item)
    return dict(clusters)


def _call_agent_subprocess(cmd, timeout=300):
    """Run a CLI agent subprocess and parse ORCH_RESULT from stdout.
    
    Returns (success: bool, result: dict).
    """
    # Force UTF-8 so emojis in print() don't crash on Windows cp1252
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

        # Find the ORCH_RESULT line in stdout
        for line in proc.stdout.splitlines():
            if line.startswith("ORCH_RESULT:"):
                payload = line[len("ORCH_RESULT:"):]
                result = json.loads(payload)
                return (result.get("status") == "ok"), result

        # No result line found
        stderr_snip = (proc.stderr or "")[:2000]
        stdout_snip = (proc.stdout or "")[-1000:]
        return False, {
            "status": "error",
            "error": f"No ORCH_RESULT in output. stderr: {stderr_snip}  stdout(tail): {stdout_snip}"
        }

    except subprocess.TimeoutExpired:
        return False, {"status": "error", "error": f"Subprocess timed out ({timeout}s)"}
    except Exception as e:
        return False, {"status": "error", "error": str(e)}


def call_links_agent_cli(url, api_key, model="gemma-3-27b-it"):
    """Call Links_Agent_gemma.py via subprocess in CLI mode.
    
    Returns (success: bool, result: dict) where result has:
      - status, code_file, output_file, data  (on success)
      - status, error                         (on failure)
    """
    cmd = [
        sys.executable, LINKS_AGENT_SCRIPT,
        "--url", url,
        "--api-key", api_key,
        "--model", model,
    ]
    print(f"  🔧 Calling: python {LINKS_AGENT_SCRIPT} --url {url[:80]}...")
    return _call_agent_subprocess(cmd, timeout=300)


def call_agent_cli(url, api_key, requirements, model="gemma-3-27b-it"):
    """Call Agent_for_single_page_gemma.py via subprocess in CLI mode.
    
    Returns (success: bool, result: dict) where result has:
      - status, code_file, output_file, data  (on success)
      - status, error                         (on failure)
    """
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
    """Fetch a page and click 'Load More' until no more content loads.

    Returns the fully-loaded HTML string.
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(2000)

        clicks = 0
        while True:
            try:
                # Common selectors for "Load More" buttons
                btn = await page.query_selector(
                    "button:has-text('Load More'), "
                    "button:has-text('load more'), "
                    "a:has-text('Load More'), "
                    "button:has-text('المزيد'), "      # Arabic "more"
                    "a:has-text('المزيد'), "
                    ".load-more, .loadmore, [data-load-more]"
                )
                if not btn or not await btn.is_visible():
                    break
                await btn.click()
                clicks += 1
                print(f"    Clicked 'Load More' ({clicks})...")
                await page.wait_for_timeout(2000)
            except Exception:
                break

        html = await page.content()
        await browser.close()
        print(f"    ✓ Fully loaded after {clicks} click(s)")
        return html


async def fetch_html_with_infinite_scroll(url):
    """Fetch a page and scroll to the bottom until no more content loads.

    Returns the fully-loaded HTML string.
    """
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


async def collect_paginated_links(base_url, total_pages, extraction_code, delay=1.5):
    """Fetch pages 2..N and extract links using existing working code.

    Returns list of {url, title} dicts (deduplicated).
    """
    seen = set()
    all_links = []

    # Build the exec namespace for the extraction code
    exec_namespace = {
        "BeautifulSoup": BeautifulSoup,
        "re": __import__("re"),
        "json": json,
        "urljoin": urljoin,
        "urlparse": __import__("urllib.parse").parse.urlparse,
    }
    exec(extraction_code, exec_namespace)
    extract_fn = exec_namespace["extract_data"]

    for page_num in range(2, total_pages + 1):
        url = base_url.format(page=page_num)
        print(f"    [{page_num}/{total_pages}] Fetching: {url[:80]}...")
        try:
            html, _ = await fetch_page_structure(url)
            if html:
                result = extract_fn(html)
                page_links = result.get("article_links", [])
                new = 0
                for link in page_links:
                    if link["url"] not in seen:
                        seen.add(link["url"])
                        all_links.append(link)
                        new += 1
                print(f"      → {len(page_links)} links, {new} new (total: {len(all_links)})")
        except Exception as e:
            print(f"      ✗ Error: {e}")

        if page_num < total_pages and delay > 0:
            time.sleep(delay)

    return all_links


async def main():
    print("=" * 60)
    print("🎯 ORCHESTRATOR — Batch Article Extraction")
    print("=" * 60)

    # ── Step 0: Setup ────────────────────────────────────────
    api_key = input("\n🔑 Enter your Gemini API key: ").strip()
    if not api_key:
        print("❌ API key is required!")
        return

    # List available models (same as the agents' interactive mode)
    print("\n⏳ Checking available models...")
    available_models = await list_available_models(api_key)

    gemma_models = [m for m in available_models if 'gemma' in m.lower()]
    other_models = [m for m in available_models if 'gemma' not in m.lower()]
    sorted_models = gemma_models + other_models

    if sorted_models:
        print(f"\n Found {len(sorted_models)} available models:")
        if gemma_models:
            print(f"  ── Gemma models ──")
        for i, m in enumerate(sorted_models, 1):
            marker = " ★" if 'gemma' in m.lower() else ""
            print(f"   {i}. {m}{marker}")

        print(f"\n Select a model:")
        print(f"   [Enter number] Choose from list above")
        print(f"   [Press Enter] Use default (gemma-3-27b-it)")

        choice = input("   → ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(sorted_models):
            model = sorted_models[int(choice) - 1]
            if model.startswith('models/'):
                model = model[7:]
            print(f"✓ Selected: {model}")
        else:
            model = 'gemma-3-27b-it'
            print(f"✓ Using default: {model}")
    else:
        model = input("\n🤖 Model name [Enter for gemma-3-27b-it]: ").strip() or "gemma-3-27b-it"

    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"\n📂 Run directory: {RUN_DIR}")

    # ── Step 1: Get article links ────────────────────────────
    print("\n How do you want to get article links?")
    print("   [1] Extract from a listing page (calls Links_Agent_gemma.py)")
    print("   [2] Load from an existing JSON file")
    link_choice = input("   → ").strip()

    if link_choice == "1":
        listing_url = input("\n🌐 Enter the listing page URL: ").strip()
        if not listing_url:
            print("❌ URL is required!")
            return

        # ── Extract links from page 1 using Links_Agent ──────
        print(f"\n⏳ Extracting article links from {listing_url}...")
        link_success, link_result = call_links_agent_cli(listing_url, api_key, model)

        if not link_success:
            error_msg = link_result.get("error", "Unknown error")
            print(f"\n❌ Links extraction failed:\n{error_msg}")
            return

        article_links = link_result.get("data", {}).get("article_links", [])
        print(f"✓ Extracted {len(article_links)} article links from page 1")

        # ── Detect pagination automatically ──────────────────
        print("\n⏳ Analyzing page for pagination...")
        html_content, structural_map = await fetch_page_structure(listing_url)
        structural_map_json = json.dumps(structural_map, indent=2) if structural_map else ""

        pag_info = None
        if structural_map_json:
            pag_info = await detect_pagination_with_llm(
                structural_map_json, listing_url, api_key, model
            )

        if pag_info:
            pag_type = pag_info["type"]
            print(f"   ✓ LLM detected pagination: {pag_type}")
            if pag_info.get("url_pattern"):
                print(f"     Pattern: {pag_info['url_pattern']}")
            if pag_info.get("total_pages"):
                print(f"     Pages: {pag_info['total_pages']}")
        else:
            pag_type = "none"
            print("   ℹ  No pagination detected (single page).")

        if pag_type == "numbered":
            url_pattern = pag_info.get("url_pattern")
            total_pages = pag_info.get("total_pages")

            # Validate LLM response — fall back to HTML parser if needed
            if not url_pattern or not total_pages or "{page}" not in str(url_pattern):
                print("   ⚠️  LLM response incomplete, trying HTML-based detection...")
                detected_pattern, detected_pages = await detect_pagination(listing_url)
                if detected_pattern and detected_pages:
                    url_pattern = detected_pattern
                    total_pages = detected_pages
                    print(f"   ✓ HTML detection: {total_pages} pages")
                    print(f"     Pattern: {url_pattern}")
                else:
                    print("   ⚠️  Could not auto-detect. Please enter manually:")
                    total_pages_str = input("   📄 How many pages total? → ").strip()
                    total_pages = int(total_pages_str) if total_pages_str.isdigit() and int(total_pages_str) >= 2 else 0
                    default_pattern = listing_url.rstrip("/") + "/page/{page}/"
                    print(f"   URL pattern (use {{page}} as placeholder):")
                    print(f"   [Enter] Use: {default_pattern}")
                    url_pattern = input("   → ").strip() or default_pattern

            # Warn for large page counts
            if total_pages and total_pages > 20:
                est_links = total_pages * len(article_links)
                print(f"\n⚠️  That's {total_pages} pages (~{est_links} links).")
                print(f"   Fetching will take ~{total_pages * 3.5 / 60:.0f} minutes.")
                print(f"   Do you want to scrape all {total_pages} pages?")
                print(f"   [Enter] Yes, scrape all")
                print(f"   [number] Only scrape first N pages")
                print(f"   [0] Skip pagination, use page 1 only")
                limit = input("   → ").strip()
                if limit == "0":
                    total_pages = 0
                    print("   ✓ Proceeding with page 1 only.")
                elif limit.isdigit() and int(limit) >= 2:
                    total_pages = int(limit)
                    print(f"   ✓ Will scrape pages 1–{total_pages}.")

            if total_pages and total_pages >= 2:
                code_file = link_result.get("code_file", "")
                if code_file and os.path.exists(code_file):
                    with open(code_file, "r", encoding="utf-8") as f:
                        extraction_code = f.read()

                    page1_urls = {l["url"] for l in article_links}

                    print(f"\n⏳ Collecting links from pages 2–{total_pages}...")
                    extra_links = await collect_paginated_links(
                        url_pattern, total_pages, extraction_code, delay=1.5
                    )

                    for link in extra_links:
                        if link["url"] not in page1_urls:
                            article_links.append(link)
                            page1_urls.add(link["url"])

                    print(f"✓ Total unique links after pagination: {len(article_links)}")
                else:
                    print(f"⚠️  Code file not found ({code_file}), proceeding with page 1 only.")

        elif pag_type == "load-more":
            print(f"\n⏳ Loading all content via 'Load More' from {listing_url}...")
            full_html = await fetch_html_with_load_more(listing_url)
            code_file = link_result.get("code_file", "")
            if code_file and os.path.exists(code_file):
                with open(code_file, "r", encoding="utf-8") as f:
                    extraction_code = f.read()
                exec_ns = {
                    "BeautifulSoup": BeautifulSoup, "re": __import__("re"),
                    "json": json, "urljoin": urljoin,
                    "urlparse": __import__("urllib.parse").parse.urlparse,
                }
                exec(extraction_code, exec_ns)
                full_result = exec_ns["extract_data"](full_html)
                extra_links = full_result.get("article_links", [])
                seen_urls = {l["url"] for l in article_links}
                for link in extra_links:
                    if link["url"] not in seen_urls:
                        article_links.append(link)
                        seen_urls.add(link["url"])
            print(f"✓ Total links after load-more: {len(article_links)}")

        elif pag_type == "infinite-scroll":
            print(f"\n⏳ Scrolling to load all content from {listing_url}...")
            full_html = await fetch_html_with_infinite_scroll(listing_url)
            code_file = link_result.get("code_file", "")
            if code_file and os.path.exists(code_file):
                with open(code_file, "r", encoding="utf-8") as f:
                    extraction_code = f.read()
                exec_ns = {
                    "BeautifulSoup": BeautifulSoup, "re": __import__("re"),
                    "json": json, "urljoin": urljoin,
                    "urlparse": __import__("urllib.parse").parse.urlparse,
                }
                exec(extraction_code, exec_ns)
                full_result = exec_ns["extract_data"](full_html)
                extra_links = full_result.get("article_links", [])
                seen_urls = {l["url"] for l in article_links}
                for link in extra_links:
                    if link["url"] not in seen_urls:
                        article_links.append(link)
                        seen_urls.add(link["url"])
            print(f"✓ Total links after infinite scroll: {len(article_links)}")

        # Save links to run dir
        with open(f"{RUN_DIR}/extracted_links.json", "w", encoding="utf-8") as f:
            json.dump({"article_links": article_links}, f, indent=2, ensure_ascii=False)

    else:
        input_file = input("\n📂 Enter JSON file path [Enter for extracted_data_pchrgaza_org.json]: ").strip()
        if not input_file:
            input_file = "extracted_data_pchrgaza_org.json"

        print(f"\n⏳ Loading links from {input_file}...")
        with open(input_file, "r", encoding="utf-8") as f:
            links_data = json.load(f)
        article_links = links_data.get("article_links", [])
        print(f"✓ Found {len(article_links)} article links")

        # Save links to run dir
        with open(f"{RUN_DIR}/input_links.json", "w", encoding="utf-8") as f:
            json.dump(article_links, f, indent=2, ensure_ascii=False)

    if not article_links:
        print("❌ No article links found!")
        return

    requirements = input("\n📝 What data to extract from each article?\n"
                         "   (e.g., 'title, date, author, article body text')\n   → ").strip()
    if not requirements:
        requirements = "title, date, author, article body text"
        print(f"   Using default: {requirements}")

    # ── Step 2: Fetch structural maps for all articles ───────
    print(f"\n⏳ Fetching structural maps for {len(article_links)} articles...")
    articles_with_maps = []
    fetch_failures = []

    for i, link in enumerate(article_links):
        url = link["url"]
        title = link.get("title", "")
        print(f"  [{i+1}/{len(article_links)}] Fetching: {title[:50]}...")

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
            fetch_failures.append({"url": url, "title": title, "reason": str(e)})
            print(f"    ❌ {str(e)[:100]}")

    print(f"\n✓ Fetched {len(articles_with_maps)} / {len(article_links)} articles")

    if not articles_with_maps:
        print("❌ Could not fetch any articles!")
        return

    # ── Step 3: Cluster articles by structural similarity ────
    print("\n⏳ Clustering articles by structure...")
    clusters = cluster_by_structure(articles_with_maps)
    print(f"✓ Found {len(clusters)} cluster(s):")
    for sig, members in clusters.items():
        print(f"  Cluster {sig}: {len(members)} article(s)")

    # Save cluster info
    cluster_info = {
        sig: [{"url": m["url"], "title": m["title"]} for m in members]
        for sig, members in clusters.items()
    }
    with open(f"{RUN_DIR}/clusters.json", "w", encoding="utf-8") as f:
        json.dump(cluster_info, f, indent=2, ensure_ascii=False)

    # ── Step 4 & 5: Process each cluster ─────────────────────
    all_extracted = []
    all_failures = list(fetch_failures)  # start with fetch failures

    for cluster_idx, (sig, members) in enumerate(clusters.items()):
        print(f"\n{'─' * 60}")
        print(f"📦 Cluster {sig} ({len(members)} articles)")
        print(f"{'─' * 60}")

        representative = members[0]
        rest = members[1:]

        # Step 4: Call Agent_for_single_page_gemma.py on the representative
        print(f"\n  🎯 Representative: {representative['title'][:60]}")
        success, agent_result = call_agent_cli(
            representative["url"], api_key, requirements, model
        )

        if not success:
            # Agent failed on representative → all members of this cluster fail
            error_msg = agent_result.get("error", "Unknown agent error")
            print(f"  ❌ Agent failed on representative: {error_msg[:200]}")
            for m in members:
                all_failures.append({
                    "url": m["url"],
                    "title": m["title"],
                    "reason": f"Agent failed on cluster representative: {error_msg[:300]}"
                })
            continue

        # Agent succeeded — save representative result
        rep_data = agent_result.get("data", {})
        code_file = agent_result.get("code_file", "")
        print(f"  ✓ Agent succeeded! Code: {code_file}")
        all_extracted.append({
            "url": representative["url"],
            "title": representative["title"],
            "data": rep_data,
        })

        if not rest:
            print(f"  (No other articles in this cluster)")
            continue

        # Step 5: Load the generated code and apply it to the rest of the cluster
        if not code_file or not os.path.exists(code_file):
            print(f"  ⚠️  Code file not found: {code_file} — skipping rest of cluster")
            for m in rest:
                all_failures.append({
                    "url": m["url"],
                    "title": m["title"],
                    "reason": "Code file from representative not found"
                })
            continue

        with open(code_file, "r", encoding="utf-8") as f:
            extraction_code = f.read()

        print(f"\n  ⏳ Applying code to {len(rest)} remaining articles...")
        for j, member in enumerate(rest):
            print(f"    [{j+1}/{len(rest)}] {member['title'][:50]}...", end=" ")
            ok, result = execute_extraction_code(extraction_code, member["html_content"])
            if ok and isinstance(result, dict):
                all_extracted.append({
                    "url": member["url"],
                    "title": member["title"],
                    "data": result,
                })
                print("✓")
            else:
                all_failures.append({
                    "url": member["url"],
                    "title": member["title"],
                    "reason": str(result)[:300],
                })
                print("❌")

    # ── Step 6: Save final results ───────────────────────────
    print(f"\n{'=' * 60}")
    print(f"📊 FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  ✓ Successfully extracted: {len(all_extracted)} articles")
    print(f"  ❌ Failed: {len(all_failures)} articles")

    success_file = f"{RUN_DIR}/extracted_data_all.json"
    with open(success_file, "w", encoding="utf-8") as f:
        json.dump(all_extracted, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Extracted data saved to: {success_file}")

    if all_failures:
        failure_file = f"{RUN_DIR}/failed_links.json"
        with open(failure_file, "w", encoding="utf-8") as f:
            json.dump(all_failures, f, indent=2, ensure_ascii=False)
        print(f"💾 Failed links saved to: {failure_file}")

    print(f"\n🏁 Orchestrator finished! Results in: {RUN_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())

