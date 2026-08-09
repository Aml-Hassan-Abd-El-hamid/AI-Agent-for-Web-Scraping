"""Auto-pagination orchestrator — the LLM detects the pagination type itself.

Same end-to-end pipeline as ``orch_interactive_pagination.py`` (extract links →
cluster articles by structure → generate extraction code once per cluster →
reuse it across pages → save incrementally), but the user no longer chooses how
the listing is paginated.

Instead, this script fetches the listing page once and asks the LLM to classify
the pagination itself:

  * numbered   — finds the page-2 URL, derives the ``{page}`` pattern, then
                 auto-advances until a page yields no new links (safety-capped).
  * load_more  — clicks the detected "load more / المزيد" button until it's gone.
  * infinite   — (ambiguous / none) auto-scrolls; if content grows it's infinite
                 scroll, otherwise it was effectively a single page.

The only user input is the API key, model, listing URL, and what data to extract.
All pagination handling is automatic.

LLM call budget:
  1    — pagination detection (this script)
  1    — link extraction code (Links Agent, page 1)
  N    — one per unique article-structure cluster (page 1 establishes most)
  +    — extra calls only if a later page surfaces a new structure
"""

import asyncio
import json
import os
import time
import traceback
from datetime import datetime
import re as _re
from urllib.parse import urljoin

import google.generativeai as genai
from bs4 import BeautifulSoup

# Reuse the entire proven pipeline (fetching, clustering, extraction, saving,
# stats reporting) from the numbered-pagination orchestrator. Importing the
# module (not just names) keeps its LLM-call / token / fetch counters consistent
# because every helper mutates that module's globals.
import orch_interactive_pagination as O
from utils import list_available_models, token_usage_from_response


# Safety cap: how many numbered pages to auto-advance through before stopping,
# in case a site never returns an empty page. The loop also stops naturally as
# soon as a page yields no new (unseen) links.
MAX_AUTO_PAGES = 500

# Max scroll rounds / load-more clicks used for dynamic listings.
DYNAMIC_MAX_ROUNDS = 30


# ═══════════════════════════════════════════════════════════════
# Pagination detection (LLM-driven, replaces the manual Phase 1)
# ═══════════════════════════════════════════════════════════════

def _gather_pagination_candidates(listing_url, html, max_links=40, max_buttons=20):
    """Pull pagination-looking anchors and load-more-looking buttons from *html*.

    Returns (link_candidates, button_candidates):
      link_candidates  — [{"text", "href"}] absolute URLs, ranked pagination-first
      button_candidates — [str] short visible texts of button-like elements
    """
    soup = BeautifulSoup(html, "lxml")

    # ── Anchors that could be pagination controls ──
    scored = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        abs_url = urljoin(listing_url, href)
        text = a.get_text(strip=True)[:40]
        key = (abs_url, text)
        if key in seen:
            continue
        seen.add(key)

        low = abs_url.lower()
        score = 0
        if _re.search(r"[?&](page|pg|p|offset|start|from)=\d+", low):
            score += 3
        if _re.search(r"/page/\d+", low):
            score += 3
        if "page" in low:
            score += 1
        if _re.search(r"/\d+/?$", low):
            score += 1
        tl = text.lower()
        if text.isdigit():
            score += 2
        if tl in ("next", "next page", "»", "›", ">", "older", "التالي",
                  "التالى", "المزيد", "التالي ›", "الصفحة التالية"):
            score += 3
        if score > 0:
            scored.append((score, {"text": text, "href": abs_url}))

    scored.sort(key=lambda x: -x[0])
    link_candidates = [c for _, c in scored[:max_links]]

    # ── Button-like elements that could be "load more" ──
    buttons = []
    bseen = set()
    for el in soup.find_all(["button", "a", "span", "div"]):
        role = (el.get("role") or "").lower()
        if el.name not in ("button", "a") and role != "button":
            continue
        text = el.get_text(strip=True)
        if not text or len(text) > 30:
            continue
        low = text.lower()
        is_load_more = any(h in low for h in O._LOAD_MORE_HINTS)
        if el.name == "button" or is_load_more:
            if text not in bseen:
                bseen.add(text)
                buttons.append(text)
        if len(buttons) >= max_buttons:
            break

    return link_candidates, buttons


def detect_pagination(listing_url, html, api_key, model="gemma-3-27b-it"):
    """Ask the LLM to classify how the listing paginates.

    Returns a dict:
      {"pagination_type": "numbered"|"load_more"|"none",
       "page2_url": str|None, "load_more_text": str|None}
    Falls back to {"pagination_type": "none", ...} on any failure.
    """
    fallback = {"pagination_type": "none", "page2_url": None, "load_more_text": None}

    link_candidates, button_candidates = _gather_pagination_candidates(listing_url, html)

    links_block = "\n".join(
        f"- {c['text'] or '(no text)'} -> {c['href']}" for c in link_candidates
    ) or "(no pagination-like links found)"
    buttons_block = "\n".join(f"- {t}" for t in button_candidates) or "(none found)"

    prompt = f"""<start_of_turn>user
You analyze a web listing/index page to determine HOW it paginates — i.e. how a
user reaches more items beyond those shown on first load.

Listing page URL:
{listing_url}

Candidate links found on the page (text -> absolute URL):
{links_block}

Candidate "load more / show more" button texts found on the page:
{buttons_block}

Decide the pagination type and respond with ONLY a JSON object:
{{
  "pagination_type": "numbered" | "load_more" | "none",
  "page2_url": "<absolute URL that leads to page 2 of THIS listing, or null>",
  "load_more_text": "<exact visible text of the load-more button, or null>"
}}

Rules:
- "numbered": the page links to numbered pages (1, 2, 3, Next) or uses URLs like
  ?page=2, /page/2/, or a trailing /2. Set page2_url to the absolute URL that
  leads to the SECOND page of THIS SAME listing. Never use a link to an
  individual article, a category, or an unrelated section.
- "load_more": there is a button/link that loads more items in place (text like
  "Load more", "Show more", "المزيد", "تحميل المزيد"). Set load_more_text to its
  exact text.
- "none": no numbered pagination and no load-more button (single page or
  infinite scroll).
- page2_url and load_more_text MUST be null unless clearly applicable.
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
        O._bump_llm_calls(agent="Pagination detection")
        O._bump_tokens("Pagination detection", token_usage_from_response(response))

        raw = (response.text or "").strip()
        if not raw:
            print("    ⚠️  Pagination detector returned an empty response.")
            return fallback

        print(f"    ℹ  Detector response: {raw[:400]}")
        parsed = json.loads(raw)

        ptype = (parsed.get("pagination_type") or "none").strip().lower()
        if ptype not in ("numbered", "load_more", "none"):
            ptype = "none"
        page2 = parsed.get("page2_url")
        if isinstance(page2, str):
            page2 = page2.strip() or None
        else:
            page2 = None
        lmt = parsed.get("load_more_text")
        if isinstance(lmt, str):
            lmt = lmt.strip() or None
        else:
            lmt = None

        # Guard: numbered needs a usable page-2 URL that isn't the listing itself.
        if ptype == "numbered" and (not page2 or page2.rstrip("/") == listing_url.rstrip("/")):
            print("    ⚠️  'numbered' but no valid page-2 URL — treating as 'none'.")
            ptype, page2 = "none", None

        return {"pagination_type": ptype, "page2_url": page2, "load_more_text": lmt}

    except (json.JSONDecodeError, ValueError) as e:
        print(f"    ⚠️  Could not parse detector JSON: {e}")
        return fallback
    except Exception as e:
        print(f"    ⚠️  Pagination detection failed: {e}\n{traceback.format_exc()}")
        return fallback


# ═══════════════════════════════════════════════════════════════
# Model picker (compact, mirrors the numbered orchestrator)
# ═══════════════════════════════════════════════════════════════

async def _pick_model(api_key):
    print("\n⏳ Checking available models...")
    available_models = await list_available_models(api_key)
    gemma = [m for m in available_models if "gemma" in m.lower()]
    other = [m for m in available_models if "gemma" not in m.lower()]
    sorted_models = gemma + other

    if not sorted_models:
        return input("\n🤖 Model name [Enter for gemma-3-27b-it]: ").strip() or "gemma-3-27b-it"

    print(f"\n Found {len(sorted_models)} available models:")
    if gemma:
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
        return model
    print("✓ Using default: gemma-3-27b-it")
    return "gemma-3-27b-it"


# ═══════════════════════════════════════════════════════════════
# Numbered page URL generator (auto page count)
# ═══════════════════════════════════════════════════════════════

def _numbered_page_iter(pattern, start_num, page2_url):
    """Yield successive listing-page URLs for numbered pagination.

    If *pattern* is available, yield pattern.format(page=n) for n = start_num,
    start_num+1, ... up to MAX_AUTO_PAGES. If no pattern could be derived, yield
    only the known page-2 URL (best effort) and stop.
    """
    if pattern and "{page}" in pattern:
        n = start_num if start_num is not None else 2
        for _ in range(MAX_AUTO_PAGES - 1):
            yield n, pattern.format(page=n)
            n += 1
    elif page2_url:
        yield 2, page2_url


# ═══════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("🎯 AUTO-PAGINATION ORCHESTRATOR (LLM detects pagination)")
    print("=" * 60)
    O._reset_llm_calls()
    O._reset_fetch_via()
    O._reset_tokens()
    O._reset_map_fits()
    run_start = time.time()
    input_urls = []

    # ── Step 0: setup ────────────────────────────────────────
    api_key = input("\n🔑 Enter your Gemini API key: ").strip()
    if not api_key:
        print("❌ API key is required!")
        return

    model = await _pick_model(api_key)

    run_dir = f"orch_runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n📂 Run directory: {run_dir}")

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
    # Phase 1: Detect pagination automatically (LLM)
    # ══════════════════════════════════════════════════════════
    print(f"\n⏳ Fetching listing page to detect pagination: {listing_url}...")
    try:
        listing_html, _ = await O.fetch_page_structure(listing_url)
    except Exception as e:
        listing_html = None
        print(f"  ⚠️  Could not fetch listing page for detection: {e}")

    pattern = None
    start_num = 2
    page2_url = None
    scroll_mode = None          # None | "scroll" | "load_more"
    scroll_selector = None
    pagination_type = "single page (auto)"

    if listing_html:
        print("\n🔎 Asking the LLM to detect the pagination type...")
        detected = detect_pagination(listing_url, listing_html, api_key, model)
        ptype = detected["pagination_type"]
        page2_url = detected["page2_url"]
        load_more_text = detected["load_more_text"]
        print(f"  🤖 Detected pagination type: {ptype}"
              + (f"  (page 2: {page2_url})" if page2_url else "")
              + (f"  (button: '{load_more_text}')" if load_more_text else ""))

        if ptype == "numbered" and page2_url:
            input_urls.append(page2_url)
            pattern, p1, p2 = O.derive_pagination_pattern(listing_url, page2_url)
            if not pattern:
                print("  ⚠️  String diff failed — asking the LLM for the pattern...")
                pattern, p1, p2 = O.derive_pagination_pattern_llm(
                    listing_url, page2_url, api_key, model)
            if pattern:
                start_num = p2 if p2 is not None else 2
                pagination_type = "numbered pagination (auto)"
                print(f"  ✓ URL pattern: {pattern} (starting at page {start_num})")
            else:
                # Fall back to just the one known next page.
                pagination_type = "numbered pagination (auto, no pattern)"
                print("  ⚠️  Could not derive a pattern; will fetch only the known page 2.")

        elif ptype == "load_more":
            scroll_mode = "load_more"
            pagination_type = "load more button (auto)"
            # Teach the auto-detector the exact button text if it's novel.
            if load_more_text and load_more_text.lower() not in O._LOAD_MORE_HINTS:
                O._LOAD_MORE_HINTS.append(load_more_text.lower())

        else:
            # "none" — could be a single page or infinite scroll. We probe by
            # scrolling; if content grows it was infinite scroll, otherwise it
            # behaves as a single page (harmless).
            scroll_mode = "scroll"
            pagination_type = "single page / infinite scroll (auto)"
    else:
        print("  ⚠️  No listing HTML — proceeding as a single page.")

    # ══════════════════════════════════════════════════════════
    # Phase 2: Page 1 — extract links (Links Agent, LLM call)
    # ══════════════════════════════════════════════════════════
    print(f"\n⏳ Extracting article links from page 1: {listing_url}...")
    link_success, link_result = O.call_links_agent_cli(listing_url, api_key, model)

    if not link_success:
        error_msg = link_result.get("error", "Unknown error")
        print(f"\n❌ Links extraction failed:\n{error_msg}")
        O.write_results_md({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": run_dir, "model": model,
            "pagination_type": pagination_type, "input_urls": input_urls,
            "requirements": requirements,
            "pages_requested": 1, "pages_processed": 0,
            "articles_extracted": 0, "articles_failed": 0, "clusters": 0,
            "llm_calls": O._LLM_CALLS,
            "llm_calls_by_agent": dict(O._LLM_CALLS_BY_AGENT),
            "tokens_by_agent": dict(O._TOKENS_BY_AGENT),
            "elapsed_seconds": time.time() - run_start,
            "errors": [f"Links extraction failed: {error_msg}"],
        })
        return

    page1_links = link_result.get("data", {}).get("article_links", [])
    page1_links, _dropped = O._filter_article_links(page1_links, listing_url, pattern)
    if _dropped:
        print(f"  🧹 Dropped {_dropped} non-article (pagination/category) link(s)")
    link_code_file = link_result.get("code_file", "")
    print(f"✓ Extracted {len(page1_links)} article links from page 1")

    if not page1_links:
        print("❌ No article links found on page 1!")
        O.write_results_md({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": run_dir, "model": model,
            "pagination_type": pagination_type, "input_urls": input_urls,
            "requirements": requirements,
            "pages_requested": 1, "pages_processed": 0,
            "articles_extracted": 0, "articles_failed": 0, "clusters": 0,
            "llm_calls": O._LLM_CALLS,
            "llm_calls_by_agent": dict(O._LLM_CALLS_BY_AGENT),
            "tokens_by_agent": dict(O._TOKENS_BY_AGENT),
            "elapsed_seconds": time.time() - run_start,
            "errors": ["No article links found on page 1"],
        })
        return

    # Load the link-extraction code so we can reuse it on later pages.
    link_extraction_code = ""
    if link_code_file and os.path.exists(link_code_file):
        with open(link_code_file, "r", encoding="utf-8") as f:
            link_extraction_code = f.read()

    # ── Dynamic listings: expand page 1 via scroll / load-more ──
    if scroll_mode:
        if not link_extraction_code:
            print("\n⚠️  No link-extraction code — cannot expand dynamic content. "
                  "Using initial links only.")
        else:
            full_html = await O.collect_listing_html(
                listing_url, mode=scroll_mode,
                max_rounds=DYNAMIC_MAX_ROUNDS, load_more_selector=scroll_selector,
            )
            if full_html:
                expanded = O.run_link_extraction_code(link_extraction_code, full_html)
                expanded, _ = O._filter_article_links(expanded, listing_url, pattern)
                merged = {l["url"]: l for l in page1_links}
                for l in expanded:
                    merged.setdefault(l["url"], l)
                grew = len(merged) > len(page1_links)
                print(f"  ✓ {len(expanded)} links after dynamic load "
                      f"(was {len(page1_links)} initially → {len(merged)} unique)")
                page1_links = list(merged.values())
                # Resolve the ambiguous "none" case for reporting.
                if scroll_mode == "scroll":
                    pagination_type = ("infinite scroll (auto)" if grew
                                       else "single page (auto)")

    # ══════════════════════════════════════════════════════════
    # Phase 2 cont.: process page 1 articles
    # ══════════════════════════════════════════════════════════
    all_extracted = []
    all_failures = []
    cluster_registry = {}
    seen_urls = set()

    print(f"\n⏳ Fetching structural maps for page 1 ({len(page1_links)} articles)...")
    arts, fails = await O.fetch_articles(page1_links, label=" [page 1]")
    all_failures.extend(fails)
    for a in page1_links:
        seen_urls.add(a["url"])

    if arts:
        print("\n⏳ Clustering page 1 articles and extracting data...")
        ext, fl, cluster_registry = O.process_page_articles(
            arts, cluster_registry, api_key, model, requirements,
        )
        all_extracted.extend(ext)
        all_failures.extend(fl)

    pages_processed = 1
    progress = {
        "current_page": 1,
        "extracted_count": len(all_extracted),
        "failed_count": len(all_failures),
        "cluster_registry": {
            sig: {"code_file": v["code_file"]} for sig, v in cluster_registry.items()
        },
    }
    O.save_incremental(run_dir, all_extracted, all_failures, progress)
    O._atomic_json_write(os.path.join(run_dir, "extracted_links.json"),
                         {"article_links": page1_links})
    print(f"\n💾 Page 1 saved ({len(all_extracted)} extracted, {len(all_failures)} failed)")

    # ══════════════════════════════════════════════════════════
    # Phase 3: Numbered pagination — auto-advance until no new links
    # ══════════════════════════════════════════════════════════
    if pattern or (pagination_type.startswith("numbered") and page2_url):
        if not link_extraction_code:
            print("\n⚠️  No link-extraction code — cannot process further pages.")
        else:
            for page_num, page_url in _numbered_page_iter(pattern, start_num, page2_url):
                print(f"\n{'─' * 60}")
                print(f"📄 Page {page_num} (auto): {page_url[:80]}")
                print(f"{'─' * 60}")

                try:
                    page_html, _ = await O.fetch_page_structure(page_url)
                except Exception as e:
                    print(f"  ❌ Failed to fetch page: {e}")
                    print("  ⏹  Stopping pagination (fetch failure).")
                    break

                if not page_html:
                    print("  ⏹  Empty response — stopping pagination.")
                    break

                new_links_raw = O.run_link_extraction_code(link_extraction_code, page_html)
                new_links_raw, _ = O._filter_article_links(new_links_raw, listing_url, pattern)
                print(f"  Found {len(new_links_raw)} links on this page")

                new_links = []
                for lnk in new_links_raw:
                    if lnk["url"] not in seen_urls:
                        seen_urls.add(lnk["url"])
                        new_links.append(lnk)
                print(f"  {len(new_links)} new (after dedup)")

                # An empty page (no new links) marks the end of the listing.
                if not new_links:
                    print("  ⏹  No new links — reached the end of pagination.")
                    break

                arts, fails = await O.fetch_articles(new_links, label=f" [page {page_num}]")
                all_failures.extend(fails)
                if arts:
                    ext, fl, cluster_registry = O.process_page_articles(
                        arts, cluster_registry, api_key, model, requirements,
                    )
                    all_extracted.extend(ext)
                    all_failures.extend(fl)

                pages_processed = page_num
                progress["current_page"] = page_num
                progress["extracted_count"] = len(all_extracted)
                progress["failed_count"] = len(all_failures)
                progress["cluster_registry"] = {
                    sig: {"code_file": v["code_file"]}
                    for sig, v in cluster_registry.items()
                }
                O.save_incremental(run_dir, all_extracted, all_failures, progress)
                print(f"  💾 Saved (total: {len(all_extracted)} extracted, "
                      f"{len(all_failures)} failed)")

                time.sleep(1.5)  # polite delay between pages

    # ══════════════════════════════════════════════════════════
    # Phase 4: Final save + stats
    # ══════════════════════════════════════════════════════════
    cluster_info = {sig: {"code_file": v.get("code_file", "")}
                    for sig, v in cluster_registry.items()}
    O._atomic_json_write(os.path.join(run_dir, "clusters.json"), cluster_info)
    O.save_incremental(run_dir, all_extracted, all_failures, {
        "status": "finished",
        "extracted_count": len(all_extracted),
        "failed_count": len(all_failures),
    })

    O._print_summary(run_dir, all_extracted, all_failures)

    O.write_results_md({
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
        "llm_calls": O._LLM_CALLS,
        "llm_calls_by_agent": dict(O._LLM_CALLS_BY_AGENT),
        "elapsed_seconds": time.time() - run_start,
        "fetch_via": dict(O._FETCH_VIA),
        "tokens_by_agent": dict(O._TOKENS_BY_AGENT),
        "extracted_data": all_extracted,
        "errors": [f.get("reason", "") for f in all_failures],
    })


if __name__ == "__main__":
    asyncio.run(main())
