"""Snapshot collector — freeze a website's HTML into a reproducible corpus.

This is a sibling of `orch_pag_numbered_only.py`: it walks a listing site with
the *same* pagination / link-extraction / fetch logic, but instead of running
the article extraction pipeline it **saves the raw HTML to disk** so the site can
be re-scraped and re-scored offline later. This is the frozen "gold corpus" the
paper needs for reproducible deployment numbers (sites change and Cloudflare
varies, so live re-runs are not reproducible — a saved snapshot is).

It reuses the orchestrator's helpers directly (pagination derivation, Links
Agent call, dynamic-content expansion, and the plain-requests→browser fetch),
so a snapshot fetches pages exactly the way a real run would.

Output layout (one folder per snapshot, under `snapshots/`)::

    snapshots/<domain>_<YYYYMMDD_HHMMSS>/
    ├── listing_page_001.html          # page-1 listing HTML
    ├── listing_page_001_expanded.html # (scroll/load-more only) fully expanded
    ├── listing_page_002.html          # page-2 listing HTML (numbered pagination)
    ├── ...
    ├── article_0001.html              # one file per unique article
    ├── article_0002.html
    ├── ...
    ├── link_extraction_code.py        # code the Links Agent generated (replayable)
    └── metadata.json                  # maps every file ↔ url/title/page + hashes

`metadata.json` is what `app.py` uses to label, and what an offline replay uses
to reconstruct the run. Each article records its URL, title, source page, byte
size, and SHA-1 (so the corpus is verifiably frozen).

LLM cost: exactly the standard budget for link discovery — **1** Links Agent
call (plus 0–1 pagination-pattern fallback). No article-extraction LLM calls are
made here; that happens later, offline, against the saved HTML.

Usage::

    python snapshot_collector.py

Then answer the prompts (listing URL, pagination type, page count) exactly as
you would for the orchestrator.
"""

import asyncio
import hashlib
import os
import re as _re
import time
import traceback
from datetime import datetime
from urllib.parse import urlparse

# Reuse the orchestrator's machinery as a namespace so its live counters
# (_LLM_CALLS, _FETCH_VIA, ...) can be read after the run.
import orch_pag_numbered_only as orch
from Agent_for_single_page_gemma import fetch_page_structure
from utils import list_available_models


# ═══════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════

def _domain_slug(url):
    """Turn a URL into a filesystem-safe domain slug (e.g. 'the_guardian_com')."""
    net = urlparse(url).netloc or "site"
    if net.startswith("www."):
        net = net[4:]
    return _re.sub(r"[^a-zA-Z0-9]+", "_", net).strip("_").lower() or "site"


def _save_html(path, html):
    """Write raw HTML to *path* as UTF-8. Returns (bytes, sha1)."""
    data = html or ""
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write(data)
    sha1 = hashlib.sha1(data.encode("utf-8", errors="replace")).hexdigest()
    return len(data), sha1


def _save_articles(arts, snapshot_dir, page_num, next_id, articles_meta):
    """Save each fetched article's HTML and append its metadata record.

    Returns the number of articles saved (so the caller can advance next_id).
    """
    saved = 0
    for art in arts:
        aid = next_id + saved
        fname = f"article_{aid:04d}.html"
        nbytes, sha1 = _save_html(
            os.path.join(snapshot_dir, fname), art.get("html_content", ""))
        articles_meta.append({
            "id": aid,
            "file": fname,
            "url": art.get("url", ""),
            "title": art.get("title", ""),
            "page_num": page_num,
            "bytes": nbytes,
            "sha1": sha1,
        })
        print(f"    💾 {fname}  ({nbytes:,} bytes)  {art.get('title', '')[:50]}")
        saved += 1
    return saved


def write_snapshot_results_md(stats, path="results.md"):
    """Append a per-snapshot section to results.md."""
    header_needed = not os.path.exists(path)
    elapsed = stats.get("elapsed_seconds", 0.0)
    mins, secs = divmod(int(elapsed), 60)

    lines = []
    if header_needed:
        lines.append("# Scraping Run Results\n")
        lines.append("Auto-generated stats, one section per run.\n")

    lines.append(f"\n## Snapshot {stats.get('timestamp', '')}\n")
    lines.append(f"- **Website:** {stats.get('website', 'N/A')}")
    lines.append(f"- **Snapshot directory:** `{stats.get('snapshot_dir', 'N/A')}`")
    lines.append(f"- **Model (Links Agent):** {stats.get('model', 'N/A')}")
    lines.append(f"- **Pagination type:** {stats.get('pagination_type', 'N/A')}")

    input_urls = stats.get("input_urls", [])
    if input_urls:
        lines.append("- **Input URLs:**")
        for u in input_urls:
            lines.append(f"  - {u}")

    if stats.get("pagination_pattern"):
        lines.append(f"- **Pagination pattern:** `{stats['pagination_pattern']}`")

    lines.append(f"- **Requirements (for later labeling):** {stats.get('requirements', 'N/A')}")
    lines.append(f"- **Listing pages saved:** {stats.get('listing_pages_saved', 0)}")
    lines.append(f"- **Articles saved:** {stats.get('articles_saved', 0)}")
    lines.append(f"- **Article fetch failures:** {stats.get('article_failures', 0)}")
    lines.append(f"- **LLM calls (link discovery only):** {stats.get('llm_calls', 0)}")

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

    lines.append(f"- **Total time:** {mins}m {secs}s")

    errors = stats.get("errors", [])
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

    print(f"\n📝 Snapshot stats appended to {path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("📸 SNAPSHOT COLLECTOR")
    print("=" * 60)

    # Reset the orchestrator's live counters (they back our stats).
    orch._reset_llm_calls()
    orch._reset_fetch_via()
    orch._reset_tokens()
    orch._reset_map_fits()

    run_start = time.time()
    input_urls = []
    pagination_type = "single page"
    errors = []

    # ── Setup: API key + model ───────────────────────────────
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

    # ── Listing URL + requirements (stored for later labeling) ──
    listing_url = input("\n🌐 Enter the listing page URL: ").strip()
    if not listing_url:
        print("❌ URL is required!")
        return
    input_urls.append(listing_url)

    website = input(
        "\n🏷️  Website name for metadata [Enter to use the domain]: "
    ).strip() or _domain_slug(listing_url)

    requirements = input(
        "\n📝 What data will be labeled per article later?\n"
        "   (e.g., 'title, date, author, article body text')\n   → "
    ).strip() or "title, date, author, article body text"

    # ── Snapshot directory ───────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = os.path.join("snapshots", f"{_domain_slug(listing_url)}_{ts}")
    os.makedirs(snapshot_dir, exist_ok=True)
    print(f"\n📂 Snapshot directory: {snapshot_dir}")

    # ══════════════════════════════════════════════════════════
    # Phase 1: Pagination setup (identical to the orchestrator)
    # ══════════════════════════════════════════════════════════
    page_urls = []
    pattern = None
    scroll_mode = None
    scroll_selector = None
    scroll_rounds = 20

    print("\n📄 How is this listing paginated?")
    print("   [1] Numbered pagination — I'll provide page 1 and page 2 URLs")
    print("   [2] No pagination — single page only")
    print("   [3] Infinite scroll — content loads as you scroll down")
    print("   [4] Load more button — content loads when a button is clicked")
    pag_choice = input("   → ").strip()

    if pag_choice == "3":
        pagination_type = "infinite scroll"
        scroll_mode = "scroll"
        rounds_str = input("\n   How many scroll rounds at most? [Enter for 20] → ").strip()
        if rounds_str.isdigit() and int(rounds_str) > 0:
            scroll_rounds = int(rounds_str)
        print(f"   ✓ Infinite scroll — up to {scroll_rounds} rounds.")
    elif pag_choice == "4":
        pagination_type = "load more button"
        scroll_mode = "load_more"
        scroll_selector = input(
            "\n   CSS selector for the 'load more' button [Enter to auto-detect by text] → "
        ).strip() or None
        rounds_str = input("   How many clicks at most? [Enter for 20] → ").strip()
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
            pattern, p1_num, p2_num = orch.derive_pagination_pattern(url1, url2)
            if pattern:
                print(f"\n   ✓ Derived URL pattern (string diff): {pattern}")
            else:
                print(f"\n   ⚠️  String diff couldn't derive pattern. Asking LLM...")
                pattern, p1_num, p2_num = orch.derive_pagination_pattern_llm(
                    url1, url2, api_key, model)
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

                total_str = input("\n   📄 How many pages to snapshot (including page 1)? → ").strip()
                if total_str.isdigit() and int(total_str) >= 2:
                    total_pages = int(total_str)
                    start_num = p2_num if p2_num is not None else 2
                    page_urls = [
                        pattern.format(page=n)
                        for n in range(start_num, start_num + total_pages - 1)
                    ]
                    print(f"   ✓ Will snapshot {total_pages} pages ({len(page_urls)} after page 1)")
                    print(f"     First: {page_urls[0]}")
                    if len(page_urls) > 1:
                        print(f"     Last:  {page_urls[-1]}")
                else:
                    print("   ⚠️  Need at least 2 pages. Proceeding with page 1 only.")
            else:
                print(f"\n   ⚠️  Could not derive pattern automatically.")
                print(f"   Enter the URL pattern manually (use {{page}} as placeholder):")
                manual_pat = input("   → ").strip()
                if manual_pat and "{page}" in manual_pat:
                    pattern = manual_pat
                    total_str = input("   📄 How many pages to snapshot (including page 1)? → ").strip()
                    if total_str.isdigit() and int(total_str) >= 2:
                        total_pages = int(total_str)
                        page_urls = [pattern.format(page=n) for n in range(2, total_pages + 1)]
                        print(f"   ✓ Will snapshot {total_pages} pages ({len(page_urls)} after page 1)")
                    else:
                        print("   ⚠️  Need at least 2 pages. Proceeding with page 1 only.")
                else:
                    print("   ⚠️  Invalid or missing pattern. Proceeding with page 1 only.")
    else:
        print("   ✓ Single page — no pagination.")

    # Metadata accumulators (saved incrementally).
    listing_pages_meta = []
    articles_meta = []
    article_failures = []
    next_article_id = 1

    def _write_metadata():
        orch._atomic_json_write(os.path.join(snapshot_dir, "metadata.json"), {
            "website": website,
            "snapshot_date": datetime.now().isoformat(timespec="seconds"),
            "pagination_type": pagination_type,
            "listing_url": listing_url,
            "input_urls": input_urls,
            "pagination_pattern": pattern,
            "requirements": requirements,
            "link_extraction_code_file": "link_extraction_code.py"
                if os.path.exists(os.path.join(snapshot_dir, "link_extraction_code.py"))
                else None,
            "listing_pages": listing_pages_meta,
            "articles": articles_meta,
            "article_fetch_failures": article_failures,
            "counts": {
                "listing_pages": len(listing_pages_meta),
                "articles": len(articles_meta),
                "article_fetch_failures": len(article_failures),
            },
        })

    # ══════════════════════════════════════════════════════════
    # Phase 2: Page 1 — link discovery (1 LLM call) + save listing HTML
    # ══════════════════════════════════════════════════════════
    print(f"\n⏳ Extracting article links from page 1: {listing_url}...")
    link_success, link_result = orch.call_links_agent_cli(listing_url, api_key, model)

    if not link_success:
        error_msg = link_result.get("error", "Unknown error")
        print(f"\n❌ Links extraction failed:\n{error_msg}")
        errors.append(f"Links extraction failed: {error_msg}")
        _write_metadata()
        write_snapshot_results_md({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "website": website, "snapshot_dir": snapshot_dir, "model": model,
            "pagination_type": pagination_type, "input_urls": input_urls,
            "pagination_pattern": pattern, "requirements": requirements,
            "listing_pages_saved": 0, "articles_saved": 0, "article_failures": 0,
            "llm_calls": orch._LLM_CALLS, "fetch_via": dict(orch._FETCH_VIA),
            "elapsed_seconds": time.time() - run_start, "errors": errors,
        })
        return

    page1_links = link_result.get("data", {}).get("article_links", [])
    page1_links, dropped = orch._filter_article_links(page1_links, listing_url, pattern)
    if dropped:
        print(f"  🧹 Dropped {dropped} non-article (pagination/category) link(s)")
    print(f"✓ Extracted {len(page1_links)} article links from page 1")

    # Save the generated link-extraction code for offline replay.
    link_code_file = link_result.get("code_file", "")
    link_extraction_code = ""
    if link_code_file and os.path.exists(link_code_file):
        with open(link_code_file, "r", encoding="utf-8") as f:
            link_extraction_code = f.read()
        with open(os.path.join(snapshot_dir, "link_extraction_code.py"), "w",
                  encoding="utf-8") as f:
            f.write(link_extraction_code)

    # Save the page-1 listing HTML (fetch our own faithful copy).
    try:
        listing_html, _ = await fetch_page_structure(listing_url)
    except Exception as e:
        listing_html = None
        errors.append(f"Failed to fetch listing page 1 HTML: {e}")
    if listing_html:
        fname = "listing_page_001.html"
        nbytes, sha1 = _save_html(os.path.join(snapshot_dir, fname), listing_html)
        listing_pages_meta.append({
            "page_num": 1, "url": listing_url, "file": fname,
            "bytes": nbytes, "sha1": sha1,
        })
        print(f"  💾 {fname}  ({nbytes:,} bytes)")
    else:
        print("  ⚠️  Could not save page-1 listing HTML (fetch returned nothing).")

    # Dynamic pagination: expand page 1 and save the fully-loaded HTML.
    if scroll_mode and link_extraction_code:
        full_html = await orch.collect_listing_html(
            listing_url, mode=scroll_mode,
            max_rounds=scroll_rounds, load_more_selector=scroll_selector)
        if full_html:
            fname = "listing_page_001_expanded.html"
            nbytes, sha1 = _save_html(os.path.join(snapshot_dir, fname), full_html)
            listing_pages_meta.append({
                "page_num": 1, "url": listing_url, "file": fname,
                "bytes": nbytes, "sha1": sha1, "expanded": True,
            })
            print(f"  💾 {fname}  ({nbytes:,} bytes)")
            expanded = orch.run_link_extraction_code(link_extraction_code, full_html)
            expanded, _ = orch._filter_article_links(expanded, listing_url, pattern)
            merged = {l["url"]: l for l in page1_links}
            for l in expanded:
                merged.setdefault(l["url"], l)
            page1_links = list(merged.values())
            print(f"  ✓ {len(page1_links)} unique links after {pagination_type}")

    if not page1_links:
        print("❌ No article links found on page 1!")
        errors.append("No article links found on page 1")
        _write_metadata()
        write_snapshot_results_md({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "website": website, "snapshot_dir": snapshot_dir, "model": model,
            "pagination_type": pagination_type, "input_urls": input_urls,
            "pagination_pattern": pattern, "requirements": requirements,
            "listing_pages_saved": len(listing_pages_meta), "articles_saved": 0,
            "article_failures": 0, "llm_calls": orch._LLM_CALLS,
            "fetch_via": dict(orch._FETCH_VIA),
            "elapsed_seconds": time.time() - run_start, "errors": errors,
        })
        return

    # Fetch + save page-1 articles.
    seen_urls = set()
    print(f"\n⏳ Fetching + saving page 1 articles ({len(page1_links)})...")
    arts, fails = await orch.fetch_articles(page1_links, label=" [page 1]")
    article_failures.extend(fails)
    for a in page1_links:
        seen_urls.add(a["url"])
    next_article_id += _save_articles(arts, snapshot_dir, 1, next_article_id, articles_meta)
    _write_metadata()
    print(f"\n💾 Page 1 done: {len(articles_meta)} articles saved, "
          f"{len(article_failures)} fetch failures")

    # ══════════════════════════════════════════════════════════
    # Phase 3: Subsequent pages (numbered pagination)
    # ══════════════════════════════════════════════════════════
    if page_urls and not link_extraction_code:
        print("\n⚠️  No link-extraction code available — cannot snapshot further pages.")
    elif page_urls:
        for idx, page_url in enumerate(page_urls):
            page_num = idx + 2
            print(f"\n{'─' * 60}")
            print(f"📄 Page {page_num}: {page_url[:80]}")
            print(f"{'─' * 60}")

            try:
                page_html, _ = await fetch_page_structure(page_url)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  ❌ Failed to fetch page: {e}")
                errors.append(f"Page {page_num} fetch failed: {e}\n{tb}")
                continue
            if not page_html:
                print("  ❌ Empty response for listing page")
                errors.append(f"Page {page_num}: empty listing response")
                continue

            # Save this listing page HTML.
            fname = f"listing_page_{page_num:03d}.html"
            nbytes, sha1 = _save_html(os.path.join(snapshot_dir, fname), page_html)
            listing_pages_meta.append({
                "page_num": page_num, "url": page_url, "file": fname,
                "bytes": nbytes, "sha1": sha1,
            })
            print(f"  💾 {fname}  ({nbytes:,} bytes)")

            # Reuse link code (no LLM) + dedup.
            new_links_raw = orch.run_link_extraction_code(link_extraction_code, page_html)
            new_links_raw, _ = orch._filter_article_links(new_links_raw, listing_url, pattern)
            print(f"  Found {len(new_links_raw)} links on this page")
            new_links = []
            for lnk in new_links_raw:
                if lnk["url"] not in seen_urls:
                    seen_urls.add(lnk["url"])
                    new_links.append(lnk)
            print(f"  {len(new_links)} new (after dedup)")

            if not new_links:
                _write_metadata()
                continue

            arts, fails = await orch.fetch_articles(new_links, label=f" [page {page_num}]")
            article_failures.extend(fails)
            next_article_id += _save_articles(
                arts, snapshot_dir, page_num, next_article_id, articles_meta)
            _write_metadata()
            print(f"  💾 Saved (total: {len(articles_meta)} articles, "
                  f"{len(article_failures)} fetch failures)")

            if idx < len(page_urls) - 1:
                time.sleep(1.5)

    # ══════════════════════════════════════════════════════════
    # Phase 4: Final metadata + results.md
    # ══════════════════════════════════════════════════════════
    _write_metadata()

    print(f"\n{'=' * 60}")
    print(f"📸 SNAPSHOT COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Website:        {website}")
    print(f"  Listing pages:  {len(listing_pages_meta)}")
    print(f"  Articles saved: {len(articles_meta)}")
    print(f"  Fetch failures: {len(article_failures)}")
    print(f"  LLM calls:      {orch._LLM_CALLS}")
    print(f"\n💾 Snapshot in: {snapshot_dir}/")
    print(f"   metadata.json — file ↔ url/title/page map + SHA-1 hashes")

    write_snapshot_results_md({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "website": website,
        "snapshot_dir": snapshot_dir,
        "model": model,
        "pagination_type": pagination_type,
        "input_urls": input_urls,
        "pagination_pattern": pattern,
        "requirements": requirements,
        "listing_pages_saved": len(listing_pages_meta),
        "articles_saved": len(articles_meta),
        "article_failures": len(article_failures),
        "llm_calls": orch._LLM_CALLS,
        "fetch_via": dict(orch._FETCH_VIA),
        "elapsed_seconds": time.time() - run_start,
        "errors": errors + [f.get("reason", "") for f in article_failures],
    })


if __name__ == "__main__":
    asyncio.run(main())
