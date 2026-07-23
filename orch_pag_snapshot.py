"""Combined orchestrator + snapshot + evaluation-table writer.

This is `orch_interactive_pagination.py` **plus** the HTML snapshotting from
`snapshot_collector.py`, in a single pass. Run it once and you get:

1. The full extraction pipeline output (clustering, per-cluster code generation,
   reuse, incremental saves) — everything the numbered-pagination orchestrator
   produces: `extracted_data_all.json`, `failed_links.json`, `clusters.json`,
   `progress.json` in a timestamped `orch_runs/` directory.
2. A frozen HTML snapshot of the site (raw listing + article HTML, the generated
   link-extraction code, and a `metadata.json` file↔URL map with SHA-1 hashes)
   under `snapshots/`, so the run is re-scorable offline later.
3. A per-run section appended to `results.md` (same stats as the orchestrator).
4. A per-run row appended to `detailed_evaluation_table.md` — a compact,
   appendix-friendly table whose objective columns are auto-filled and whose last
   two columns (**Manually checked articles**, **Notes**) are left blank for you
   to fill in by hand after review.

All heavy logic is reused from `orch_interactive_pagination.py` and `snapshot_collector.py`;
only the top-level flow is assembled here, so the original orchestrator is
unchanged.

Usage::

    python orch_pag_snapshot.py

Answer the prompts exactly as you would for `orch_interactive_pagination.py`.
"""

import asyncio
import os
import time
import traceback
from datetime import datetime

# Reuse the orchestrator as a namespace so its live counters and every helper
# (fetching, clustering, extraction, pagination, results.md) are shared.
import orch_interactive_pagination as orch
from snapshot_collector import _save_html, _save_articles, _domain_slug
from Agent_for_single_page_gemma import fetch_page_structure
from utils import list_available_models


# ═══════════════════════════════════════════════════════════════
# detailed_evaluation_table.md  (paper appendix)
# ═══════════════════════════════════════════════════════════════

_EVAL_TABLE_COLUMNS = [
    "#", "Website", "Date", "Pagination", "Pages", "Extracted", "Failed",
    "Clusters", "LLM calls", "Calls/article", "Body N/A %", "Run dir",
    "Snapshot dir", "Manually checked articles", "Notes",
]


def _next_row_num(path):
    """Count existing data rows in the eval table to number the next one."""
    if not os.path.exists(path):
        return 1
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("|"):
                continue
            if s.startswith("| #"):            # header row
                continue
            if set(s) <= set("|-: "):          # separator row
                continue
            n += 1
    return n + 1


def _body_na_pct(all_extracted):
    """Return the N/A percentage of the long/body field, or None."""
    total, na_stats = orch._field_na_stats(all_extracted)
    if total == 0 or not na_stats:
        return None
    long_keys = ("body", "content", "text", "article", "description", "summary")
    for field, na, tot, pct in na_stats:
        if any(k in field.lower() for k in long_keys):
            return pct
    return None


def write_eval_table(stats, path="detailed_evaluation_table.md"):
    """Append one auto-filled row (with two blank manual columns) per run."""
    header_needed = not os.path.exists(path)
    num = _next_row_num(path)

    lines = []
    if header_needed:
        lines.append("# Detailed Evaluation Table\n")
        lines.append(
            "Auto-generated per-run summary for the paper appendix. The objective "
            "columns are filled automatically; fill in **Manually checked articles** "
            "(e.g. article IDs from the snapshot's `metadata.json`) and **Notes** by "
            "hand after review.\n"
        )
        lines.append("| " + " | ".join(_EVAL_TABLE_COLUMNS) + " |")
        lines.append("|" + "|".join(["---"] * len(_EVAL_TABLE_COLUMNS)) + "|")

    ext = stats.get("articles_extracted", 0) or 0
    llm = stats.get("llm_calls", 0) or 0
    cpa = (llm / ext) if ext else 0.0
    body_na = stats.get("body_na_pct")
    body_na_str = f"{body_na:.0f}%" if isinstance(body_na, (int, float)) else "—"

    row = [
        str(num),
        stats.get("website", "N/A"),
        stats.get("date", ""),
        stats.get("pagination_type", "N/A"),
        str(stats.get("pages_processed", 0)),
        str(ext),
        str(stats.get("articles_failed", 0)),
        str(stats.get("clusters", 0)),
        str(llm),
        f"{cpa:.2f}",
        body_na_str,
        f"`{stats.get('run_dir', '')}`",
        f"`{stats.get('snapshot_dir', '')}`",
        " ",   # Manually checked articles — fill by hand
        " ",   # Notes — fill by hand
    ]
    lines.append("| " + " | ".join(row) + " |")

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"📝 Eval-table row appended to {path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("🎯📸 ORCHESTRATOR + SNAPSHOT")
    print("=" * 60)

    orch._reset_llm_calls()
    orch._reset_fetch_via()
    orch._reset_tokens()
    orch._reset_map_fits()

    run_start = time.time()
    input_urls = []
    pagination_type = "single page"

    # ── Setup ────────────────────────────────────────────────
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
        elif choice:
            model = choice[7:] if choice.startswith("models/") else choice
            print(f"✓ Selected: {model}")
        else:
            model = "gemma-3-27b-it"
            print(f"✓ Using default: {model}")
    else:
        model = input("\n🤖 Model name [Enter for gemma-3-27b-it]: ").strip() or "gemma-3-27b-it"

    run_dir = f"orch_runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n📂 Run directory: {run_dir}")

    listing_url = input("\n🌐 Enter the listing page URL: ").strip()
    if not listing_url:
        print("❌ URL is required!")
        return
    input_urls.append(listing_url)

    website = input(
        "\n🏷️  Website name for the snapshot/eval table [Enter to use the domain]: "
    ).strip() or _domain_slug(listing_url)

    requirements = input(
        "\n📝 What data to extract from each article?\n"
        "   (e.g., 'title, date, author, article body text')\n   → "
    ).strip() or "title, date, author, article body text"

    # Snapshot directory (parallel to the run directory).
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = os.path.join("snapshots", f"{_domain_slug(listing_url)}_{ts}")
    os.makedirs(snapshot_dir, exist_ok=True)
    print(f"📸 Snapshot directory: {snapshot_dir}")

    # ── Phase 1: pagination setup (identical to the orchestrator) ──
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
                print(f"\n   ⚠️  Could not derive pattern automatically.")
                print(f"   Enter the URL pattern manually (use {{page}} as placeholder):")
                manual_pat = input("   → ").strip()
                if manual_pat and "{page}" in manual_pat:
                    pattern = manual_pat
                    total_str = input("   📄 How many pages to scrape (including page 1)? → ").strip()
                    if total_str.isdigit() and int(total_str) >= 2:
                        total_pages = int(total_str)
                        page_urls = [pattern.format(page=n) for n in range(2, total_pages + 1)]
                        print(f"   ✓ Will scrape {total_pages} pages ({len(page_urls)} after page 1)")
                    else:
                        print("   ⚠️  Need at least 2 pages. Proceeding with page 1 only.")
                else:
                    print("   ⚠️  Invalid or missing pattern. Proceeding with page 1 only.")
    else:
        print("   ✓ Single page — no pagination.")

    # Snapshot metadata accumulators.
    listing_pages_meta = []
    articles_meta = []
    snapshot_failures = []
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
            "run_dir": run_dir,
            "link_extraction_code_file": "link_extraction_code.py"
                if os.path.exists(os.path.join(snapshot_dir, "link_extraction_code.py"))
                else None,
            "listing_pages": listing_pages_meta,
            "articles": articles_meta,
            "article_fetch_failures": snapshot_failures,
            "counts": {
                "listing_pages": len(listing_pages_meta),
                "articles": len(articles_meta),
                "article_fetch_failures": len(snapshot_failures),
            },
        })

    # ── Phase 2: Page 1 — link discovery (LLM call #1) ──
    print(f"\n⏳ Extracting article links from page 1: {listing_url}...")
    link_success, link_result = orch.call_links_agent_cli(listing_url, api_key, model)

    if not link_success:
        error_msg = link_result.get("error", "Unknown error")
        print(f"\n❌ Links extraction failed:\n{error_msg}")
        orch.write_results_md({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": run_dir, "model": model,
            "pagination_type": pagination_type, "input_urls": input_urls,
            "requirements": requirements,
            "pages_requested": len(page_urls) + 1, "pages_processed": 0,
            "articles_extracted": 0, "articles_failed": 0, "clusters": 0,
            "llm_calls": orch._LLM_CALLS,
            "llm_calls_by_agent": dict(orch._LLM_CALLS_BY_AGENT),
            "tokens_by_agent": dict(orch._TOKENS_BY_AGENT),
            "elapsed_seconds": time.time() - run_start,
            "errors": [f"Links extraction failed: {error_msg}"],
        })
        _write_metadata()
        return

    page1_links = link_result.get("data", {}).get("article_links", [])
    page1_links, dropped = orch._filter_article_links(page1_links, listing_url, pattern)
    if dropped:
        print(f"  🧹 Dropped {dropped} non-article (pagination/category) link(s)")
    link_code_file = link_result.get("code_file", "")
    print(f"✓ Extracted {len(page1_links)} article links from page 1")

    # Snapshot: save the generated link-extraction code.
    link_extraction_code = ""
    if link_code_file and os.path.exists(link_code_file):
        with open(link_code_file, "r", encoding="utf-8") as f:
            link_extraction_code = f.read()
        with open(os.path.join(snapshot_dir, "link_extraction_code.py"), "w",
                  encoding="utf-8") as f:
            f.write(link_extraction_code)

    # Snapshot: save page-1 listing HTML.
    try:
        listing_html, _ = await fetch_page_structure(listing_url)
    except Exception as e:
        listing_html = None
        snapshot_failures.append({"url": listing_url, "title": "",
                                  "reason": f"listing fetch failed: {e}"})
    if listing_html:
        fname = "listing_page_001.html"
        nbytes, sha1 = _save_html(os.path.join(snapshot_dir, fname), listing_html)
        listing_pages_meta.append({"page_num": 1, "url": listing_url, "file": fname,
                                   "bytes": nbytes, "sha1": sha1})

    if not page1_links:
        print("❌ No article links found on page 1!")
        orch.write_results_md({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": run_dir, "model": model,
            "pagination_type": pagination_type, "input_urls": input_urls,
            "requirements": requirements,
            "pages_requested": len(page_urls) + 1, "pages_processed": 0,
            "articles_extracted": 0, "articles_failed": 0, "clusters": 0,
            "llm_calls": orch._LLM_CALLS,
            "llm_calls_by_agent": dict(orch._LLM_CALLS_BY_AGENT),
            "tokens_by_agent": dict(orch._TOKENS_BY_AGENT),
            "elapsed_seconds": time.time() - run_start,
            "errors": ["No article links found on page 1"],
        })
        _write_metadata()
        return

    # Dynamic pagination: expand page 1 and snapshot the expanded HTML.
    if scroll_mode and link_extraction_code:
        full_html = await orch.collect_listing_html(
            listing_url, mode=scroll_mode,
            max_rounds=scroll_rounds, load_more_selector=scroll_selector)
        if full_html:
            fname = "listing_page_001_expanded.html"
            nbytes, sha1 = _save_html(os.path.join(snapshot_dir, fname), full_html)
            listing_pages_meta.append({"page_num": 1, "url": listing_url, "file": fname,
                                       "bytes": nbytes, "sha1": sha1, "expanded": True})
            expanded = orch.run_link_extraction_code(link_extraction_code, full_html)
            expanded, _ = orch._filter_article_links(expanded, listing_url, pattern)
            print(f"  ✓ {len(expanded)} links after {pagination_type} "
                  f"(was {len(page1_links)} on initial load)")
            merged = {l["url"]: l for l in page1_links}
            for l in expanded:
                merged.setdefault(l["url"], l)
            page1_links = list(merged.values())
            print(f"  ✓ {len(page1_links)} unique links total")

    # ── Phase 2 cont.: process page 1 (extract) + snapshot articles ──
    all_extracted = []
    all_failures = []
    cluster_registry = {}
    seen_urls = set()

    print(f"\n⏳ Fetching structural maps for page 1 ({len(page1_links)} articles)...")
    arts, fails = await orch.fetch_articles(page1_links, label=" [page 1]")
    all_failures.extend(fails)
    snapshot_failures.extend(fails)
    for a in page1_links:
        seen_urls.add(a["url"])

    # Snapshot: save article HTML before extraction.
    next_article_id += _save_articles(arts, snapshot_dir, 1, next_article_id, articles_meta)
    _write_metadata()

    if arts:
        print("\n⏳ Clustering page 1 articles and extracting data...")
        ext, fl, cluster_registry = orch.process_page_articles(
            arts, cluster_registry, api_key, model, requirements)
        all_extracted.extend(ext)
        all_failures.extend(fl)

    max_pages = len(page_urls) + 1
    progress = {
        "current_page": 1, "total_pages": max_pages,
        "extracted_count": len(all_extracted), "failed_count": len(all_failures),
        "cluster_registry": {sig: {"code_file": v["code_file"]}
                             for sig, v in cluster_registry.items()},
    }
    orch.save_incremental(run_dir, all_extracted, all_failures, progress)
    orch._atomic_json_write(os.path.join(run_dir, "extracted_links.json"),
                            {"article_links": page1_links})
    print(f"\n💾 Page 1 saved ({len(all_extracted)} extracted, {len(all_failures)} failed)")

    # ── Phase 3: pages 2..N (extract + snapshot) ──
    if page_urls and not link_extraction_code:
        print("\n⚠️  No link-extraction code available — cannot process further pages.")
    elif page_urls:
        for idx, page_url in enumerate(page_urls):
            page_num = idx + 2
            print(f"\n{'─' * 60}")
            print(f"📄 Page {page_num}/{max_pages}: {page_url[:80]}")
            print(f"{'─' * 60}")

            try:
                page_html, _ = await fetch_page_structure(page_url)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  ❌ Failed to fetch page: {e}\n{tb}")
                continue
            if not page_html:
                print("  ❌ Empty response for listing page")
                continue

            # Snapshot: save this listing page HTML.
            fname = f"listing_page_{page_num:03d}.html"
            nbytes, sha1 = _save_html(os.path.join(snapshot_dir, fname), page_html)
            listing_pages_meta.append({"page_num": page_num, "url": page_url,
                                       "file": fname, "bytes": nbytes, "sha1": sha1})

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
                progress["current_page"] = page_num
                orch.save_incremental(run_dir, all_extracted, all_failures, progress)
                _write_metadata()
                continue

            arts, fails = await orch.fetch_articles(new_links, label=f" [page {page_num}]")
            all_failures.extend(fails)
            snapshot_failures.extend(fails)

            # Snapshot: save article HTML before extraction.
            next_article_id += _save_articles(
                arts, snapshot_dir, page_num, next_article_id, articles_meta)
            _write_metadata()

            if arts:
                ext, fl, cluster_registry = orch.process_page_articles(
                    arts, cluster_registry, api_key, model, requirements)
                all_extracted.extend(ext)
                all_failures.extend(fl)

            progress["current_page"] = page_num
            progress["extracted_count"] = len(all_extracted)
            progress["failed_count"] = len(all_failures)
            progress["cluster_registry"] = {sig: {"code_file": v["code_file"]}
                                            for sig, v in cluster_registry.items()}
            orch.save_incremental(run_dir, all_extracted, all_failures, progress)
            print(f"  💾 Saved (total: {len(all_extracted)} extracted, "
                  f"{len(all_failures)} failed)")

            if idx < len(page_urls) - 1:
                time.sleep(1.5)

    # ── Phase 4: final saves ──
    cluster_info = {sig: {"code_file": v.get("code_file", "")}
                    for sig, v in cluster_registry.items()}
    orch._atomic_json_write(os.path.join(run_dir, "clusters.json"), cluster_info)
    orch.save_incremental(run_dir, all_extracted, all_failures, {
        "status": "finished",
        "extracted_count": len(all_extracted),
        "failed_count": len(all_failures),
    })
    _write_metadata()

    orch._print_summary(run_dir, all_extracted, all_failures)

    pages_processed = (len(page_urls) + 1) if page_urls else 1

    # results.md — identical to the orchestrator's per-run stats.
    orch.write_results_md({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": run_dir, "model": model,
        "pagination_type": pagination_type, "input_urls": input_urls,
        "requirements": requirements,
        "pages_requested": pages_processed, "pages_processed": pages_processed,
        "articles_extracted": len(all_extracted), "articles_failed": len(all_failures),
        "clusters": len(cluster_registry),
        "llm_calls": orch._LLM_CALLS,
        "llm_calls_by_agent": dict(orch._LLM_CALLS_BY_AGENT),
        "elapsed_seconds": time.time() - run_start,
        "fetch_via": dict(orch._FETCH_VIA),
        "tokens_by_agent": dict(orch._TOKENS_BY_AGENT),
        "extracted_data": all_extracted,
        "errors": [f.get("reason", "") for f in all_failures],
    })

    # detailed_evaluation_table.md — appendix row (manual columns blank).
    write_eval_table({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "website": website,
        "pagination_type": pagination_type,
        "pages_processed": pages_processed,
        "articles_extracted": len(all_extracted),
        "articles_failed": len(all_failures),
        "clusters": len(cluster_registry),
        "llm_calls": orch._LLM_CALLS,
        "body_na_pct": _body_na_pct(all_extracted),
        "run_dir": run_dir,
        "snapshot_dir": snapshot_dir,
    })

    print(f"\n📸 Snapshot saved in: {snapshot_dir}/ "
          f"({len(articles_meta)} article HTML files)")


if __name__ == "__main__":
    asyncio.run(main())
