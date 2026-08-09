"""
Paginated Links Collector for pchrgaza.org testimonies.

Fetches all 19 listing pages, extracts article links using the
already-working extraction code, deduplicates, and saves a single
JSON file ready for orch.py (option 2).

Usage:
    python paginated_links_collector.py
    python paginated_links_collector.py --pages 19 --delay 2
    python paginated_links_collector.py --output my_links.json
"""

import asyncio
import argparse
import json
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


BASE_LISTING_URL = "https://pchrgaza.org/ar/category/genocide-on-gaza-ar/testimonies-from-the-war-ar/page/{page}/"
OUTPUT_FILE = "all_pchrgaza_links.json"


def extract_links_from_html(html_content: str) -> list:
    """
    Extract article links from a single listing page HTML.
    Reuses the working logic from code/gemma_gen_code_98604.py.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    base_url = "https://pchrgaza.org/"

    article_links = []
    for item in soup.select("div.news-item"):
        a_tag = item.select_one("a.title.four-lines")
        if not a_tag or not a_tag.get("href"):
            continue

        href = a_tag["href"]
        if href.startswith("javascript:") or href == "#":
            continue

        url = urljoin(base_url, href)
        title = a_tag.get_text(strip=True)
        article_links.append({"url": url, "title": title})

    return article_links


async def fetch_page_html(page_obj, url: str) -> str:
    """Fetch a single listing page's HTML using Playwright."""
    await page_obj.goto(url, wait_until="domcontentloaded", timeout=60000)
    await page_obj.wait_for_timeout(2000)
    return await page_obj.content()


async def collect_all_links(total_pages: int, delay: float) -> list:
    """Fetch all listing pages and collect deduplicated article links."""
    seen_urls = set()
    all_links = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page_obj = await browser.new_page()

        for page_num in range(1, total_pages + 1):
            url = BASE_LISTING_URL.format(page=page_num)
            print(f"[{page_num}/{total_pages}] Fetching: {url}")

            try:
                html = await fetch_page_html(page_obj, url)
                links = extract_links_from_html(html)
                new_count = 0
                for link in links:
                    if link["url"] not in seen_urls:
                        seen_urls.add(link["url"])
                        all_links.append(link)
                        new_count += 1
                print(f"    → {len(links)} links found, {new_count} new  (total unique: {len(all_links)})")
            except Exception as e:
                print(f"    ✗ Error on page {page_num}: {e}")

            if page_num < total_pages and delay > 0:
                time.sleep(delay)

        await browser.close()

    return all_links


def main():
    parser = argparse.ArgumentParser(description="Collect article links from all paginated listing pages.")
    parser.add_argument("--pages", type=int, default=19, help="Total number of listing pages (default: 19)")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay in seconds between page fetches (default: 1.5)")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help=f"Output JSON file (default: {OUTPUT_FILE})")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  pchrgaza.org Paginated Links Collector")
    print(f"  Pages: 1–{args.pages}  |  Delay: {args.delay}s  |  Output: {args.output}")
    print(f"{'='*60}\n")

    all_links = asyncio.run(collect_all_links(args.pages, args.delay))

    output = {"article_links": all_links}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done! {len(all_links)} unique article links saved to {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
