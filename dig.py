"""
Standalone fetch diagnostic.

Bypasses the orchestrator and the subprocess layer entirely: it calls
fetch_page_structure() directly and prints the full chained error
(_LAST_FETCH_ERROR) plus a short HTML preview if a page comes back.

Usage:
    python diagnose_fetch.py
    python diagnose_fetch.py https://mana.net/category/articles/
"""

import asyncio
import sys

import Links_Agent_gemma_cloudflare as agent


async def run(url: str):
    print("=" * 70)
    print(f"🔬 DIAGNOSTIC FETCH — {url}")
    print("=" * 70)

    html, smap = await agent.fetch_page_structure(url)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    if html is None:
        print("❌ html_content is None — fetch failed at every layer.\n")
        print("── Captured error (_LAST_FETCH_ERROR) ──")
        print(agent._LAST_FETCH_ERROR or "(nothing captured)")
        return

    print(f"✓ HTML received: {len(html)} bytes")
    print(f"  Structural map nodes (top level): "
          f"{len(smap) if smap else 0}")

    if not smap:
        print("⚠  Structural map is EMPTY (page fetched but no target tags matched).")

    print("\n── First 800 chars of HTML ──")
    print(html[:800])

    # Quick sanity check: does it still look like a challenge page?
    if agent._is_challenge_page(html):
        print("\n⚠  WARNING: HTML still matches a Cloudflare/bot challenge marker.")


def main():
    url = sys.argv[1]
    asyncio.run(run(url))

if __name__ == "__main__":
    main()