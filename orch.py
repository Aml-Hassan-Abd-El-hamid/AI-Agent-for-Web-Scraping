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
from collections import defaultdict
from datetime import datetime

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

        print(f"\n⏳ Extracting article links from {listing_url}...")
        link_success, link_result = call_links_agent_cli(listing_url, api_key, model)

        if not link_success:
            error_msg = link_result.get("error", "Unknown error")
            print(f"\n❌ Links extraction failed:\n{error_msg}")
            return

        article_links = link_result.get("data", {}).get("article_links", [])
        print(f"✓ Extracted {len(article_links)} article links")

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

