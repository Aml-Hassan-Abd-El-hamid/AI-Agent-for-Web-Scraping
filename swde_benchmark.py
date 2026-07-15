"""SWDE benchmark harness for the web-scraping agent (paper P0 item).

Runs the project's LLM code-synthesis extractor on the standard **SWDE**
(Structured Web Data Extraction) dataset so its accuracy can be compared to the
published numbers for MarkupLM, FreeDOM, ZeroShotCeres, SimpleDOM, etc.

Why this script exists
----------------------
SWDE is the canonical benchmark in this line of research. Reporting SWDE
page-level attribute F1 is what lets a research venue place this system next to
prior work. Everything here reuses the existing agent (no pipeline edits) and,
crucially, the extraction step touches *no network* — SWDE pages are local HTML,
so only the per-site LLM code-generation call needs the API.

How it mirrors the paper's method
---------------------------------
For each (vertical, site), the harness generates ONE `extract_data` function
from a single representative page's structural map, then *reuses* that function
across every page of the site. This is exactly the paper's template-clustering /
code-reuse story: 1 LLM call amortized over ~2000 pages per site.

⚠️ COMPARISON CAVEAT (read before quoting numbers)
--------------------------------------------------
The standard SWDE protocol for MarkupLM / FreeDOM / SimpleDOM is *supervised
seed-site transfer*: train on k in {1..10} seed sites of a vertical, test on the
held-out sites. This harness instead does **per-site zero-label wrapper
induction**: it generates code from the target site's own structure using no
labels at all. So the honest comparison is against:
  * ZeroShotCeres (zero-shot / unseen-site) numbers,
  * classic unsupervised wrapper-induction baselines,
  * recent LLM-based SWDE results reporting a zero-/few-shot setting.
State this setting explicitly in the paper; do NOT compare directly to the k=10
supervised rows without saying so.

Workflow
--------
1. Download + unzip SWDE so you have a folder with `sourceCode/` and
   `groundtruth/` (the standard layout).
2. Build gold + manifest (offline, stdlib only)::

       python swde_benchmark.py prepare --swde-root path/to/swde \
           --out-dir swde_prepared

3. Run the extractor over sites (online, needs API key)::

       python swde_benchmark.py run --swde-root path/to/swde \
           --prepared-dir swde_prepared --pred-dir swde_pred \
           --api-key $env:GOOGLE_API_KEY --model gemma-3-27b-it \
           --pages-per-site 200 --limit-sites 2

4. Score with the SWDE page-level attribute F1 protocol (offline)::

       python swde_benchmark.py score --prepared-dir swde_prepared \
           --pred-dir swde_pred --append-results
"""

import argparse
import glob
import json
import os
import re
import sys


# ════════════════════════════════════════════════════════════════════
# Shared: SWDE value normalization + IO
# ════════════════════════════════════════════════════════════════════

def _norm(value):
    """Normalize a string for SWDE matching: lowercase, strip, collapse ws."""
    if value is None:
        return ""
    s = str(value).replace("\xa0", " ").strip().lower()
    s = s.strip("\"'\u201c\u201d\u2018\u2019 \t")
    return re.sub(r"\s+", " ", s)


def _norm_attr(name):
    """Canonicalize an attribute/field key (spaces/hyphens -> underscore)."""
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def _read_text(path):
    """Read an HTML file tolerantly (SWDE pages use mixed encodings)."""
    with open(path, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8", errors="ignore")


def _resolve_dirs(root, source_subdir, gt_subdir):
    """Locate the sourceCode/ and groundtruth/ roots, tolerating one nesting."""
    src = os.path.join(root, source_subdir)
    gt = os.path.join(root, gt_subdir)
    # Tolerate the common sourceCode/sourceCode double-nesting.
    if os.path.isdir(os.path.join(src, source_subdir)):
        src = os.path.join(src, source_subdir)
    if not os.path.isdir(src):
        # Maybe root already IS the sourceCode dir and gt is a sibling.
        if os.path.basename(os.path.normpath(root)) == source_subdir:
            src = root
            gt = os.path.join(os.path.dirname(os.path.normpath(root)), gt_subdir)
    return src, gt


# ════════════════════════════════════════════════════════════════════
# prepare — parse groundtruth into gold + build a run manifest
# ════════════════════════════════════════════════════════════════════

_PAGEID_RE = re.compile(r"^\d{4}$")
_GT_NAME_RE = re.compile(r"^(?P<vertical>[^-]+)-(?P<site>.+)-(?P<attr>[^-]+)\.txt$")


def _parse_groundtruth_file(path):
    """Parse one SWDE groundtruth .txt into {pageid: [values]}.

    Robust to the 1-2 header lines by only accepting rows whose first tab field
    is a 4-digit page id. '<NULL>' values are dropped.
    """
    per_page = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or not _PAGEID_RE.match(parts[0]):
                continue
            page_id = parts[0]
            # parts[1] is the value count; parts[2:] are the values.
            values = [v for v in parts[2:] if v and v != "<NULL>"]
            per_page[page_id] = values
    return per_page


def _site_dir_map(vertical_src_dir, vertical):
    """Map plain site name -> its source folder (folder embeds a page count)."""
    mapping = {}
    for d in glob.glob(os.path.join(vertical_src_dir, f"{vertical}-*")):
        if not os.path.isdir(d):
            continue
        base = os.path.basename(d)
        # base looks like 'auto-aol(2000)'; strip vertical prefix + (count).
        m = re.match(rf"^{re.escape(vertical)}-(?P<site>.+?)\(\d+\)$", base)
        site = m.group("site") if m else base[len(vertical) + 1:]
        mapping[site] = d
    return mapping


def cmd_prepare(args):
    src_root, gt_root = _resolve_dirs(
        args.swde_root, args.source_subdir, args.groundtruth_subdir)
    if not os.path.isdir(src_root) or not os.path.isdir(gt_root):
        print(f"Could not find SWDE dirs.\n  sourceCode: {src_root}\n"
              f"  groundtruth: {gt_root}\n"
              f"Point --swde-root at the folder containing "
              f"'{args.source_subdir}' and '{args.groundtruth_subdir}'.")
        return 1

    verticals = sorted(
        d for d in os.listdir(gt_root)
        if os.path.isdir(os.path.join(gt_root, d))
    )
    if args.verticals:
        wanted = {v.strip() for v in args.verticals.split(",")}
        verticals = [v for v in verticals if v in wanted]

    gold_dir = os.path.join(args.out_dir, "gold")
    os.makedirs(gold_dir, exist_ok=True)

    manifest = {"verticals": {}}
    total_pages = 0

    for vertical in verticals:
        gt_vdir = os.path.join(gt_root, vertical)
        src_vdir = os.path.join(src_root, vertical)
        site_dirs = _site_dir_map(src_vdir, vertical)

        # gold[site][page_id][attr] = [values]; also collect attributes.
        gold = {}
        attributes = set()
        for gt_file in glob.glob(os.path.join(gt_vdir, f"{vertical}-*.txt")):
            m = _GT_NAME_RE.match(os.path.basename(gt_file))
            if not m:
                continue
            site = m.group("site")
            attr = _norm_attr(m.group("attr"))
            attributes.add(attr)
            per_page = _parse_groundtruth_file(gt_file)
            site_gold = gold.setdefault(site, {})
            for page_id, values in per_page.items():
                site_gold.setdefault(page_id, {})[attr] = values

        attributes = sorted(attributes)
        v_sites = {}
        for site, pages in gold.items():
            site_dir = site_dirs.get(site)
            if not site_dir:
                # No matching source folder — skip (gold without HTML).
                continue
            page_ids = sorted(pages.keys())
            v_sites[site] = {"dir": site_dir, "page_ids": page_ids}
            total_pages += len(page_ids)

            # Write per-site gold keyed by pageKey with full value lists.
            gold_out = {}
            for page_id in page_ids:
                key = f"{vertical}/{site}/{page_id}"
                gold_out[key] = {a: pages[page_id].get(a, []) for a in attributes}
            with open(os.path.join(gold_dir, f"{vertical}__{site}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(gold_out, f, ensure_ascii=False)

        manifest["verticals"][vertical] = {
            "attributes": attributes,
            "sites": v_sites,
        }
        print(f"  {vertical}: {len(v_sites)} sites, "
              f"attrs={attributes}")

    with open(os.path.join(args.out_dir, "manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    n_sites = sum(len(v["sites"]) for v in manifest["verticals"].values())
    print(f"\nPrepared {len(manifest['verticals'])} verticals, "
          f"{n_sites} sites, {total_pages} pages.")
    print(f"  Manifest: {os.path.join(args.out_dir, 'manifest.json')}")
    print(f"  Gold:     {gold_dir}/<vertical>__<site>.json")
    return 0


# ════════════════════════════════════════════════════════════════════
# run — synthesize one extractor per site, reuse across its pages
# ════════════════════════════════════════════════════════════════════

def _page_path(site_dir, page_id):
    p = os.path.join(site_dir, f"{page_id}.htm")
    if os.path.isfile(p):
        return p
    alt = os.path.join(site_dir, f"{page_id}.html")
    return alt if os.path.isfile(alt) else p


def cmd_run(args):
    # Heavy imports are deferred so `prepare`/`score` stay stdlib-only.
    import google.generativeai as genai  # noqa: F401 (configured via GemmaAgent)
    from bs4 import BeautifulSoup
    from Agent_for_single_page_gemma import (
        create_structural_map, execute_extraction_code, GemmaAgent,
    )

    with open(os.path.join(args.prepared_dir, "manifest.json"),
              "r", encoding="utf-8") as f:
        manifest = json.load(f)

    verticals = manifest["verticals"]
    if args.verticals:
        wanted = {v.strip() for v in args.verticals.split(",")}
        verticals = {k: v for k, v in verticals.items() if k in wanted}

    os.makedirs(args.pred_dir, exist_ok=True)
    agent = GemmaAgent(api_key=args.api_key, model_name=args.model)

    total_sites = 0
    total_llm_calls = 0
    total_pages = 0

    for vertical, vinfo in verticals.items():
        attributes = vinfo["attributes"]
        requirements = ", ".join(attributes)
        sites = list(vinfo["sites"].items())
        if args.limit_sites:
            sites = sites[: args.limit_sites]

        for site, sinfo in sites:
            out_path = os.path.join(args.pred_dir, f"{vertical}__{site}.json")
            if os.path.isfile(out_path) and not args.overwrite:
                print(f"  skip {vertical}/{site} (exists)")
                continue

            site_dir = sinfo["dir"]
            page_ids = sinfo["page_ids"]
            if args.pages_per_site:
                page_ids = page_ids[: args.pages_per_site]
            if not page_ids:
                continue

            # ── Generate one extractor from a representative page ──
            rep_id = page_ids[0]
            rep_html = _read_text(_page_path(site_dir, rep_id))
            smap = create_structural_map(BeautifulSoup(rep_html, "html.parser"))
            smap_json = json.dumps(smap, ensure_ascii=False)
            if len(smap_json) > args.max_map_chars:
                smap_json = smap_json[: args.max_map_chars]

            rep_key = f"{vertical}/{site}/{rep_id}"
            code = agent.generate_extraction_code(
                smap_json, requirements, page_url=rep_key)
            total_llm_calls += 1
            ok, res = execute_extraction_code(code, rep_html)
            attempt = 0
            while (not ok or not isinstance(res, dict)) \
                    and attempt < args.repair_retries:
                err = res if not ok else "extract_data returned a non-dict"
                code = agent.generate_extraction_code(
                    smap_json, requirements, page_url=rep_key,
                    error_context=str(err))
                total_llm_calls += 1
                ok, res = execute_extraction_code(code, rep_html)
                attempt += 1

            if not ok or not isinstance(res, dict):
                print(f"  ⚠️ {vertical}/{site}: code gen failed "
                      f"({str(res)[:80]}) — writing empty predictions")

            # ── Reuse the code across every page of the site (no LLM) ──
            preds = {}
            for page_id in page_ids:
                key = f"{vertical}/{site}/{page_id}"
                try:
                    html = _read_text(_page_path(site_dir, page_id))
                except Exception:
                    preds[key] = {a: "" for a in attributes}
                    continue
                pok, pres = execute_extraction_code(code, html)
                row = {}
                if pok and isinstance(pres, dict):
                    # Map predicted keys onto SWDE attributes by normalized name.
                    norm_lookup = {_norm_attr(k): v for k, v in pres.items()}
                    for a in attributes:
                        val = norm_lookup.get(a, "")
                        if isinstance(val, (list, tuple)):
                            val = val[0] if val else ""
                        row[a] = "" if val is None else str(val)
                else:
                    row = {a: "" for a in attributes}
                preds[key] = row

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(preds, f, ensure_ascii=False)

            total_sites += 1
            total_pages += len(page_ids)
            reuse = (len(page_ids) / max(1, total_llm_calls))
            print(f"  ✓ {vertical}/{site}: {len(page_ids)} pages, "
                  f"code from 1 page (LLM calls so far: {total_llm_calls})")

    print(f"\nRun complete: {total_sites} sites, {total_pages} pages, "
          f"{total_llm_calls} LLM calls "
          f"({total_pages / max(1, total_llm_calls):.0f} pages/call).")
    print(f"  Predictions: {args.pred_dir}/<vertical>__<site>.json")
    return 0


# ════════════════════════════════════════════════════════════════════
# score — SWDE page-level attribute precision / recall / F1
# ════════════════════════════════════════════════════════════════════

def cmd_score(args):
    gold_dir = os.path.join(args.prepared_dir, "gold")
    gold_files = sorted(glob.glob(os.path.join(gold_dir, "*.json")))
    if not gold_files:
        print(f"No gold files under {gold_dir}. Run `prepare` first.")
        return 1

    wanted = ({v.strip() for v in args.verticals.split(",")}
              if args.verticals else None)

    # Accumulate per (vertical, attribute) page-level counts.
    # counts[(vertical, attr)] = [correct, pred_nonempty, gold_nonempty]
    counts = {}
    scored_sites = 0
    missing_pred = 0

    for gf in gold_files:
        stem = os.path.basename(gf)[:-5]  # strip .json
        vertical, _, site = stem.partition("__")
        if wanted and vertical not in wanted:
            continue
        with open(gf, "r", encoding="utf-8") as f:
            gold = json.load(f)
        pred_path = os.path.join(args.pred_dir, f"{vertical}__{site}.json")
        if not os.path.isfile(pred_path):
            missing_pred += 1
            continue
        with open(pred_path, "r", encoding="utf-8") as f:
            pred = json.load(f)
        scored_sites += 1

        for key, gattrs in gold.items():
            prow = pred.get(key, {})
            for attr, gvals in gattrs.items():
                slot = counts.setdefault((vertical, attr), [0, 0, 0])
                g_has = bool(gvals)
                pval = prow.get(attr, "")
                p_has = not (pval is None or str(pval).strip() == "")
                if g_has:
                    slot[2] += 1
                if p_has:
                    slot[1] += 1
                if g_has and p_has:
                    gold_norm = {_norm(v) for v in gvals}
                    if _norm(pval) in gold_norm:
                        slot[0] += 1

    if not counts:
        print("No overlapping gold/prediction pairs found. Did `run` write "
              f"predictions into {args.pred_dir}?")
        return 1

    # Per-attribute F1, then per-vertical mean, then overall mean-of-verticals.
    def _prf(correct, pred_n, gold_n):
        p = correct / pred_n if pred_n else 0.0
        r = correct / gold_n if gold_n else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        return p, r, f1

    per_vertical = {}
    for (vertical, attr), (c, pn, gn) in sorted(counts.items()):
        p, r, f1 = _prf(c, pn, gn)
        per_vertical.setdefault(vertical, []).append((attr, p, r, f1, c, pn, gn))

    lines = []
    lines.append("## SWDE benchmark (page-level attribute F1)")
    lines.append("")
    lines.append("_Setting: per-site zero-label wrapper induction (code "
                 "synthesized from each\ntarget site's own structure, no seed "
                 "sites). Compare to zero-shot / unseen-site\nnumbers "
                 "(e.g. ZeroShotCeres), NOT the supervised k-seed rows._")
    lines.append("")
    lines.append(f"- Sites scored: {scored_sites}"
                 + (f"  (missing predictions for {missing_pred} sites)"
                    if missing_pred else ""))
    lines.append("")
    lines.append("| Vertical | Attribute | P | R | F1 | correct | pred | gold |")
    lines.append("|---|---|---|---|---|---|---|---|")

    vertical_f1s = []
    for vertical in sorted(per_vertical):
        rows = per_vertical[vertical]
        for attr, p, r, f1, c, pn, gn in rows:
            lines.append(f"| {vertical} | {attr} | {p:.3f} | {r:.3f} | "
                         f"{f1:.3f} | {c} | {pn} | {gn} |")
        v_f1 = sum(row[3] for row in rows) / len(rows)
        vertical_f1s.append(v_f1)
        lines.append(f"| **{vertical}** | **(avg)** | | | **{v_f1:.3f}** | "
                     f"| | |")

    overall = sum(vertical_f1s) / len(vertical_f1s) if vertical_f1s else 0.0
    # Micro over all attributes/pages, for reference.
    tot_c = sum(v[0] for v in counts.values())
    tot_pn = sum(v[1] for v in counts.values())
    tot_gn = sum(v[2] for v in counts.values())
    mp, mr, mf1 = _prf(tot_c, tot_pn, tot_gn)
    lines.append("")
    lines.append(f"- **Overall F1 (mean of per-vertical avgs):** {overall:.3f}")
    lines.append(f"- **Micro F1 (pooled):** {mf1:.3f} "
                 f"(P={mp:.3f}, R={mr:.3f})")

    report = "\n".join(lines)
    print("\n" + report + "\n")

    if args.json:
        payload = {
            "overall_f1": overall,
            "micro": {"precision": mp, "recall": mr, "f1": mf1},
            "per_vertical": {
                v: {row[0]: {"precision": row[1], "recall": row[2],
                             "f1": row[3], "correct": row[4],
                             "pred": row[5], "gold": row[6]}
                    for row in rows}
                for v, rows in per_vertical.items()
            },
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote scores JSON: {args.json}")

    if args.append_results:
        with open(args.results_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + report + "\n")
        print(f"Appended SWDE scores to {args.results_path}")
    return 0


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        description="SWDE benchmark harness for the web-scraping agent.")
    sub = parser.add_subparsers(dest="command", required=True)

    common_root = argparse.ArgumentParser(add_help=False)
    common_root.add_argument(
        "--swde-root", required=True,
        help="Folder containing SWDE 'sourceCode' and 'groundtruth'.")
    common_root.add_argument("--source-subdir", default="sourceCode")
    common_root.add_argument("--groundtruth-subdir", default="groundtruth")

    # prepare -------------------------------------------------------------
    pp = sub.add_parser("prepare", parents=[common_root],
                        help="Parse groundtruth -> gold + run manifest.")
    pp.add_argument("--out-dir", default="swde_prepared",
                    help="Where to write gold/ and manifest.json.")
    pp.add_argument("--verticals",
                    help="Comma-separated subset (default: all).")
    pp.set_defaults(func=cmd_prepare)

    # run -----------------------------------------------------------------
    pr = sub.add_parser("run", parents=[common_root],
                        help="Synthesize one extractor per site and reuse it.")
    pr.add_argument("--prepared-dir", default="swde_prepared")
    pr.add_argument("--pred-dir", default="swde_pred")
    pr.add_argument("--api-key", required=True, help="Google AI API key.")
    pr.add_argument("--model", default="gemma-3-27b-it")
    pr.add_argument("--verticals", help="Comma-separated subset (default: all).")
    pr.add_argument("--limit-sites", type=int,
                    help="Max sites per vertical (for a quick pilot run).")
    pr.add_argument("--pages-per-site", type=int,
                    help="Cap scored pages per site (controls time; SWDE has "
                         "~2000/site).")
    pr.add_argument("--repair-retries", type=int, default=2,
                    help="Self-repair attempts on the representative page.")
    pr.add_argument("--max-map-chars", type=int, default=24000,
                    help="Truncate the structural-map JSON in the prompt.")
    pr.add_argument("--overwrite", action="store_true",
                    help="Re-run sites even if a prediction file exists.")
    pr.set_defaults(func=cmd_run)

    # score ---------------------------------------------------------------
    ps = sub.add_parser("score",
                        help="SWDE page-level attribute P/R/F1.")
    ps.add_argument("--prepared-dir", default="swde_prepared")
    ps.add_argument("--pred-dir", default="swde_pred")
    ps.add_argument("--verticals", help="Comma-separated subset (default: all).")
    ps.add_argument("--json", help="Optional path to write a scores JSON.")
    ps.add_argument("--append-results", action="store_true",
                    help="Append the SWDE table to results.md.")
    ps.add_argument("--results-path", default="results.md")
    ps.set_defaults(func=cmd_score)

    return parser


def main():
    args = build_parser().parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
