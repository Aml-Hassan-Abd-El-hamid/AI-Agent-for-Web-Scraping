"""Experiment & analysis harness for the system paper.

This is a single, self-contained script that adds the evaluation scaffolding the
paper needs *without touching the existing pipeline*. It has two subcommands:

  audit     Offline. Scans orchestrator run outputs for "silent" extraction
            failures — clusters/pages that were counted as extracted but whose
            fields are actually empty (N/A). This is the aljadeed-style failure
            where template-matched code returned nothing yet still counted as a
            success. Stdlib-only, so it runs with no API key / network / browser.

  baseline  Online. Runs a direct-LLM extraction baseline (structural map -> JSON
            in one LLM call per article, no code synthesis and no cross-page
            reuse) over the same URL set the pipeline used. Saves predictions in
            the exact format evaluate.py consumes, so you get an apples-to-apples
            F1 comparison plus the cost contrast (1 LLM call/article for the
            baseline vs the pipeline's amortized <0.02 calls/article).

Typical use
-----------
Audit every run and fail if any silent failure is found (good for CI)::

    python paper_experiments.py audit --fail-on-silent

Audit one run and dump a JSON report::

    python paper_experiments.py audit --run orch_runs/run_20260707_190029 \
        --json audit_report.json

Run the direct-LLM baseline over the same URLs a pipeline run used, then score
it against a gold file and append the numbers to results.md::

    python paper_experiments.py baseline \
        --from-run orch_runs/run_20260704_025628 \
        --requirements "title, date, author, article body text" \
        --api-key $env:GOOGLE_API_KEY --model gemma-3-27b-it \
        --out baseline_pred.json --gold gold.json --append-results

The baseline predictions can also be scored later with the existing tool::

    python evaluate.py --pred baseline_pred.json --gold gold.json
"""

import argparse
import glob
import json
import os
import re
import sys

# Keep the module import-light: heavy deps (playwright, google.generativeai,
# bs4, requests) are only needed by `baseline`, so they're imported lazily
# inside that command. `audit` stays pure-stdlib and always runnable.

# Mirror of utils.is_na_value so the audit needs no heavy imports. Kept in sync
# with the shared predicate the orchestrator and evaluator use.
_NA_TOKENS = {"", "n/a", "na", "none", "null", "-", "--"}


def _is_na(v):
    """True if a field value is effectively empty / not-available."""
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().lower() in _NA_TOKENS
    if isinstance(v, (list, dict, tuple)):
        return len(v) == 0
    return False


# ════════════════════════════════════════════════════════════════════
# audit — detect silent (counted-but-empty) extraction failures
# ════════════════════════════════════════════════════════════════════

def _load_extracted(path):
    """Load an orchestrator extracted_data_all.json into [{url,title,data}]."""
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    records = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            data = item.get("data")
            if not isinstance(data, dict):
                data = {k: v for k, v in item.items()
                        if k not in ("url", "title", "reason")}
            records.append({
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "data": data,
            })
    elif isinstance(raw, dict):
        for url, data in raw.items():
            records.append({
                "url": url,
                "title": "",
                "data": data if isinstance(data, dict) else {},
            })
    return records


def _audit_run(run_dir, na_field_threshold, min_articles):
    """Audit one run directory. Returns a stats dict (or None if no data)."""
    pred_path = os.path.join(run_dir, "extracted_data_all.json")
    if not os.path.isfile(pred_path):
        return None
    records = _load_extracted(pred_path)
    n = len(records)
    if n == 0:
        return {
            "run_dir": run_dir, "n_records": 0, "fields": {},
            "fully_empty_records": 0, "silent_fields": [], "flagged": False,
        }

    # Collect the union of field names in insertion order.
    fields = []
    seen = set()
    for r in records:
        for k in r["data"]:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    field_stats = {}
    for field in fields:
        na = sum(1 for r in records if _is_na(r["data"].get(field)))
        field_stats[field] = {
            "na": na,
            "total": n,
            "na_rate": na / n if n else 0.0,
        }

    fully_empty = sum(
        1 for r in records
        if r["data"] and all(_is_na(v) for v in r["data"].values())
    )

    # A "silent field failure" = a field that is almost always empty on a run
    # that has enough articles AND where at least one other field is mostly
    # populated. That combination means the template matched (so it counted as
    # a success) but this specific field extracted nothing across the board —
    # exactly the aljadeed date/author/body case.
    populated_exists = any(
        s["na_rate"] < na_field_threshold for s in field_stats.values()
    )
    silent_fields = []
    if n >= min_articles and populated_exists:
        for field, s in field_stats.items():
            if s["na_rate"] >= na_field_threshold:
                silent_fields.append(field)

    flagged = bool(silent_fields) or (n >= min_articles and fully_empty == n)

    return {
        "run_dir": run_dir,
        "n_records": n,
        "fields": field_stats,
        "fully_empty_records": fully_empty,
        "silent_fields": silent_fields,
        "flagged": flagged,
    }


def _print_audit(stats_list, na_field_threshold):
    pct = int(na_field_threshold * 100)
    print(f"\n{'=' * 72}")
    print("SILENT-FAILURE AUDIT")
    print(f"  A field is flagged when it is N/A for >= {pct}% of a run's "
          f"articles\n  while at least one other field is populated "
          f"(template matched but\n  extracted nothing — counted as success).")
    print(f"{'=' * 72}")

    flagged_any = False
    for s in stats_list:
        if s is None:
            continue
        run = os.path.basename(s["run_dir"].rstrip("/\\"))
        if s["n_records"] == 0:
            print(f"\n• {run}: no extracted records.")
            continue

        marker = "⚠️  FLAGGED" if s["flagged"] else "ok"
        print(f"\n• {run}  ({s['n_records']} articles)  [{marker}]")
        for field, fs in s["fields"].items():
            bad = fs["na_rate"] >= na_field_threshold
            tag = "  <-- silent failure" if field in s["silent_fields"] else ""
            flag = "!" if bad else " "
            print(f"    {flag} {field:<24} "
                  f"N/A {fs['na']}/{fs['total']} ({fs['na_rate'] * 100:.0f}%)"
                  f"{tag}")
        if s["fully_empty_records"]:
            print(f"      fully-empty records (all fields N/A): "
                  f"{s['fully_empty_records']}/{s['n_records']}")
        if s["flagged"]:
            flagged_any = True

    print(f"\n{'=' * 72}")
    if flagged_any:
        print("RESULT: silent failures detected. These runs report articles as")
        print("        'extracted' while key fields are empty — do not use their")
        print("        counts as success metrics in the paper without a caveat.")
    else:
        print("RESULT: no silent failures detected under current thresholds.")
    print(f"{'=' * 72}\n")
    return flagged_any


def cmd_audit(args):
    if args.run:
        run_dirs = [args.run]
    else:
        run_dirs = sorted(
            d for d in glob.glob(os.path.join(args.runs_root, "*"))
            if os.path.isdir(d)
        )
    if not run_dirs:
        print(f"No run directories found under {args.runs_root!r}.")
        return 0

    stats_list = [
        _audit_run(d, args.na_field_threshold, args.min_articles)
        for d in run_dirs
    ]
    flagged_any = _print_audit(stats_list, args.na_field_threshold)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([s for s in stats_list if s], f,
                      indent=2, ensure_ascii=False)
        print(f"Wrote audit report: {args.json}")

    if args.fail_on_silent and flagged_any:
        return 1
    return 0


# ════════════════════════════════════════════════════════════════════
# baseline — direct-LLM extraction (no code synthesis, no reuse)
# ════════════════════════════════════════════════════════════════════

def _fields_from_requirements(requirements):
    """Split a free-text requirements string into individual field names.

    'title, date, author, article body text' -> ['title','date','author',
    'article body text'] — matching how the pipeline names output fields.
    """
    parts = re.split(r"[,;\n]+", requirements)
    return [p.strip() for p in parts if p.strip()]


def _urls_for_baseline(args):
    """Resolve the URL set for the baseline from the mutually-exclusive inputs."""
    urls = []
    if args.from_run:
        pred_path = os.path.join(args.from_run, "extracted_data_all.json")
        for r in _load_extracted(pred_path):
            if r["url"]:
                urls.append(r["url"])
    if args.urls_file:
        with open(args.urls_file, "r", encoding="utf-8") as f:
            for line in f:
                u = line.strip()
                if u and not u.startswith("#"):
                    urls.append(u)
    if args.url:
        urls.extend(args.url)

    # De-dup, preserve order.
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    if args.limit:
        deduped = deduped[: args.limit]
    return deduped


def _build_direct_prompt(structural_map_json, requirements, fields, page_url):
    """Prompt that asks the LLM to emit JSON values directly (no code)."""
    field_list = ", ".join(f'"{f}"' for f in fields)
    return f"""<start_of_turn>user
You are a web-data extraction expert. You are given an HTML structural map (JSON)
for a single web page, a target URL, and the fields to extract. Return ONLY a
single JSON object with exactly these keys: {field_list}.

Rules:
- Extract the value for each field directly from the structural map's text.
- Use the string "N/A" for a field only if it genuinely does not appear.
- Do NOT write code. Do NOT explain. Output ONLY the JSON object.

USER REQUIREMENTS: {requirements}
TARGET URL: {page_url}

STRUCTURAL MAP:
{structural_map_json}

Output the JSON object now.
<end_of_turn>
<start_of_turn>model
"""


def _parse_json_object(text, fields):
    """Best-effort parse of a JSON object from an LLM response."""
    if not text:
        return {f: "N/A" for f in fields}
    # Prefer a fenced ```json block if present.
    fenced = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    candidate = fenced[-1] if fenced else text
    # Grab the outermost {...} span.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start:end + 1]
    try:
        obj = json.loads(candidate)
    except Exception:
        return {f: "N/A" for f in fields}
    if not isinstance(obj, dict):
        return {f: "N/A" for f in fields}
    # Normalize to the requested field set.
    return {f: obj.get(f, "N/A") for f in fields}


def cmd_baseline(args):
    # Lazy heavy imports so `audit` never pays for them.
    import asyncio
    import google.generativeai as genai
    from utils import (
        _generate_with_retry, get_token_usage, reset_token_usage,
    )
    from Agent_for_single_page_gemma import fetch_page_structure

    urls = _urls_for_baseline(args)
    if not urls:
        print("No URLs to process. Provide --from-run, --urls-file, or --url.")
        return 1
    fields = _fields_from_requirements(args.requirements)
    if not fields:
        print("Could not parse any field names from --requirements.")
        return 1

    print(f"Direct-LLM baseline: {len(urls)} URLs, fields={fields}, "
          f"model={args.model}")

    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel(args.model)
    reset_token_usage()

    async def run():
        predictions = []
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] {url[:80]}")
            try:
                _html, smap = await fetch_page_structure(url)
            except Exception as e:
                print(f"    fetch failed: {e}")
                predictions.append({"url": url, "title": "",
                                    "data": {f: "N/A" for f in fields}})
                continue
            if not smap:
                print("    empty structural map")
                predictions.append({"url": url, "title": "",
                                    "data": {f: "N/A" for f in fields}})
                continue

            smap_json = json.dumps(smap, ensure_ascii=False)
            if len(smap_json) > args.max_map_chars:
                smap_json = smap_json[: args.max_map_chars]
            prompt = _build_direct_prompt(
                smap_json, args.requirements, fields, url)
            try:
                resp = _generate_with_retry(model, prompt)
                data = _parse_json_object(getattr(resp, "text", ""), fields)
            except Exception as e:
                print(f"    LLM failed: {e}")
                data = {f: "N/A" for f in fields}
            predictions.append({
                "url": url,
                "title": data.get("title", "") if "title" in fields else "",
                "data": data,
            })
        return predictions

    predictions = asyncio.run(run())

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\nWrote baseline predictions: {args.out}  ({len(predictions)} records)")

    # Cost / call contrast for the paper.
    usage = get_token_usage()
    calls = usage.get("calls", 0)
    total_tokens = usage.get("total", 0)
    per_article = (calls / len(predictions)) if predictions else 0.0
    est_cost = (usage.get("prompt", 0) / 1e6) * args.price_in \
        + (usage.get("candidates", 0) / 1e6) * args.price_out
    print("\n-- Baseline cost profile (contrast with pipeline reuse) --")
    print(f"  LLM calls: {calls}  ({per_article:.2f} calls/article)")
    print(f"  Tokens: {total_tokens:,} "
          f"({usage.get('prompt', 0):,} in + {usage.get('candidates', 0):,} out)")
    print(f"  Estimated cost: ${est_cost:.4f} "
          f"(@ ${args.price_in}/${args.price_out} per 1M in/out)")

    # Optional scoring against a gold file, reusing the existing evaluator.
    if args.gold:
        from evaluate import evaluate as _evaluate, _format_report
        results = _evaluate(args.out, args.gold, f1_threshold=args.f1_threshold)
        report = _format_report(results)
        # Retitle so it's clearly the baseline in results.md.
        report = report.replace("## Evaluation (gold-set)",
                                "## Evaluation (gold-set) — Direct-LLM baseline")
        print("\n" + report)
        if args.append_results:
            with open(args.results_path, "a", encoding="utf-8") as f:
                f.write("\n\n" + report + "\n")
            print(f"\nAppended baseline evaluation to {args.results_path}")

    return 0


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        description="Experiment & analysis harness for the system paper.")
    sub = parser.add_subparsers(dest="command", required=True)

    # audit ----------------------------------------------------------------
    pa = sub.add_parser(
        "audit",
        help="Scan run outputs for silent (counted-but-empty) failures.")
    pa.add_argument("--run", help="A single run directory to audit.")
    pa.add_argument("--runs-root", default="orch_runs",
                    help="Root folder of run directories (default: orch_runs).")
    pa.add_argument("--na-field-threshold", type=float, default=0.9,
                    help="Field N/A-rate at/above which it's a silent failure "
                         "(default: 0.9).")
    pa.add_argument("--min-articles", type=int, default=5,
                    help="Only flag runs with at least this many articles "
                         "(default: 5).")
    pa.add_argument("--json", help="Optional path to write a JSON report.")
    pa.add_argument("--fail-on-silent", action="store_true",
                    help="Exit non-zero if any silent failure is detected.")
    pa.set_defaults(func=cmd_audit)

    # baseline -------------------------------------------------------------
    pb = sub.add_parser(
        "baseline",
        help="Run the direct-LLM extraction baseline (1 call/article, no reuse).")
    src = pb.add_argument_group("URL source (combine as needed)")
    src.add_argument("--from-run",
                     help="Reuse the URL set from this run's "
                          "extracted_data_all.json.")
    src.add_argument("--urls-file", help="Text file with one URL per line.")
    src.add_argument("--url", action="append",
                     help="A URL to process (repeatable).")
    pb.add_argument("--limit", type=int,
                    help="Process at most this many URLs.")
    pb.add_argument("--requirements", required=True,
                    help="Comma-separated fields, e.g. "
                         "'title, date, author, article body text'.")
    pb.add_argument("--api-key", required=True, help="Google AI API key.")
    pb.add_argument("--model", default="gemma-3-27b-it",
                    help="Model name (default: gemma-3-27b-it).")
    pb.add_argument("--out", default="baseline_pred.json",
                    help="Where to write predictions (default: baseline_pred.json).")
    pb.add_argument("--max-map-chars", type=int, default=24000,
                    help="Truncate the structural map JSON to this many chars.")
    pb.add_argument("--gold", help="Optional gold file to score against.")
    pb.add_argument("--f1-threshold", type=float, default=0.6,
                    help="Long-field token-F1 cutoff (default: 0.6).")
    pb.add_argument("--append-results", action="store_true",
                    help="Append the baseline evaluation to results.md.")
    pb.add_argument("--results-path", default="results.md",
                    help="results.md path for --append-results.")
    pb.add_argument("--price-in", type=float, default=0.075,
                    help="USD per 1M input tokens (paper figure).")
    pb.add_argument("--price-out", type=float, default=0.30,
                    help="USD per 1M output tokens (paper figure).")
    pb.set_defaults(func=cmd_baseline)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
