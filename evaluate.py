"""Gold-set evaluation harness for the web-scraping agent.

Compares the orchestrator's `extracted_data_all.json` against a hand-labeled
gold file and reports per-field precision / recall / F1 plus overall macro and
micro scores. This is the accuracy metric the paper needs alongside the runtime
/ cost stats that the orchestrator already emits into results.md.

Standard-library only (imports just `is_na_value` from utils) so it stays fast
and does not pull in google.generativeai.

Typical workflow
----------------
1. Run a scrape so a run directory contains `extracted_data_all.json`.
2. Make a blank gold template from it and fill in the true values by hand::

       python evaluate.py --make-template orch_runs/<run>/extracted_data_all.json \
           --out gold.json --prefill

   (`--prefill` copies the predicted values in so you only correct mistakes.)
3. Score a prediction file against the gold set::

       python evaluate.py --pred orch_runs/<run>/extracted_data_all.json \
           --gold gold.json

   Add `--append-results` to write an Evaluation section into results.md.

Metrics
-------
* Short fields (title / date / author / …): normalized exact match
  (lowercase, strip, collapse internal whitespace).
* Long fields (body / content / text / …): SQuAD-style token-level F1. A
  prediction counts as "correct" when its token-F1 >= --f1-threshold
  (default 0.6); the mean token-F1 is also reported.
* Per field: precision = correct / predicted-non-empty,
  recall = correct / gold-non-empty, F1 = harmonic mean.
* Overall: macro (mean of per-field F1) and micro (pooled counts).
"""

import argparse
import json
import os
import re
import sys

# Import the shared N/A predicate the same way the agents do, so "empty"
# means the same thing here as it does in the orchestrator's N/A audit.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import is_na_value  # noqa: E402


# Fields treated as long free-text and scored with token-level F1 instead of
# exact match. Matched case-insensitively against the field name.
_LONG_FIELDS = {"body", "content", "text", "article", "description", "summary"}


def _is_long_field(name):
    return name.lower() in _LONG_FIELDS


def _norm_text(value):
    """Lowercase, strip, and collapse internal whitespace for comparison."""
    if value is None:
        return ""
    s = str(value).strip().lower()
    return re.sub(r"\s+", " ", s)


def _tokens(value):
    """Word tokens for SQuAD-style F1 (alphanumeric runs, lowercased)."""
    return re.findall(r"\w+", _norm_text(value))


def _token_f1(pred, gold):
    """SQuAD-style token-level F1 between two strings (0.0–1.0)."""
    pred_toks = _tokens(pred)
    gold_toks = _tokens(gold)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0

    # Multiset intersection (count shared tokens up to their min frequency).
    common = 0
    gold_pool = {}
    for t in gold_toks:
        gold_pool[t] = gold_pool.get(t, 0) + 1
    for t in pred_toks:
        if gold_pool.get(t, 0) > 0:
            common += 1
            gold_pool[t] -= 1
    if common == 0:
        return 0.0

    precision = common / len(pred_toks)
    recall = common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def _load_records(path):
    """Load a scrape output file and normalize to {url: {field: value}}.

    Accepts either the orchestrator's list form
    ``[{"url", "title", "data": {...}}, ...]`` or an already-normalized
    ``{url: {field: value}}`` mapping (as produced by --make-template).
    """
    # utf-8-sig tolerates a BOM that some editors add to hand-labeled gold files.
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    normalized = {}
    if isinstance(raw, dict):
        # Already {url: {field: value}}.
        for url, data in raw.items():
            normalized[url] = data if isinstance(data, dict) else {}
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not url:
                continue
            data = item.get("data")
            if not isinstance(data, dict):
                # Fall back to the item itself minus bookkeeping keys.
                data = {k: v for k, v in item.items()
                        if k not in ("url", "reason")}
            normalized[url] = data
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")
    return normalized


def make_template(pred_path, out_path, prefill=False):
    """Write a blank (or prefilled) gold template from a prediction file."""
    pred = _load_records(pred_path)
    template = {}
    for url, data in pred.items():
        template[url] = {
            field: (value if prefill else "")
            for field, value in data.items()
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    total_fields = sum(len(v) for v in template.values())
    mode = "prefilled" if prefill else "blank"
    print(f"Wrote {mode} gold template: {out_path}")
    print(f"  {len(template)} URLs, {total_fields} fields to verify.")


def evaluate(pred_path, gold_path, f1_threshold=0.6):
    """Score predictions against gold. Returns a results dict."""
    pred = _load_records(pred_path)
    gold = _load_records(gold_path)

    # Union of every field seen in the gold set (gold defines what matters).
    fields = []
    seen = set()
    for data in gold.values():
        for field in data:
            if field not in seen:
                seen.add(field)
                fields.append(field)

    per_field = {}
    for field in fields:
        long_field = _is_long_field(field)
        correct = 0
        pred_nonempty = 0
        gold_nonempty = 0
        f1_sum = 0.0
        f1_count = 0

        for url, gdata in gold.items():
            gval = gdata.get(field)
            pval = (pred.get(url) or {}).get(field)
            g_has = not is_na_value(gval)
            p_has = not is_na_value(pval)
            if g_has:
                gold_nonempty += 1
            if p_has:
                pred_nonempty += 1

            if long_field:
                # Only score token-F1 where the gold has a reference answer.
                if g_has:
                    tf1 = _token_f1(pval if p_has else "", gval)
                    f1_sum += tf1
                    f1_count += 1
                    if tf1 >= f1_threshold:
                        correct += 1
            else:
                if g_has and p_has and _norm_text(pval) == _norm_text(gval):
                    correct += 1

        precision = correct / pred_nonempty if pred_nonempty else 0.0
        recall = correct / gold_nonempty if gold_nonempty else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        entry = {
            "type": "long" if long_field else "short",
            "correct": correct,
            "pred_nonempty": pred_nonempty,
            "gold_nonempty": gold_nonempty,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if long_field and f1_count:
            entry["mean_token_f1"] = f1_sum / f1_count
        per_field[field] = entry

    # Overall macro (mean of per-field F1) and micro (pooled counts).
    scored = [e for e in per_field.values() if e["gold_nonempty"] > 0]
    macro_f1 = (sum(e["f1"] for e in scored) / len(scored)) if scored else 0.0
    tot_correct = sum(e["correct"] for e in scored)
    tot_pred = sum(e["pred_nonempty"] for e in scored)
    tot_gold = sum(e["gold_nonempty"] for e in scored)
    micro_p = tot_correct / tot_pred if tot_pred else 0.0
    micro_r = tot_correct / tot_gold if tot_gold else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) else 0.0)

    return {
        "n_urls": len(gold),
        "f1_threshold": f1_threshold,
        "per_field": per_field,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
    }


def _format_report(results):
    """Render evaluation results as readable markdown lines."""
    lines = []
    lines.append("## Evaluation (gold-set)")
    lines.append("")
    lines.append(f"- **Gold URLs:** {results['n_urls']}")
    lines.append(f"- **Long-field F1 threshold:** {results['f1_threshold']}")
    lines.append(
        f"- **Macro F1:** {results['macro_f1']:.3f}  |  "
        f"**Micro F1:** {results['micro_f1']:.3f} "
        f"(P={results['micro_precision']:.3f}, "
        f"R={results['micro_recall']:.3f})"
    )
    lines.append("")
    lines.append("| Field | Type | Precision | Recall | F1 | Correct | Pred | Gold | Mean tok-F1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for field, e in results["per_field"].items():
        mean_tf1 = (f"{e['mean_token_f1']:.3f}"
                    if "mean_token_f1" in e else "—")
        lines.append(
            f"| {field} | {e['type']} | {e['precision']:.3f} | "
            f"{e['recall']:.3f} | {e['f1']:.3f} | {e['correct']} | "
            f"{e['pred_nonempty']} | {e['gold_nonempty']} | {mean_tf1} |"
        )
    return "\n".join(lines)


def _append_to_results(report_text, path="results.md"):
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n\n" + report_text + "\n")
    print(f"Appended evaluation section to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Gold-set F1 evaluation for the web-scraping agent.")
    parser.add_argument("--pred", help="Prediction file (extracted_data_all.json).")
    parser.add_argument("--gold", help="Hand-labeled gold file.")
    parser.add_argument("--make-template", dest="make_template",
                        help="Build a gold template from this prediction file.")
    parser.add_argument("--out", help="Output path for --make-template.")
    parser.add_argument("--prefill", action="store_true",
                        help="Prefill the template with predicted values.")
    parser.add_argument("--f1-threshold", type=float, default=0.6,
                        help="Token-F1 cutoff for a long field to count as correct.")
    parser.add_argument("--append-results", action="store_true",
                        help="Append the report as an Evaluation section in results.md.")
    parser.add_argument("--results-path", default="results.md",
                        help="results.md path for --append-results.")
    args = parser.parse_args()

    if args.make_template:
        if not args.out:
            parser.error("--make-template requires --out")
        make_template(args.make_template, args.out, prefill=args.prefill)
        return

    if not args.pred or not args.gold:
        parser.error("Provide --pred and --gold (or use --make-template).")

    results = evaluate(args.pred, args.gold, f1_threshold=args.f1_threshold)
    report = _format_report(results)
    print(report)

    if args.append_results:
        _append_to_results(report, path=args.results_path)


if __name__ == "__main__":
    main()
