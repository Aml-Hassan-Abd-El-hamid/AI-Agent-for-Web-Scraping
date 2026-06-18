"""
Evaluation Flask App
Mirrors the terminal orchestrator + adds a ground-truth evaluation panel.
Run: python app.py
"""

import json
import os
import re
import difflib
import unicodedata
from flask import Flask, render_template, request, jsonify, session
from datetime import datetime

app = Flask(__name__)
app.secret_key = "eval-app-secret-2025"

# ─── In-memory storage for this session ───────────────────────────────────────
# In production you'd persist to disk; for evaluation purposes memory is fine.
SCRAPED_RESULTS = {}   # domain -> list of {url, title, data:{field:value}}
GROUND_TRUTH    = {}   # domain -> list of {url, fields:{field:value}}


# ══════════════════════════════════════════════════════════════════════════════
# Text normalisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _strip_html(text: str) -> str:
    """Remove HTML tags."""
    return re.sub(r"<[^>]+>", "", text)

def _normalise(text: str) -> str:
    """
    Canonical form for comparison:
    - unicode NFC
    - strip HTML tags
    - collapse all whitespace (including Arabic tatweel, ZWNJ, etc.) to single space
    - strip leading/trailing space
    - lowercase (keeps Arabic as-is; lowercases Latin)
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = unicodedata.normalize("NFC", text)
    text = _strip_html(text)
    # collapse whitespace including \u200c, \u200d, \u00a0, \u0640 (tatweel)
    text = re.sub(r"[\s\u00a0\u200b-\u200f\u0640]+", " ", text)
    return text.strip().lower()


def _field_exact_match(pred: str, gold: str) -> bool:
    return _normalise(pred) == _normalise(gold)


def _field_token_f1(pred: str, gold: str) -> dict:
    """Token-level F1 — same metric used in SQuAD."""
    pred_tokens = _normalise(pred).split()
    gold_tokens = _normalise(gold).split()
    if not pred_tokens and not gold_tokens:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0}
    if not pred_tokens or not gold_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    common = set(pred_tokens) & set(gold_tokens)
    num_common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    precision = num_common / len(pred_tokens)
    recall    = num_common / len(gold_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4)}


def _char_diff(pred: str, gold: str) -> list[dict]:
    """
    Return a list of diff segments showing what differs between
    normalised pred and gold. Each segment: {type: 'equal'|'insert'|'delete'|'replace', text, pred_text, gold_text}
    """
    pred_n = _normalise(pred)
    gold_n = _normalise(gold)
    matcher = difflib.SequenceMatcher(None, gold_n, pred_n, autojunk=False)
    segments = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            segments.append({"type": "equal", "text": gold_n[i1:i2]})
        elif op == "insert":
            segments.append({"type": "insert", "pred_text": pred_n[j1:j2], "gold_text": ""})
        elif op == "delete":
            segments.append({"type": "delete", "pred_text": "", "gold_text": gold_n[i1:i2]})
        elif op == "replace":
            segments.append({"type": "replace",
                              "pred_text": pred_n[j1:j2],
                              "gold_text": gold_n[i1:i2]})
    return segments


# ══════════════════════════════════════════════════════════════════════════════
# Core evaluation logic
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_domain(domain: str, scraped_list: list, ground_truth_list: list) -> dict:
    """
    Match scraped articles to ground-truth articles by URL (or by order if URL not provided),
    then compute per-field and aggregate metrics.
    """
    # Build URL → scraped lookup
    scraped_by_url = {art["url"]: art for art in scraped_list}

    field_stats   = {}   # field -> {tp, fp, fn, exact_matches, total}
    article_evals = []

    for gt_art in ground_truth_list:
        gt_url    = gt_art.get("url", "").strip()
        gt_fields = gt_art.get("fields", {})

        # Match by URL; fall back to first unmatched scraped article
        matched = scraped_by_url.get(gt_url)
        if matched is None and scraped_list:
            matched = scraped_list[0]  # best-effort

        if matched is None:
            article_evals.append({
                "url": gt_url,
                "title": gt_art.get("title", ""),
                "matched": False,
                "fields": {}
            })
            continue

        pred_data = matched.get("data", {})
        field_results = {}

        for field, gold_val in gt_fields.items():
            pred_val = pred_data.get(field, "")
            exact    = _field_exact_match(str(pred_val), str(gold_val))
            scores   = _field_token_f1(str(pred_val), str(gold_val))
            diff     = _char_diff(str(pred_val), str(gold_val))

            field_results[field] = {
                "gold":    str(gold_val),
                "pred":    str(pred_val),
                "exact":   exact,
                "f1":      scores["f1"],
                "precision": scores["precision"],
                "recall":  scores["recall"],
                "diff":    diff,
            }

            # Accumulate for macro average
            if field not in field_stats:
                field_stats[field] = {"exact": 0, "f1_sum": 0.0, "total": 0}
            field_stats[field]["total"] += 1
            if exact:
                field_stats[field]["exact"] += 1
            field_stats[field]["f1_sum"] += scores["f1"]

        article_evals.append({
            "url":     gt_url or matched.get("url", ""),
            "title":   gt_art.get("title") or matched.get("title", ""),
            "matched": True,
            "fields":  field_results,
        })

    # Aggregate per-field and overall
    per_field_summary = {}
    all_f1s = []
    all_exact = []
    for field, stats in field_stats.items():
        n = stats["total"]
        exact_rate = stats["exact"] / n if n else 0
        macro_f1   = stats["f1_sum"] / n if n else 0
        per_field_summary[field] = {
            "exact_match": round(exact_rate, 4),
            "macro_f1":    round(macro_f1, 4),
            "n":           n,
        }
        all_f1s.append(macro_f1)
        all_exact.append(exact_rate)

    overall = {
        "mean_f1":         round(sum(all_f1s) / len(all_f1s), 4) if all_f1s else 0,
        "mean_exact":      round(sum(all_exact) / len(all_exact), 4) if all_exact else 0,
        "articles_evaluated": len(article_evals),
        "articles_matched":   sum(1 for a in article_evals if a["matched"]),
    }

    return {
        "domain":          domain,
        "overall":         overall,
        "per_field":       per_field_summary,
        "article_details": article_evals,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/load_scraped", methods=["POST"])
def load_scraped():
    """Load a scraped JSON file (the extracted_data_all.json from a run)."""
    global SCRAPED_RESULTS
    data = request.get_json()
    domain  = data.get("domain", "").strip()
    content = data.get("json_content", "")  # raw JSON string pasted by user

    if not domain:
        return jsonify({"ok": False, "error": "Domain name is required"})
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        return jsonify({"ok": False, "error": f"Invalid JSON: {e}"})

    # Support both a list directly or wrapped in {"article_links": [...]}
    if isinstance(parsed, list):
        articles = parsed
    elif isinstance(parsed, dict):
        articles = parsed.get("articles", parsed.get("article_links", [parsed]))
    else:
        return jsonify({"ok": False, "error": "Unrecognised JSON shape"})

    SCRAPED_RESULTS[domain] = articles
    return jsonify({"ok": True, "count": len(articles), "sample_fields": _sample_fields(articles)})


def _sample_fields(articles):
    for a in articles:
        data = a.get("data", {})
        if data:
            return list(data.keys())
    return []


@app.route("/api/save_ground_truth", methods=["POST"])
def save_ground_truth():
    """Save a single ground-truth article for a domain."""
    global GROUND_TRUTH
    data   = request.get_json()
    domain = data.get("domain", "").strip()
    article = data.get("article", {})   # {url, title, fields:{field:value}}

    if not domain:
        return jsonify({"ok": False, "error": "Domain is required"})
    if domain not in GROUND_TRUTH:
        GROUND_TRUTH[domain] = []

    GROUND_TRUTH[domain].append(article)
    return jsonify({"ok": True, "total_gt": len(GROUND_TRUTH[domain])})


@app.route("/api/clear_ground_truth", methods=["POST"])
def clear_ground_truth():
    global GROUND_TRUTH
    domain = request.get_json().get("domain", "")
    if domain in GROUND_TRUTH:
        GROUND_TRUTH[domain] = []
    return jsonify({"ok": True})


@app.route("/api/get_domains", methods=["GET"])
def get_domains():
    scraped = list(SCRAPED_RESULTS.keys())
    gt      = list(GROUND_TRUTH.keys())
    all_domains = list(set(scraped + gt))
    return jsonify({
        "domains": all_domains,
        "scraped": scraped,
        "ground_truth": gt,
    })


@app.route("/api/get_scraped_article", methods=["POST"])
def get_scraped_article():
    """Return a scraped article by URL (to auto-fill field names in the GT form)."""
    data   = request.get_json()
    domain = data.get("domain", "")
    url    = data.get("url", "").strip()
    articles = SCRAPED_RESULTS.get(domain, [])

    for a in articles:
        if a.get("url", "").strip() == url:
            return jsonify({"ok": True, "article": a})

    # If no URL match, return first article's field structure
    if articles:
        return jsonify({"ok": True, "article": articles[0], "note": "URL not matched; returning first"})
    return jsonify({"ok": False, "error": "No scraped articles for this domain"})


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """Run evaluation for one or all domains."""
    data   = request.get_json()
    domain = data.get("domain", "all")

    if domain == "all":
        domains_to_eval = list(GROUND_TRUTH.keys())
    else:
        domains_to_eval = [domain]

    results = []
    for d in domains_to_eval:
        scraped = SCRAPED_RESULTS.get(d, [])
        gt      = GROUND_TRUTH.get(d, [])
        if not gt:
            continue
        result = evaluate_domain(d, scraped, gt)
        results.append(result)

    return jsonify({"ok": True, "results": results})


@app.route("/api/export_results", methods=["POST"])
def export_results():
    """Return a JSON export of all evaluation results."""
    data   = request.get_json()
    domain = data.get("domain", "all")
    if domain == "all":
        domains_to_eval = list(GROUND_TRUTH.keys())
    else:
        domains_to_eval = [domain]

    all_results = []
    for d in domains_to_eval:
        scraped = SCRAPED_RESULTS.get(d, [])
        gt      = GROUND_TRUTH.get(d, [])
        if gt:
            all_results.append(evaluate_domain(d, scraped, gt))

    return jsonify({"ok": True, "export": all_results, "timestamp": datetime.now().isoformat()})


if __name__ == "__main__":
    print("🚀 Evaluation App running at http://localhost:5000")
    app.run(debug=True, port=5000)