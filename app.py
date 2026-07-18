"""
Evaluation Flask App
Mirrors the terminal orchestrator + adds a ground-truth evaluation panel.
Run: python app.py
"""

import glob
import json
import os
import re
import difflib
import unicodedata
from flask import Flask, render_template, request, jsonify, session, send_file, abort
from datetime import datetime
from urllib.parse import urlparse

app = Flask(__name__)
app.secret_key = "eval-app-secret-2025"

# ─── Storage ──────────────────────────────────────────────────────────────────
# Scraped predictions + snapshot index live in memory; ground truth is also
# persisted to disk under gold/ so labeling survives restarts.
SCRAPED_RESULTS = {}   # domain -> list of {url, title, data:{field:value}}
GROUND_TRUTH    = {}   # domain -> list of {url, title, fields:{field:value}}
SNAPSHOT_INDEX  = {}   # domain -> {"dir": <snapshot folder name>, "articles": [...]}

WORKSPACE     = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR      = "orch_runs"
SNAPSHOTS_DIR = "snapshots"
GOLD_DIR      = "gold"
os.makedirs(os.path.join(WORKSPACE, GOLD_DIR), exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Disk helpers (safe paths + gold persistence)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_path(base, *parts):
    """Resolve base/parts and refuse anything that escapes *base* (no traversal)."""
    base_real = os.path.realpath(os.path.join(WORKSPACE, base))
    target = os.path.realpath(os.path.join(base_real, *parts))
    if target == base_real or target.startswith(base_real + os.sep):
        return target
    return None


def _gold_path(domain):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", domain).strip("_") or "domain"
    return os.path.join(WORKSPACE, GOLD_DIR, f"{safe}.json")


def _save_gold_to_disk(domain):
    """Persist a domain's ground-truth list to gold/<domain>.json."""
    with open(_gold_path(domain), "w", encoding="utf-8") as f:
        json.dump(GROUND_TRUTH.get(domain, []), f, indent=2, ensure_ascii=False)


def _load_all_gold():
    """Reload every gold/<domain>.json into GROUND_TRUTH on startup."""
    for p in glob.glob(os.path.join(WORKSPACE, GOLD_DIR, "*.json")):
        domain = os.path.splitext(os.path.basename(p))[0]
        try:
            with open(p, encoding="utf-8") as f:
                GROUND_TRUTH[domain] = json.load(f)
        except Exception:
            pass


def _read_json(path, default=None):
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return default


def _run_domain(articles):
    """Best-effort domain label for a run, from the first article's URL."""
    for a in (articles or []):
        url = a.get("url", "") if isinstance(a, dict) else ""
        if url:
            net = urlparse(url).netloc
            if net.startswith("www."):
                net = net[4:]
            if net:
                return net
    return ""


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

        # Strict URL match. An unmatched gold article is scored against empty
        # predictions (a real recall miss) rather than silently borrowing
        # another article's data, which would corrupt the metrics.
        matched   = scraped_by_url.get(gt_url)
        is_matched = matched is not None
        pred_data  = matched.get("data", {}) if is_matched else {}

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
            "url":     gt_url or (matched.get("url", "") if matched else ""),
            "title":   gt_art.get("title") or (matched.get("title", "") if matched else ""),
            "matched": is_matched,
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
    """Save (upsert by URL) a single ground-truth article and persist to disk."""
    global GROUND_TRUTH
    data   = request.get_json()
    domain = data.get("domain", "").strip()
    article = data.get("article", {})   # {url, title, fields:{field:value}}

    if not domain:
        return jsonify({"ok": False, "error": "Domain is required"})

    lst = GROUND_TRUTH.setdefault(domain, [])
    url = (article.get("url") or "").strip()
    replaced = False
    if url:
        for i, a in enumerate(lst):
            if (a.get("url") or "").strip() == url:
                lst[i] = article          # upsert: re-labeling a URL overwrites
                replaced = True
                break
    if not replaced:
        lst.append(article)

    _save_gold_to_disk(domain)
    return jsonify({"ok": True, "total_gt": len(lst), "updated": replaced})


@app.route("/api/clear_ground_truth", methods=["POST"])
def clear_ground_truth():
    global GROUND_TRUTH
    domain = request.get_json().get("domain", "")
    if domain in GROUND_TRUTH:
        GROUND_TRUTH[domain] = []
        _save_gold_to_disk(domain)
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


@app.route("/api/get_ground_truth", methods=["GET"])
def get_ground_truth():
    """Return all persisted ground truth so the UI can hydrate on page load."""
    return jsonify({"ok": True, "ground_truth": GROUND_TRUTH})


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


# ══════════════════════════════════════════════════════════════════════════════
# Disk loading: orchestrator runs + snapshots (no more copy-paste)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/list_runs", methods=["GET"])
def list_runs():
    """List orch_runs/ runs and snapshots/ folders available on disk."""
    runs = []
    for d in sorted(glob.glob(os.path.join(WORKSPACE, RUNS_DIR, "run_*")), reverse=True):
        f = os.path.join(d, "extracted_data_all.json")
        if os.path.exists(f):
            data = _read_json(f, [])
            data = data if isinstance(data, list) else []
            runs.append({
                "name": os.path.basename(d),
                "articles": len(data),
                "website": _run_domain(data),
            })

    snaps = []
    for d in sorted(glob.glob(os.path.join(WORKSPACE, SNAPSHOTS_DIR, "*")), reverse=True):
        meta_f = os.path.join(d, "metadata.json")
        if os.path.isdir(d) and os.path.exists(meta_f):
            m = _read_json(meta_f, {}) or {}
            snaps.append({
                "name": os.path.basename(d),
                "website": m.get("website", ""),
                "articles": (m.get("counts", {}) or {}).get("articles",
                            len(m.get("articles", []))),
                "run_dir": m.get("run_dir", ""),
                "date": m.get("snapshot_date", ""),
            })

    return jsonify({"ok": True, "runs": runs, "snapshots": snaps})


@app.route("/api/load_run", methods=["POST"])
def load_run():
    """Load a run's extracted_data_all.json straight from disk into a domain."""
    data     = request.get_json()
    run_name = os.path.basename((data.get("run_name") or "").rstrip("/"))
    domain   = (data.get("domain") or run_name).strip()

    target = _safe_path(RUNS_DIR, run_name)
    if not target or not os.path.isdir(target):
        return jsonify({"ok": False, "error": "Run not found"})

    articles = _read_json(os.path.join(target, "extracted_data_all.json"), None)
    if not isinstance(articles, list):
        return jsonify({"ok": False, "error": "extracted_data_all.json missing or malformed"})

    SCRAPED_RESULTS[domain] = articles
    return jsonify({"ok": True, "domain": domain, "count": len(articles),
                    "sample_fields": _sample_fields(articles)})


@app.route("/api/load_snapshot", methods=["POST"])
def load_snapshot():
    """Load a snapshot: join predictions from its linked run and index the HTML
    files so the labeling pane can render each article and prefill fields."""
    data      = request.get_json()
    snap_name = os.path.basename((data.get("snapshot_name") or "").rstrip("/"))

    target = _safe_path(SNAPSHOTS_DIR, snap_name)
    if not target or not os.path.isdir(target):
        return jsonify({"ok": False, "error": "Snapshot not found"})

    meta = _read_json(os.path.join(target, "metadata.json"), None)
    if not isinstance(meta, dict):
        return jsonify({"ok": False, "error": "metadata.json missing or malformed"})

    domain = (data.get("domain") or meta.get("website") or snap_name).strip()
    arts_meta = meta.get("articles", [])

    # Join predictions from the linked orchestrator run (by URL).
    preds_by_url = {}
    run_name = os.path.basename((meta.get("run_dir") or "").rstrip("/"))
    if run_name:
        rt = _safe_path(RUNS_DIR, run_name)
        if rt:
            for a in (_read_json(os.path.join(rt, "extracted_data_all.json"), []) or []):
                if isinstance(a, dict):
                    preds_by_url[(a.get("url") or "").strip()] = a

    # SCRAPED_RESULTS drives evaluation; prefer the joined predictions.
    SCRAPED_RESULTS[domain] = list(preds_by_url.values())

    SNAPSHOT_INDEX[domain] = {
        "dir": snap_name,
        "articles": [
            {
                "id": a.get("id"),
                "file": a.get("file"),
                "url": (a.get("url") or "").strip(),
                "title": a.get("title", ""),
                "pred": preds_by_url.get((a.get("url") or "").strip(), {}).get("data", {}),
            }
            for a in arts_meta if a.get("file")
        ],
    }

    return jsonify({
        "ok": True, "domain": domain,
        "count": len(arts_meta), "predicted": len(preds_by_url),
        "sample_fields": _sample_fields(SCRAPED_RESULTS[domain]),
    })


@app.route("/api/snapshot_articles", methods=["POST"])
def snapshot_articles():
    """Return the indexed article list for a domain, flagged with labeled state."""
    domain = request.get_json().get("domain", "")
    idx = SNAPSHOT_INDEX.get(domain)
    if not idx:
        return jsonify({"ok": False, "error": "No snapshot loaded for this domain"})

    labeled = {(a.get("url") or "").strip() for a in GROUND_TRUTH.get(domain, [])}
    return jsonify({
        "ok": True, "dir": idx["dir"],
        "articles": [
            {
                "id": a["id"], "file": a["file"], "url": a["url"],
                "title": a["title"], "pred": a["pred"],
                "labeled": a["url"] in labeled,
            }
            for a in idx["articles"]
        ],
    })


@app.route("/snapshot_html/<snap>/<fname>")
def snapshot_html(snap, fname):
    """Serve a saved article HTML file for the labeling render pane."""
    if not re.match(r"^[\w.\-]+$", snap) or not re.match(r"^[\w.\-]+\.html$", fname):
        abort(404)
    target = _safe_path(SNAPSHOTS_DIR, snap, fname)
    if not target or not os.path.exists(target):
        abort(404)
    return send_file(target, mimetype="text/html")


if __name__ == "__main__":
    _load_all_gold()
    print("🚀 Evaluation App running at http://localhost:5000")
    print(f"   Loaded gold for {len(GROUND_TRUTH)} domain(s) from {GOLD_DIR}/")
    app.run(debug=True, port=5000)