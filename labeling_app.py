"""Flask labeling app for snapshot-based paper evaluation.

Run:
    python labeling_app.py

The app is intentionally separate from app.py so the existing evaluation dashboard
keeps working unchanged. It supports:
- launching orch_pag_snapshot.py from a regular web form
- browsing saved snapshots
- sampling random snapshot articles for manual checking
- saving per-article labels and corrected fields
- comparing pasted manual text against system predictions
- updating detailed_evaluation_table.md with checked article IDs
- writing manual_labeling_table.md with one row per labeling event
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import difflib
import subprocess
import sys
import threading
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

from utils import list_available_models


WORKSPACE = Path(__file__).resolve().parent
SNAPSHOTS_DIR = WORKSPACE / "snapshots"
RUNS_DIR = WORKSPACE / "orch_runs"
LABELS_DIR = WORKSPACE / "labels"
LOGS_DIR = LABELS_DIR / "run_logs"
DETAILED_EVAL_TABLE = WORKSPACE / "detailed_evaluation_table.md"
MANUAL_LABEL_TABLE = WORKSPACE / "manual_labeling_table.md"

LABELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = "snapshot-labeling-app"

JOBS: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Disk helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open(encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def _safe_snapshot_path(snapshot_name: str) -> Path | None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", snapshot_name or ""):
        return None
    path = (SNAPSHOTS_DIR / snapshot_name).resolve()
    try:
        path.relative_to(SNAPSHOTS_DIR.resolve())
    except ValueError:
        return None
    if not path.is_dir():
        return None
    return path


def _safe_snapshot_file(snapshot_name: str, filename: str) -> Path | None:
    snap_path = _safe_snapshot_path(snapshot_name)
    if not snap_path or not re.fullmatch(r"[A-Za-z0-9_.-]+\.html", filename or ""):
        return None
    path = (snap_path / filename).resolve()
    try:
        path.relative_to(snap_path.resolve())
    except ValueError:
        return None
    if not path.exists():
        return None
    return path


def _label_state_path(snapshot_name: str) -> Path:
    return LABELS_DIR / f"{_safe_name(snapshot_name)}_labels.json"


def _label_events_path(snapshot_name: str) -> Path:
    return LABELS_DIR / f"{_safe_name(snapshot_name)}_events.jsonl"


def _checked_links_path(snapshot_name: str) -> Path:
    return LABELS_DIR / f"{_safe_name(snapshot_name)}_checked_links.md"


def _failure_notes_path(snapshot_name: str) -> Path:
    return LABELS_DIR / f"{_safe_name(snapshot_name)}_failure_notes.json"


def _na_notes_path(snapshot_name: str) -> Path:
    return LABELS_DIR / f"{_safe_name(snapshot_name)}_na_notes.json"


def _snapshot_label_dir(snapshot_name: str, kind: str) -> Path:
    """Directory that holds per-labeller JSONL files for one snapshot/kind.

    kind is one of "articles", "failures", "na". Each labeller writes only to
    their own <labeler>.jsonl file, so git can merge across labellers without
    conflicts and no labeller ever rewrites a file owned by someone else.
    """
    return LABELS_DIR / _safe_name(snapshot_name) / kind


def _labeler_jsonl_path(snapshot_name: str, kind: str, labeler: str) -> Path:
    directory = _snapshot_label_dir(snapshot_name, kind)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_safe_name(labeler)}.jsonl"


def _iter_jsonl_records(directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not directory.is_dir():
        return records
    for path in sorted(directory.glob("*.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if isinstance(rec, dict):
                    records.append(rec)
        except Exception:
            continue
    records.sort(key=lambda r: str(r.get("timestamp", "")))
    return records


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_labels(snapshot_name: str) -> dict[str, Any]:
    """Merge every labeller's article records; latest write per article wins."""
    result: dict[str, Any] = {}
    for rec in _iter_jsonl_records(_snapshot_label_dir(snapshot_name, "articles")):
        article_id = rec.get("article_id")
        if article_id is None:
            continue
        result[str(article_id)] = rec
    if result:
        return result
    legacy = _read_json(_label_state_path(snapshot_name), {})
    return legacy if isinstance(legacy, dict) else {}


def _load_keyed_notes(snapshot_name: str, kind: str) -> dict[str, Any]:
    """Merge every labeller's notes for one kind; latest write per key wins."""
    notes: dict[str, Any] = {}
    for rec in _iter_jsonl_records(_snapshot_label_dir(snapshot_name, kind)):
        key = rec.get("key")
        if key:
            notes[str(key)] = rec
    return notes


def _record_label(snapshot_name: str, labeler: str, label: dict[str, Any]) -> None:
    _append_jsonl(_labeler_jsonl_path(snapshot_name, "articles", labeler), label)


def _record_note(snapshot_name: str, kind: str, labeler: str, note: dict[str, Any]) -> None:
    _append_jsonl(_labeler_jsonl_path(snapshot_name, kind, labeler), note)


def _load_snapshot_meta(snapshot_name: str) -> dict[str, Any] | None:
    snap_path = _safe_snapshot_path(snapshot_name)
    if not snap_path:
        return None
    meta = _read_json(snap_path / "metadata.json")
    return meta if isinstance(meta, dict) else None


def _load_predictions(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    run_dir = Path(str(meta.get("run_dir", ""))).name
    if not run_dir:
        return {}
    extracted_path = RUNS_DIR / run_dir / "extracted_data_all.json"
    extracted = _read_json(extracted_path, [])
    if not isinstance(extracted, list):
        return {}
    by_url = {}
    for item in extracted:
        if isinstance(item, dict):
            url = str(item.get("url", "")).strip()
            if url:
                by_url[url] = item
    return by_url


def _linked_run_path(meta: dict[str, Any]) -> Path | None:
    run_dir = Path(str(meta.get("run_dir", ""))).name
    if not run_dir:
        return None
    path = (RUNS_DIR / run_dir).resolve()
    try:
        path.relative_to(RUNS_DIR.resolve())
    except ValueError:
        return None
    return path if path.is_dir() else None


def _load_run_list(meta: dict[str, Any], filename: str) -> list[dict[str, Any]]:
    run_path = _linked_run_path(meta)
    if not run_path:
        return []
    data = _read_json(run_path / filename, [])
    return data if isinstance(data, list) else []


def _is_na_value(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    text = _normalise_text(value)
    return text in {"", "n/a", "na", "none", "null", "not available", "غير متوفر"}


def _na_field_stats(extracted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, int]] = {}
    for article in extracted:
        data = article.get("data", {}) if isinstance(article, dict) else {}
        if not isinstance(data, dict):
            continue
        for field, value in data.items():
            stats = counts.setdefault(str(field), {"total": 0, "na": 0})
            stats["total"] += 1
            if _is_na_value(value):
                stats["na"] += 1
    rows = []
    for field, stats in sorted(counts.items()):
        total = stats["total"]
        na = stats["na"]
        rows.append({"field": field, "total": total, "na": na, "pct": round((na / total * 100), 1) if total else 0})
    return rows


def _sample_na_articles(extracted: list[dict[str, Any]], field: str, sample_size: int, seed: str) -> list[dict[str, Any]]:
    matches = []
    for index, article in enumerate(extracted, 1):
        data = article.get("data", {}) if isinstance(article, dict) else {}
        if isinstance(data, dict) and field in data and _is_na_value(data.get(field)):
            item = dict(article)
            item["sample_id"] = index
            matches.append(item)
    rng = random.Random(seed or datetime.now().isoformat())
    rng.shuffle(matches)
    return matches[: max(0, sample_size)]


def _snapshot_cards() -> list[dict[str, Any]]:
    cards = []
    for meta_path in sorted(SNAPSHOTS_DIR.glob("*/metadata.json"), reverse=True):
        snapshot_name = meta_path.parent.name
        meta = _read_json(meta_path, {}) or {}
        labels = _load_labels(snapshot_name)
        counts = meta.get("counts", {}) if isinstance(meta.get("counts"), dict) else {}
        cards.append({
            "name": snapshot_name,
            "website": meta.get("website", snapshot_name),
            "date": meta.get("snapshot_date", ""),
            "pagination_type": meta.get("pagination_type", ""),
            "run_dir": meta.get("run_dir", ""),
            "articles": counts.get("articles", len(meta.get("articles", []) or [])),
            "checked": len(labels),
        })
    return cards


def _article_by_id(meta: dict[str, Any], article_id: int) -> dict[str, Any] | None:
    for article in meta.get("articles", []) or []:
        if int(article.get("id", -1)) == article_id:
            return article
    return None


def _sample_articles(meta: dict[str, Any], labels: dict[str, Any], sample_size: int, seed: str, include_checked: bool) -> list[dict[str, Any]]:
    articles = list(meta.get("articles", []) or [])
    if not include_checked:
        articles = [a for a in articles if str(a.get("id")) not in labels]
    rng = random.Random(seed or datetime.now().isoformat())
    rng.shuffle(articles)
    return articles[: max(0, sample_size)]


def _md_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ").replace("|", "\\|")
    return text.strip()


def _label_summary(labels: dict[str, Any]) -> dict[str, Any]:
    decisions: dict[str, int] = {}
    for label in labels.values():
        decision = label.get("decision", "unknown")
        decisions[decision] = decisions.get(decision, 0) + 1
    checked_ids = sorted(int(k) for k in labels.keys() if str(k).isdigit())
    return {"count": len(labels), "ids": checked_ids, "decisions": decisions}


def _normalise_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\s\u00a0\u200b-\u200f\u0640]+", " ", text)
    return text.strip().lower()


def _token_scores(predicted: str, manual: str) -> dict[str, float]:
    pred_tokens = _normalise_text(predicted).split()
    manual_tokens = _normalise_text(manual).split()
    if not pred_tokens and not manual_tokens:
        return {"exact": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not manual_tokens:
        return {"exact": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    common = set(pred_tokens) & set(manual_tokens)
    overlap = sum(min(pred_tokens.count(token), manual_tokens.count(token)) for token in common)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(manual_tokens)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = 1.0 if _normalise_text(predicted) == _normalise_text(manual) else 0.0
    return {
        "exact": round(exact, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _diff_segments(predicted: str, manual: str) -> list[dict[str, str]]:
    pred = _normalise_text(predicted)
    gold = _normalise_text(manual)
    matcher = difflib.SequenceMatcher(None, pred, gold, autojunk=False)
    segments = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        segments.append({
            "type": op,
            "predicted": pred[i1:i2],
            "manual": gold[j1:j2],
        })
    return segments


def _pick_pred_field(pred_data: dict[str, Any], candidates: tuple[str, ...]) -> str:
    for key, value in pred_data.items():
        key_l = str(key).lower()
        if any(candidate.lower() == key_l for candidate in candidates):
            return "" if value is None else str(value)
    for key, value in pred_data.items():
        key_l = str(key).lower()
        if any(candidate.lower() in key_l for candidate in candidates):
            return "" if value is None else str(value)
    return ""


def _write_checked_links(snapshot_name: str, meta: dict[str, Any], labels: dict[str, Any]) -> Path:
    articles_by_id = {str(a.get("id")): a for a in meta.get("articles", []) or []}
    lines = [
        f"# Checked Links: {snapshot_name}",
        "",
        f"Website: {meta.get('website', '')}",
        f"Snapshot date: {meta.get('snapshot_date', '')}",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Article ID | Page | Decision | Labeler | URL |",
        "|---|---|---|---|---|",
    ]
    for article_id in sorted(labels, key=lambda x: int(x) if str(x).isdigit() else 10**9):
        label = labels[article_id]
        article = articles_by_id.get(str(article_id), {})
        lines.append(
            "| " + " | ".join([
                _md_cell(article_id),
                _md_cell(article.get("page_num", "")),
                _md_cell(label.get("decision", "")),
                _md_cell(label.get("labeler", "")),
                _md_cell(article.get("url", label.get("url", ""))),
            ]) + " |"
        )
    path = _checked_links_path(snapshot_name)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _append_manual_label_table(snapshot_name: str, meta: dict[str, Any], article: dict[str, Any], label: dict[str, Any]) -> None:
    header_needed = not MANUAL_LABEL_TABLE.exists()
    if header_needed:
        MANUAL_LABEL_TABLE.write_text(
            "# Manual Labeling Table\n\n"
            "One row is appended whenever a labeller checks an article from a frozen snapshot.\n\n"
            "| Timestamp | Labeler | Website | Snapshot | Article ID | Page | URL | Decision | Title OK | Date OK | Author OK | Body OK | Body completeness | Changed fields | Notes |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    changed_fields = [field for field in ("title", "date", "author", "body") if label.get(f"gold_{field}", "").strip()]
    row = [
        label.get("timestamp", ""),
        label.get("labeler", ""),
        meta.get("website", ""),
        snapshot_name,
        article.get("id", ""),
        article.get("page_num", ""),
        article.get("url", ""),
        label.get("decision", ""),
        label.get("title_status", ""),
        label.get("date_status", ""),
        label.get("author_status", ""),
        label.get("body_status", ""),
        label.get("body_completeness", ""),
        ", ".join(changed_fields),
        label.get("notes", ""),
    ]
    with MANUAL_LABEL_TABLE.open("a", encoding="utf-8") as f:
        f.write("| " + " | ".join(_md_cell(v) for v in row) + " |\n")


def _update_detailed_eval_table(snapshot_name: str, meta: dict[str, Any], labels: dict[str, Any]) -> None:
    if not DETAILED_EVAL_TABLE.exists():
        return
    summary = _label_summary(labels)
    checked_links = _write_checked_links(snapshot_name, meta, labels)
    rel_links = checked_links.relative_to(WORKSPACE).as_posix()
    ids_preview = ", ".join(str(i) for i in summary["ids"][:20])
    if len(summary["ids"]) > 20:
        ids_preview += ", ..."
    checked_cell = f"{summary['count']} checked: {ids_preview}; links: [{rel_links}]({rel_links})"
    decision_bits = ", ".join(f"{k}={v}" for k, v in sorted(summary["decisions"].items()))
    notes_cell = f"Manual labels updated {datetime.now().strftime('%Y-%m-%d')}; {decision_bits}"

    lines = DETAILED_EVAL_TABLE.read_text(encoding="utf-8").splitlines()
    updated = False
    new_lines = []
    for line in lines:
        if line.startswith("|") and snapshot_name in line and not line.startswith("| #"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 15:
                cells[13] = checked_cell
                cells[14] = notes_cell if not cells[14] else f"{cells[14]}; {notes_cell}"
                line = "| " + " | ".join(_md_cell(c) for c in cells) + " |"
                updated = True
        new_lines.append(line)
    if updated:
        DETAILED_EVAL_TABLE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _append_failure_notes_table(snapshot_name: str, meta: dict[str, Any], note: dict[str, Any]) -> None:
    path = WORKSPACE / "failure_analysis_table.md"
    if not path.exists():
        path.write_text(
            "# Failure Analysis Table\n\n"
            "One row per manually reviewed failed link.\n\n"
            "| Timestamp | Labeler | Website | Snapshot | URL | Category | Should count as article? | Notes | Failure reason |\n"
            "|---|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    row = [
        note.get("timestamp", ""),
        note.get("labeler", ""),
        meta.get("website", ""),
        snapshot_name,
        note.get("url", ""),
        note.get("category", ""),
        note.get("should_count_as_article", ""),
        note.get("notes", ""),
        note.get("reason", ""),
    ]
    with path.open("a", encoding="utf-8") as f:
        f.write("| " + " | ".join(_md_cell(v) for v in row) + " |\n")


def _append_na_notes_table(snapshot_name: str, meta: dict[str, Any], note: dict[str, Any]) -> None:
    path = WORKSPACE / "na_field_analysis_table.md"
    if not path.exists():
        path.write_text(
            "# N/A Field Analysis Table\n\n"
            "One row per manually reviewed N/A field sample.\n\n"
            "| Timestamp | Labeler | Website | Snapshot | Field | URL | Explanation | Expected? | Notes |\n"
            "|---|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    row = [
        note.get("timestamp", ""),
        note.get("labeler", ""),
        meta.get("website", ""),
        snapshot_name,
        note.get("field", ""),
        note.get("url", ""),
        note.get("explanation", ""),
        note.get("expected", ""),
        note.get("notes", ""),
    ]
    with path.open("a", encoding="utf-8") as f:
        f.write("| " + " | ".join(_md_cell(v) for v in row) + " |\n")


# ---------------------------------------------------------------------------
# Orchestrator job helpers
# ---------------------------------------------------------------------------

def _read_process_output(job_id: str, process: subprocess.Popen[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
        process.wait()
        log_file.write(f"\n[process exited with code {process.returncode}]\n")
        log_file.flush()
    if job_id in JOBS:
        JOBS[job_id]["returncode"] = process.returncode
        JOBS[job_id]["finished_at"] = datetime.now().isoformat(timespec="seconds")


def _build_orchestrator_answers(form: dict[str, str]) -> list[str]:
    pagination = form.get("pagination_type", "2")
    listing_url = form.get("listing_url", "").strip()
    answers = [
        form.get("api_key", "").strip(),
        form.get("model_choice", "").strip(),
        listing_url,
        form.get("website", "").strip(),
        form.get("requirements", "").strip() or "title, date, author, article body text",
        pagination,
    ]
    if pagination == "1":
        answers.extend([
            listing_url,
            form.get("page2_url", "").strip(),
            form.get("pagination_pattern", "").strip(),
            form.get("total_pages", "").strip(),
        ])
    elif pagination == "3":
        answers.append(form.get("scroll_rounds", "").strip())
    elif pagination == "4":
        answers.extend([
            form.get("load_more_selector", "").strip(),
            form.get("scroll_rounds", "").strip(),
        ])
    return answers


def _start_orchestrator_job(answers: list[str]) -> str:
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    log_path = LOGS_DIR / f"orch_pag_snapshot_{job_id}.log"
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        [sys.executable, "-u", "orch_pag_snapshot.py"],
        cwd=str(WORKSPACE),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )
    JOBS[job_id] = {
        "id": job_id,
        "process": process,
        "log_path": log_path,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "returncode": None,
    }
    if process.stdin is not None:
        process.stdin.write("\n".join(answers) + "\n")
        process.stdin.close()
    thread = threading.Thread(target=_read_process_output, args=(job_id, process, log_path), daemon=True)
    thread.start()
    return job_id


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    jobs = [
        {
            "id": job_id,
            "started_at": job.get("started_at"),
            "returncode": job.get("returncode"),
        }
        for job_id, job in sorted(JOBS.items(), reverse=True)
    ]
    return render_template("labeling.html", page="index", snapshots=_snapshot_cards(), jobs=jobs)


@app.route("/api/models", methods=["POST"])
def available_models():
    api_key = (request.get_json(silent=True) or {}).get("api_key", "").strip()
    if not api_key:
        return jsonify({"ok": False, "error": "Enter the Gemini API key first."}), 400
    models = asyncio.run(list_available_models(api_key))
    gemma_models = [model for model in models if "gemma" in model.lower()]
    other_models = [model for model in models if "gemma" not in model.lower()]
    return jsonify({"ok": True, "gemma_models": gemma_models, "other_models": other_models})


@app.route("/runs/start", methods=["POST"])
def start_run():
    if not request.form.get("api_key", "").strip():
        return redirect(url_for("index", error="api_key_required"))
    if not request.form.get("listing_url", "").strip():
        return redirect(url_for("index", error="listing_url_required"))
    answers = _build_orchestrator_answers(request.form)
    job_id = _start_orchestrator_job(answers)
    return redirect(url_for("job_page", job_id=job_id))


@app.route("/jobs/<job_id>")
def job_page(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        abort(404)
    return render_template("labeling.html", page="job", job_id=job_id)


@app.route("/api/compare", methods=["POST"])
def compare_text():
    data = request.get_json() or {}
    predicted = data.get("predicted", "")
    manual = data.get("manual", "")
    return jsonify({
        "ok": True,
        "scores": _token_scores(predicted, manual),
        "diff": _diff_segments(predicted, manual),
    })


@app.route("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Job not found"}), 404
    log_path = job["log_path"]
    output = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    return jsonify({
        "ok": True,
        "output": output[-30000:],
        "returncode": job.get("returncode"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
    })


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Job not found"}), 404
    process: subprocess.Popen[str] = job["process"]
    if process.poll() is not None:
        job["returncode"] = process.returncode
        return jsonify({"ok": False, "error": "Job already finished", "returncode": process.returncode})
    job["cancel_requested_at"] = datetime.now().isoformat(timespec="seconds")
    with job["log_path"].open("a", encoding="utf-8") as f:
        f.write("\n[cancel requested by labeller]\n")
    process.terminate()
    return jsonify({"ok": True})


@app.route("/api/jobs/<job_id>/input", methods=["POST"])
def job_input(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Job not found"}), 404
    process: subprocess.Popen[str] = job["process"]
    if process.poll() is not None:
        return jsonify({"ok": False, "error": "Job already finished"})
    answer = request.form.get("answer", "")
    assert process.stdin is not None
    process.stdin.write(answer + "\n")
    process.stdin.flush()
    with job["log_path"].open("a", encoding="utf-8") as f:
        f.write("\n[answered]\n")
    return jsonify({"ok": True})


@app.route("/snapshots/<snapshot_name>")
def snapshot_page(snapshot_name: str):
    meta = _load_snapshot_meta(snapshot_name)
    if not meta:
        abort(404)
    labels = _load_labels(snapshot_name)
    sample_size = int(request.args.get("n", "10") or 10)
    seed = request.args.get("seed", "")
    include_checked = request.args.get("include_checked") == "1"
    articles = _sample_articles(meta, labels, sample_size, seed, include_checked)
    summary = _label_summary(labels)
    return render_template(
        "labeling.html",
        page="snapshot",
        snapshot_name=snapshot_name,
        meta=meta,
        articles=articles,
        labels=labels,
        failed_links=_load_run_list(meta, "failed_links.json"),
        dropped_links=_load_run_list(meta, "dropped_links.json"),
        na_stats=_na_field_stats(_load_run_list(meta, "extracted_data_all.json")),
        summary=summary,
        sample_size=sample_size,
        seed=seed,
        include_checked=include_checked,
    )


@app.route("/snapshots/<snapshot_name>/failures", methods=["GET", "POST"])
def review_failures(snapshot_name: str):
    meta = _load_snapshot_meta(snapshot_name)
    if not meta:
        abort(404)
    failures = _load_run_list(meta, "failed_links.json")
    notes = _load_keyed_notes(snapshot_name, "failures")
    if request.method == "POST":
        key = request.form.get("key", "")
        labeler = request.form.get("labeler", "").strip() or "anonymous"
        note = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "labeler": labeler,
            "website": meta.get("website", ""),
            "url": request.form.get("url", ""),
            "title": request.form.get("title", ""),
            "reason": request.form.get("reason", ""),
            "category": request.form.get("category", "needs_review"),
            "should_count_as_article": request.form.get("should_count_as_article", "unknown"),
            "notes": request.form.get("notes", "").strip(),
        }
        note["key"] = key or _safe_name(note["url"])
        _record_note(snapshot_name, "failures", labeler, note)
        return redirect(url_for("review_failures", snapshot_name=snapshot_name, saved=1))
    return render_template("labeling.html", page="failures", snapshot_name=snapshot_name, meta=meta, failures=failures, notes=notes)


@app.route("/snapshots/<snapshot_name>/dropped", methods=["GET", "POST"])
def review_dropped(snapshot_name: str):
    meta = _load_snapshot_meta(snapshot_name)
    if not meta:
        abort(404)
    dropped = _load_run_list(meta, "dropped_links.json")
    notes = _load_keyed_notes(snapshot_name, "dropped")
    if request.method == "POST":
        key = request.form.get("key", "")
        labeler = request.form.get("labeler", "").strip() or "anonymous"
        note = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "labeler": labeler,
            "website": meta.get("website", ""),
            "url": request.form.get("url", ""),
            "title": request.form.get("title", ""),
            "reason": request.form.get("reason", ""),
            "verdict": request.form.get("verdict", "needs_review"),
            "notes": request.form.get("notes", "").strip(),
        }
        note["key"] = key or _safe_name(note["url"])
        _record_note(snapshot_name, "dropped", labeler, note)
        return redirect(url_for("review_dropped", snapshot_name=snapshot_name, saved=1))
    return render_template("labeling.html", page="dropped", snapshot_name=snapshot_name, meta=meta, dropped=dropped, notes=notes)


@app.route("/snapshots/<snapshot_name>/na", methods=["GET", "POST"])
def review_na(snapshot_name: str):
    meta = _load_snapshot_meta(snapshot_name)
    if not meta:
        abort(404)
    extracted = _load_run_list(meta, "extracted_data_all.json")
    notes = _load_keyed_notes(snapshot_name, "na")
    if request.method == "POST":
        key = request.form.get("key", "")
        labeler = request.form.get("labeler", "").strip() or "anonymous"
        note = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "labeler": labeler,
            "website": meta.get("website", ""),
            "field": request.form.get("field", ""),
            "url": request.form.get("url", ""),
            "title": request.form.get("title", ""),
            "explanation": request.form.get("explanation", "needs_review"),
            "expected": request.form.get("expected", "unknown"),
            "notes": request.form.get("notes", "").strip(),
        }
        note["key"] = key or _safe_name(note["field"] + "_" + note["url"])
        _record_note(snapshot_name, "na", labeler, note)
        return redirect(url_for("review_na", snapshot_name=snapshot_name, field=note["field"], saved=1))
    field = request.args.get("field", "")
    sample_size = int(request.args.get("n", "10") or 10)
    seed = request.args.get("seed", "")
    samples = _sample_na_articles(extracted, field, sample_size, seed) if field else []
    return render_template(
        "labeling.html",
        page="na",
        snapshot_name=snapshot_name,
        meta=meta,
        na_stats=_na_field_stats(extracted),
        selected_field=field,
        samples=samples,
        notes=notes,
        sample_size=sample_size,
        seed=seed,
    )


@app.route("/snapshots/<snapshot_name>/articles/<int:article_id>", methods=["GET", "POST"])
def label_article(snapshot_name: str, article_id: int):
    meta = _load_snapshot_meta(snapshot_name)
    if not meta:
        abort(404)
    article = _article_by_id(meta, article_id)
    if not article:
        abort(404)
    labels = _load_labels(snapshot_name)
    predictions = _load_predictions(meta)
    pred = predictions.get(str(article.get("url", "")).strip(), {})
    pred_data = pred.get("data", {}) if isinstance(pred, dict) else {}

    if request.method == "POST":
        label = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "snapshot": snapshot_name,
            "website": meta.get("website", ""),
            "article_id": article_id,
            "url": article.get("url", ""),
            "file": article.get("file", ""),
            "page_num": article.get("page_num", ""),
            "labeler": request.form.get("labeler", "").strip() or "anonymous",
            "decision": request.form.get("decision", "needs_review"),
            "title_status": request.form.get("title_status", "not_checked"),
            "date_status": request.form.get("date_status", "not_checked"),
            "author_status": request.form.get("author_status", "not_checked"),
            "body_status": request.form.get("body_status", "not_checked"),
            "body_completeness": request.form.get("body_completeness", "not_checked"),
            "gold_title": request.form.get("gold_title", "").strip(),
            "gold_date": request.form.get("gold_date", "").strip(),
            "gold_author": request.form.get("gold_author", "").strip(),
            "gold_body": request.form.get("gold_body", "").strip(),
            "notes": request.form.get("notes", "").strip(),
            "predicted_data": pred_data,
        }
        _record_label(snapshot_name, label["labeler"], label)
        labels[str(article_id)] = label
        next_unchecked = None
        for candidate in meta.get("articles", []) or []:
            cid = str(candidate.get("id"))
            if cid not in labels:
                next_unchecked = candidate.get("id")
                break
        if request.form.get("save_next") and next_unchecked:
            return redirect(url_for("label_article", snapshot_name=snapshot_name, article_id=next_unchecked, saved=1))
        return redirect(url_for("snapshot_page", snapshot_name=snapshot_name, saved=1))

    existing = labels.get(str(article_id), {})
    return render_template(
        "labeling.html",
        page="article",
        snapshot_name=snapshot_name,
        meta=meta,
        article=article,
        pred_data=pred_data,
        pred_values={
            "title": _pick_pred_field(pred_data, ("title",)),
            "date": _pick_pred_field(pred_data, ("date", "published", "time")),
            "author": _pick_pred_field(pred_data, ("author", "writer", "byline")),
            "body": _pick_pred_field(pred_data, ("article body text", "article_body_text", "body", "content", "text")),
        },
        existing=existing,
    )


@app.route("/snapshots/<snapshot_name>/html/<filename>")
def snapshot_html(snapshot_name: str, filename: str):
    path = _safe_snapshot_file(snapshot_name, filename)
    if not path:
        abort(404)
    html = path.read_text(encoding="utf-8", errors="replace")
    # Resolve the article's original URL so relative CSS/images render, and
    # inject a <base> tag. The iframe itself is sandboxed (scripts disabled) in
    # the template so analytics/consent scripts cannot blank or redirect the
    # saved page during manual review.
    meta = _load_snapshot_meta(snapshot_name) or {}
    url = ""
    for article in meta.get("articles", []) or []:
        if article.get("file") == filename:
            url = str(article.get("url", "")).strip()
            break
    if url:
        base_tag = f'<base href="{url.replace(chr(34), "%22")}">'
        # Keep wide media inside the preview pane so it does not force horizontal scroll.
        fit_style = "<style>img,figure,video,iframe{max-width:100% !important;height:auto !important;}</style>"
        html = re.sub(r"(<head[^>]*>)", lambda m: m.group(1) + base_tag + fit_style, html, count=1, flags=re.IGNORECASE)
    return app.response_class(html, mimetype="text/html")


if __name__ == "__main__":
    print("Snapshot Labeling App running at http://localhost:5001")
    app.run(debug=True, port=5001, use_reloader=False)
