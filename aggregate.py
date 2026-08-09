"""Rebuild the aggregate label tables from per-labeller JSONL files.

Run by the project maintainer only (labellers never run this):

    python aggregate.py

Labellers write append-only, per-labeller files that git merges without conflicts:
    labels/<snapshot>/articles/<labeler>.jsonl
    labels/<snapshot>/failures/<labeler>.jsonl
    labels/<snapshot>/na/<labeler>.jsonl

This script regenerates the derived markdown tables from those files. The derived
tables are gitignored, so no labeller ever commits (and conflicts on) them.
"""

from __future__ import annotations

import labeling_app as la

MANUAL_TABLE = la.MANUAL_LABEL_TABLE
FAILURE_TABLE = la.WORKSPACE / "failure_analysis_table.md"
NA_TABLE = la.WORKSPACE / "na_field_analysis_table.md"


def _snapshot_names() -> list[str]:
    return sorted(p.parent.name for p in la.SNAPSHOTS_DIR.glob("*/metadata.json"))


def rebuild() -> None:
    # Start the append-only tables from scratch so re-running is deterministic.
    for path in (MANUAL_TABLE, FAILURE_TABLE, NA_TABLE):
        if path.exists():
            path.unlink()

    snapshots = _snapshot_names()
    for snapshot_name in snapshots:
        meta = la._load_snapshot_meta(snapshot_name)
        if not meta:
            continue

        # Manual labeling table: one row per labeling event, across all labellers.
        for rec in la._iter_jsonl_records(la._snapshot_label_dir(snapshot_name, "articles")):
            article = {
                "id": rec.get("article_id", ""),
                "page_num": rec.get("page_num", ""),
                "url": rec.get("url", ""),
            }
            la._append_manual_label_table(snapshot_name, meta, article, rec)

        # Failure analysis table.
        for rec in la._iter_jsonl_records(la._snapshot_label_dir(snapshot_name, "failures")):
            la._append_failure_notes_table(snapshot_name, meta, rec)

        # N/A field analysis table.
        for rec in la._iter_jsonl_records(la._snapshot_label_dir(snapshot_name, "na")):
            la._append_na_notes_table(snapshot_name, meta, rec)

        # Per-snapshot checked-links file + detailed evaluation table column.
        labels = la._load_labels(snapshot_name)
        if labels:
            la._write_checked_links(snapshot_name, meta, labels)
            la._update_detailed_eval_table(snapshot_name, meta, labels)

    print(f"Rebuilt aggregate tables from {len(snapshots)} snapshot(s):")
    for path in (MANUAL_TABLE, FAILURE_TABLE, NA_TABLE):
        print(f"  {path.name}: {'written' if path.exists() else 'no data'}")


if __name__ == "__main__":
    rebuild()
