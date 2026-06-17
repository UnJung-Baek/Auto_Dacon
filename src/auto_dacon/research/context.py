from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def read_text_if_exists(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:] if len(text) > max_chars else text


def compact_dataframe_profile(path: Path, target_column: str | None = None) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(path, nrows=20000)
    missing_ratio = df.isna().mean().sort_values(ascending=False)
    missing_nonzero = missing_ratio[missing_ratio > 0].head(50).round(4).to_dict()
    numeric_cols = list(df.select_dtypes(include="number").columns)
    object_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    profile: dict[str, Any] = {
        "file": path.name,
        "rows_profiled": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_nonzero_top50": missing_nonzero,
        "numeric_columns": numeric_cols,
        "categorical_columns": object_cols,
    }
    profile["categorical_cardinality"] = {
        col: int(df[col].nunique(dropna=True)) for col in object_cols
    }
    if target_column and target_column in df.columns:
        target = pd.to_numeric(df[target_column], errors="coerce")
        profile["target_summary"] = target.describe().round(4).to_dict()
    numeric_summary_cols = []
    if target_column and target_column in numeric_cols:
        numeric_summary_cols.append(target_column)
    numeric_summary_cols.extend(
        col for col in missing_nonzero if col in numeric_cols and col not in numeric_summary_cols
    )
    numeric_summary_cols.extend(col for col in numeric_cols[:20] if col not in numeric_summary_cols)
    if numeric_summary_cols:
        profile["selected_numeric_summary"] = (
            df[numeric_summary_cols[:40]]
            .describe()
            .round(4)
            .to_dict()
        )
    return profile


def load_project_research_context(project_dir: Path, metadata: dict) -> dict[str, Any]:
    notes_dir = project_dir / "notes"
    outputs_dir = project_dir / "outputs"
    data_dir = project_dir / "data"
    target_column = metadata.get("target_column")
    context_paths = [
        project_dir / "competition_context.md",
        notes_dir / "competition_context.md",
        notes_dir / "competition.md",
        notes_dir / "score_summary.md",
        notes_dir / "latest_score.md",
        notes_dir / "first_place_code_review.md",
        notes_dir / "react5_first_place_code_gap_closure.txt",
    ]
    note_blocks = {
        str(path.relative_to(project_dir)): read_text_if_exists(path)
        for path in context_paths
        if path.exists()
    }
    for path in sorted(notes_dir.glob("*.md")) + sorted(notes_dir.glob("*.txt")):
        rel = str(path.relative_to(project_dir))
        if rel not in note_blocks:
            note_blocks[rel] = read_text_if_exists(path, max_chars=8000)

    score_history_path = notes_dir / "score_history.jsonl"
    score_history = []
    if score_history_path.exists():
        for line in score_history_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                try:
                    score_history.append(json.loads(line))
                except json.JSONDecodeError:
                    score_history.append({"raw": line})

    submissions = []
    if outputs_dir.exists():
        for path in sorted(outputs_dir.glob("submission*.csv")):
            submissions.append(
                {
                    "file": str(path.relative_to(project_dir)),
                    "size_bytes": path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )

    data_profiles = {}
    for filename in ["train.csv", "test.csv", "sample_submission.csv"]:
        path = data_dir / filename
        if path.exists():
            data_profiles[filename] = compact_dataframe_profile(path, target_column)

    return {
        "metadata": metadata,
        "score_history": score_history[-20:],
        "submissions": submissions[-30:],
        "data_profiles": data_profiles,
        "notes": note_blocks,
    }


def compact_research_context_for_prompt(context: dict[str, Any]) -> dict[str, Any]:
    metadata = context.get("metadata", {})
    compact_metadata = {
        key: metadata.get(key)
        for key in [
            "task_id",
            "competition_url",
            "id_column",
            "target_column",
            "metric",
            "prepared_at",
        ]
        if key in metadata
    }
    compact_profiles = {}
    for name, profile in context.get("data_profiles", {}).items():
        compact_profiles[name] = {
            "file": profile.get("file"),
            "rows_profiled": profile.get("rows_profiled"),
            "n_columns": profile.get("n_columns"),
            "columns": profile.get("columns"),
            "missing_nonzero_top50": profile.get("missing_nonzero_top50"),
            "categorical_columns": profile.get("categorical_columns"),
            "categorical_cardinality": profile.get("categorical_cardinality"),
            "target_summary": profile.get("target_summary"),
        }
    compact: dict[str, Any] = {
        "metadata": compact_metadata,
        "score_history": context.get("score_history", []),
        "submissions": context.get("submissions", []),
        "data_profiles": compact_profiles,
    }

    priority_names = (
        "latest_score",
        "score_summary",
        "first_place",
        "gap_closure",
        "competition_context",
        "latest_research_warm_start",
    )
    notes = context.get("notes", {})
    prioritized_notes = sorted(
        notes.items(),
        key=lambda item: (
            0 if any(name in item[0].lower() for name in priority_names) else 1,
            item[0].lower(),
        ),
    )
    compact_notes = {}
    total_note_chars = 0
    has_node_outputs = any(key in context for key in ["analyst_outputs", "hypothesis_outputs", "critic_outputs"])
    note_budget = 16000 if has_node_outputs else 24000
    for rel, text in prioritized_notes:
        per_note_limit = 5000 if any(name in rel.lower() for name in priority_names) else 1800
        clipped = text[-per_note_limit:] if len(text) > per_note_limit else text
        if total_note_chars + len(clipped) > note_budget:
            compact_notes[rel] = "<omitted due to prompt budget>"
            continue
        compact_notes[rel] = clipped
        total_note_chars += len(clipped)
    compact["notes"] = compact_notes

    for key in ["analyst_outputs", "hypothesis_outputs", "critic_outputs"]:
        if key in context:
            compact[key] = [
                {
                    "model": item.get("model"),
                    "content": item.get("content", "")[-1400:],
                }
                for item in context[key]
            ]
    if "selector_decision" in context:
        selector = context["selector_decision"]
        compact["selector_decision"] = selector[-4000:] if len(selector) > 4000 else selector
    return compact
