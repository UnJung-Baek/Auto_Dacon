from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from auto_dacon.research.context import (  # noqa: E402
    compact_research_context_for_prompt,
    load_project_research_context,
)
from auto_dacon.research.prompts import research_messages  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_context_loads_notes_scores_submissions_and_tiny_data_profiles(tmp_path: Path) -> None:
    project = tmp_path / "competition_repo"
    metadata = {
        "task_id": "demo",
        "competition_url": "https://dacon.io/competitions/1",
        "id_column": "id",
        "target_column": "target",
        "metric": "MAE",
    }
    _write(project / "auto_dacon_task.json", "{}\n")
    _write(project / "notes" / "competition_context.md", "Use MAE and preserve submission columns.\n")
    _write(project / "notes" / "score_history.jsonl", '{"experiment":"baseline","public_score":0.7}\n')
    _write(project / "outputs" / "submission_baseline.csv", "id,target\n2,0.1\n")
    _write(project / "data" / "train.csv", "id,feature,category,target\n1,10,a,0.1\n2,,b,0.2\n")
    _write(project / "data" / "test.csv", "id,feature,category\n3,11,a\n")
    _write(project / "data" / "sample_submission.csv", "id,target\n3,0\n")

    context = load_project_research_context(project, metadata)

    assert context["metadata"]["metric"] == "MAE"
    assert context["score_history"] == [{"experiment": "baseline", "public_score": 0.7}]
    assert context["submissions"][0]["file"] == "outputs\\submission_baseline.csv" or context["submissions"][0]["file"] == "outputs/submission_baseline.csv"
    assert "notes/competition_context.md" in context["notes"] or "notes\\competition_context.md" in context["notes"]
    assert context["data_profiles"]["train.csv"]["rows_profiled"] == 2
    assert context["data_profiles"]["train.csv"]["target_summary"]["mean"] == 0.15


def test_context_and_prompt_compacting_prioritizes_notes_and_limits_large_content() -> None:
    context = {
        "metadata": {"task_id": "demo", "metric": "MAE", "ignored": "drop me"},
        "score_history": [{"experiment": "baseline", "public_score": 0.7}],
        "submissions": [],
        "data_profiles": {
            "train.csv": {
                "file": "train.csv",
                "rows_profiled": 2,
                "n_columns": 3,
                "columns": ["id", "target", "extra"],
                "dtypes": {"extra": "float64"},
                "selected_numeric_summary": {"extra": {"mean": 1.0}},
                "categorical_columns": [],
            }
        },
        "notes": {
            "notes/latest_score.md": "latest score evidence\n" + "L" * 6000,
            "notes/random.md": "R" * 3000,
        },
        "selector_decision": "S" * 5000,
        "analyst_outputs": [{"model": "fake/analyst", "content": "A" * 2000}],
    }

    compact = compact_research_context_for_prompt(context)
    messages = research_messages("selector", context, "choose the next experiment")

    assert compact["metadata"] == {"task_id": "demo", "metric": "MAE"}
    assert "dtypes" not in compact["data_profiles"]["train.csv"]
    assert "selected_numeric_summary" not in compact["data_profiles"]["train.csv"]
    assert len(compact["notes"]["notes/latest_score.md"]) == 5000
    assert len(compact["notes"]["notes/random.md"]) == 1800
    assert len(compact["selector_decision"]) == 4000
    assert len(compact["analyst_outputs"][0]["content"]) == 1400
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "choose the next experiment" in messages[1]["content"]
