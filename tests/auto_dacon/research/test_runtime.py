from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from auto_dacon.research.runtime import (  # noqa: E402
    DEFAULT_NODE_MAX_TOKENS,
    DEFAULT_SELECTOR_MAX_TOKENS,
    DEFAULT_WARM_START_MAX_TOKENS,
    ResearchRuntime,
    safe_model_name,
)
from auto_dacon.research.schemas import NodeSpec  # noqa: E402


class RecordingFakeClient:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set()
        self.calls: list[dict[str, Any]] = []

    def chat(self, model: str, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens})
        if model in self.failures:
            raise RuntimeError(f"planned failure for {model}")
        return f"response from {model} with {max_tokens} tokens"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "competition_repo"
    metadata = {
        "task_id": "demo",
        "competition_url": "https://dacon.io/competitions/1",
        "id_column": "id",
        "target_column": "target",
        "metric": "MAE",
    }
    _write(project / "auto_dacon_task.json", json.dumps(metadata))
    _write(project / "notes" / "competition_context.md", "Regression task with MAE metric.\n")
    _write(project / "notes" / "score_history.jsonl", '{"experiment":"baseline","public_score":0.7}\n')
    _write(project / "outputs" / "submission_baseline.csv", "id,target\n2,0.1\n")
    _write(project / "data" / "train.csv", "id,feature,target\n1,10,0.1\n")
    _write(project / "data" / "test.csv", "id,feature\n2,11\n")
    _write(project / "data" / "sample_submission.csv", "id,target\n2,0\n")
    return project


def _node(role: str, model: str, max_tokens: int = DEFAULT_NODE_MAX_TOKENS) -> NodeSpec:
    return NodeSpec(role=role, model=model, instruction=f"Perform {role} work.", max_tokens=max_tokens)


def _artifact_path(project: Path, result: object, kind: str) -> Path:
    artifact = next(item for item in result.artifacts if item.kind == kind)
    return project / artifact.path


def test_runtime_with_fake_client_writes_artifacts_and_uses_default_token_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network should not be used by fake-client runtime tests")

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("requests.post", fail_network)
    project = _project(tmp_path)
    client = RecordingFakeClient()

    result = ResearchRuntime(client=client).run_round(
        project_dir=project,
        round_id="round/one",
        panel_nodes=[
            _node("analyst", "fake/analyst"),
            _node("hypothesis", "fake/hypothesis", max_tokens=123),
            _node("critic", "fake/critic"),
        ],
        selector_node=_node("selector", "fake/selector"),
        warm_start_node=_node("warm_start_builder", "fake/warm"),
    )

    round_dir = project / "notes" / "research_rounds" / "round_one"
    assert _artifact_path(project, result, "context").is_file()
    assert (round_dir / f"analyst_{safe_model_name('fake/analyst')}.md").is_file()
    assert (round_dir / f"hypothesis_{safe_model_name('fake/hypothesis')}.md").is_file()
    assert (round_dir / f"critic_{safe_model_name('fake/critic')}.md").is_file()
    assert _artifact_path(project, result, "selector").name == "selector_decision.md"
    assert _artifact_path(project, result, "selector").is_file()
    assert _artifact_path(project, result, "warm_start").name == "warm_start_for_react.txt"
    assert _artifact_path(project, result, "warm_start").is_file()
    assert _artifact_path(project, result, "latest_warm_start").is_file()
    assert _artifact_path(project, result, "latest_warm_start").read_text(encoding="utf-8") == _artifact_path(
        project,
        result,
        "warm_start",
    ).read_text(encoding="utf-8")

    calls_by_model = {call["model"]: call for call in client.calls}
    assert calls_by_model["fake/analyst"]["max_tokens"] == DEFAULT_NODE_MAX_TOKENS
    assert calls_by_model["fake/hypothesis"]["max_tokens"] == 123
    assert calls_by_model["fake/critic"]["max_tokens"] == DEFAULT_NODE_MAX_TOKENS
    assert calls_by_model["fake/selector"]["max_tokens"] == DEFAULT_SELECTOR_MAX_TOKENS
    assert calls_by_model["fake/warm"]["max_tokens"] == DEFAULT_WARM_START_MAX_TOKENS
    assert "Regression task with MAE metric." in calls_by_model["fake/selector"]["messages"][1]["content"]


def test_runtime_mixed_panel_failure_records_failure_and_continues(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = RecordingFakeClient(failures={"fake/broken"})

    result = ResearchRuntime(client=client).run_round(
        project_dir=project,
        round_id="mixed",
        panel_nodes=[
            _node("analyst", "fake/broken"),
            _node("analyst", "fake/good"),
            _node("hypothesis", "fake/hypothesis"),
            _node("critic", "fake/critic"),
        ],
        selector_node=_node("selector", "fake/selector"),
        warm_start_node=_node("warm_start_builder", "fake/warm"),
    )

    failed, succeeded = result.analyst_results
    assert failed.error.startswith("RuntimeError: planned failure")
    assert failed.content.startswith("ERROR from fake/broken")
    assert (project / "notes" / "research_rounds" / "mixed" / f"analyst_{safe_model_name('fake/broken')}.md").is_file()
    assert succeeded.error == ""
    assert succeeded.content.startswith("response from fake/good")
    assert _artifact_path(project, result, "selector").is_file()
    assert _artifact_path(project, result, "latest_warm_start").is_file()


def test_runtime_all_failed_panel_raises_before_selector_or_warm_start(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = RecordingFakeClient(failures={"fake/broken-a", "fake/broken-b"})

    with pytest.raises(RuntimeError, match="All models failed in research node: analyst"):
        ResearchRuntime(client=client).run_round(
            project_dir=project,
            round_id="all_failed",
            panel_nodes=[
                _node("analyst", "fake/broken-a"),
                _node("analyst", "fake/broken-b"),
                _node("hypothesis", "fake/hypothesis"),
                _node("critic", "fake/critic"),
            ],
            selector_node=_node("selector", "fake/selector"),
            warm_start_node=_node("warm_start_builder", "fake/warm"),
        )

    assert [call["model"] for call in client.calls] == ["fake/broken-a", "fake/broken-b"]
    round_dir = project / "notes" / "research_rounds" / "all_failed"
    assert (round_dir / "context_snapshot.json").is_file()
    assert (round_dir / f"analyst_{safe_model_name('fake/broken-a')}.md").is_file()
    assert (round_dir / f"analyst_{safe_model_name('fake/broken-b')}.md").is_file()
    assert not (round_dir / "selector_decision.md").exists()
    assert not (round_dir / "warm_start_for_react.txt").exists()
    assert not (project / "notes" / "latest_research_warm_start.txt").exists()


def test_runtime_selector_failure_does_not_write_warm_start(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = RecordingFakeClient(failures={"fake/selector"})

    with pytest.raises(RuntimeError, match="Required research node failed: selector"):
        ResearchRuntime(client=client).run_round(
            project_dir=project,
            round_id="selector_failed",
            panel_nodes=[
                _node("analyst", "fake/analyst"),
                _node("hypothesis", "fake/hypothesis"),
                _node("critic", "fake/critic"),
            ],
            selector_node=_node("selector", "fake/selector"),
            warm_start_node=_node("warm_start_builder", "fake/warm"),
        )

    round_dir = project / "notes" / "research_rounds" / "selector_failed"
    assert not (round_dir / "selector_decision.md").exists()
    assert not (round_dir / "warm_start_for_react.txt").exists()
    assert not (project / "notes" / "latest_research_warm_start.txt").exists()


def test_runtime_warm_start_failure_does_not_update_latest_warm_start(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = RecordingFakeClient(failures={"fake/warm"})

    with pytest.raises(RuntimeError, match="Required research node failed: warm_start_builder"):
        ResearchRuntime(client=client).run_round(
            project_dir=project,
            round_id="warm_start_failed",
            panel_nodes=[
                _node("analyst", "fake/analyst"),
                _node("hypothesis", "fake/hypothesis"),
                _node("critic", "fake/critic"),
            ],
            selector_node=_node("selector", "fake/selector"),
            warm_start_node=_node("warm_start_builder", "fake/warm"),
        )

    round_dir = project / "notes" / "research_rounds" / "warm_start_failed"
    assert (round_dir / "selector_decision.md").is_file()
    assert not (round_dir / "warm_start_for_react.txt").exists()
    assert not (project / "notes" / "latest_research_warm_start.txt").exists()
