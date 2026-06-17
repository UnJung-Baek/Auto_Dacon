from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = REPO_ROOT / "third_party" / "data_science" / "env.py"
DANGEROUS_SUBMIT_TOKENS = [
    "competition_submit",
    "submit_to_kaggle",
    "KaggleApi",
    "join_competition",
]


def _function_source(name: str) -> str:
    source = ENV_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"Function not found: {name}")


def test_submission_action_path_records_local_csv_without_submit_api() -> None:
    action_source = _function_source("action_dependent_step")

    assert "DataScienceStageNames.SEND_SUBMISSION" in action_source
    assert "record_local_submission" in action_source
    for token in DANGEROUS_SUBMIT_TOKENS:
        assert token not in action_source


def test_local_submission_recorder_only_updates_local_observation_state() -> None:
    recorder_source = _function_source("record_local_submission")

    assert "DSObsKey.SUBMISSION_SENT_SUCCESSFULLY" in recorder_source
    assert "DSObsKey.SENT_SUBMISSION_NAMES" in recorder_source
    for token in DANGEROUS_SUBMIT_TOKENS:
        assert token not in recorder_source
