from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from auto_dacon.research.schemas import (  # noqa: E402
    Critique,
    EvidenceItem,
    HypothesisProposal,
    NodeResult,
    NodeRole,
    NodeSpec,
    ResearchArtifact,
    ResearchRunConfig,
    ResearchRunResult,
    ResearchStage,
    RiskAssessment,
    SelectionDecision,
    WarmStartInstruction,
)


def _evidence() -> EvidenceItem:
    return EvidenceItem(
        source="notes/score_history.jsonl",
        summary="baseline public score improved",
        kind="score",
        weight=0.8,
        metadata={"experiment": "baseline", "public_score": 0.42},
    )


def _risk() -> RiskAssessment:
    return RiskAssessment(
        leakage="low if fit inside folds",
        public_overfit="medium",
        runtime="low",
        reproducibility="high",
        mitigations=["freeze folds", "compare with baseline"],
    )


def test_research_run_result_roundtrips_nested_schema() -> None:
    node = NodeSpec(
        role=NodeRole.ANALYST,
        model="fake/analyst",
        instruction="summarize the current score evidence",
        name="analyst-a",
    )
    proposal = HypothesisProposal(
        hypothesis_id="h1",
        title="Add fold-safe count encoding",
        rationale="Categorical cardinality is high and baseline underfits.",
        implementation_sketch="Fit count encoders on each training fold only.",
        validation_method="Use the existing folds and compare against baseline.",
        expected_effect="small MAE improvement",
        runtime_cost="low",
        leakage_risk="controlled by fold fitting",
        proposer_model="fake/hypothesis",
        evidence=[_evidence()],
        risks=[_risk()],
    )
    critique = Critique(
        critic_model="fake/critic",
        target_hypothesis_ids=["h1"],
        decision="accept with guardrails",
        rationale="Fast experiment with clear leakage controls.",
        weaknesses=["may overfit rare categories"],
        guardrails=["reject if validation regresses"],
        evidence=[_evidence()],
        risks=[_risk()],
    )
    selection = SelectionDecision(
        selector_model="fake/selector",
        selected_hypothesis_ids=["h1"],
        rejected_hypothesis_ids=["h2"],
        rationale="Best evidence-to-risk tradeoff.",
        validation_guardrails=["keep fold protocol unchanged"],
        implementation_priorities=["minimal feature change"],
        content="Implement h1 next.",
    )
    warm_start = WarmStartInstruction(
        model="fake/warm",
        content="Review notes and implement fold-safe encoding.",
        selected_hypothesis_ids=["h1"],
        validation_protocol="Run baseline comparison on frozen folds.",
        leakage_warnings=["never fit encoders on validation rows"],
        fallback_plan="Abort if validation worsens.",
    )
    result = ResearchRunResult(
        config=ResearchRunConfig(project_dir="C:/tmp/project", round_id="r1", node_specs=[node]),
        analyst_results=[NodeResult(role="analyst", model="fake/analyst", content="analysis")],
        hypotheses=[proposal],
        critiques=[critique],
        selection=selection,
        warm_start=warm_start,
        artifacts=[ResearchArtifact(path="notes/research_rounds/r1/context_snapshot.json", kind="context", description="snapshot")],
    )

    restored = ResearchRunResult.from_dict(result.to_dict())

    assert restored.to_dict() == result.to_dict()
    assert restored.config.node_specs[0].stage == ResearchStage.ANALYSIS
    assert restored.hypotheses[0].evidence[0].metadata["public_score"] == 0.42


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: EvidenceItem(source="", summary="missing source"), "EvidenceItem.source"),
        (lambda: RiskAssessment(leakage="low", public_overfit="", runtime="low", reproducibility="high"), "RiskAssessment.public_overfit"),
        (
            lambda: NodeSpec(role="unknown", model="fake/model", instruction="do work"),
            "NodeSpec.role must be one of",
        ),
        (
            lambda: Critique(critic_model="fake/critic", target_hypothesis_ids=[], decision="reject", rationale="no target"),
            "Critique.target_hypothesis_ids is required",
        ),
        (
            lambda: ResearchRunConfig(project_dir="proj", round_id="r1", node_specs=[]),
            "ResearchRunConfig.node_specs is required",
        ),
    ],
)
def test_schema_validation_rejects_invalid_values(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
