from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class NodeRole(str, Enum):
    ANALYST = "analyst"
    HYPOTHESIS = "hypothesis"
    CRITIC = "critic"
    SELECTOR = "selector"
    WARM_START_BUILDER = "warm_start_builder"


class ResearchStage(str, Enum):
    ANALYSIS = "analysis"
    HYPOTHESIS_PROPOSAL = "hypothesis_proposal"
    CRITIQUE = "critique"
    SELECTION = "selection"
    WARM_START = "warm_start"


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required and must be a non-empty string")
    return value


def _ensure_list(value: list[Any], field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return value


def _enum_value(value: Enum | str, enum_type: type[Enum], field_name: str) -> str:
    if isinstance(value, enum_type):
        return str(value.value)
    if isinstance(value, str):
        allowed = {str(item.value) for item in enum_type}
        if value in allowed:
            return value
    allowed_values = ", ".join(str(item.value) for item in enum_type)
    raise ValueError(f"{field_name} must be one of: {allowed_values}")


def _json_value(value: Any, field_name: str) -> JsonValue:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return {item.name: _json_value(getattr(value, item.name), f"{field_name}.{item.name}") for item in fields(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{field_name}[]") for item in value]
    if isinstance(value, tuple):
        return [_json_value(item, f"{field_name}[]") for item in value]
    if isinstance(value, Mapping):
        output: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name} keys must be strings")
            output[key] = _json_value(item, f"{field_name}.{key}")
        return output
    raise ValueError(f"{field_name} must be JSON-compatible")


def _json_mapping(value: Any, field_name: str) -> dict[str, JsonValue]:
    converted = _json_value(value, field_name)
    if not isinstance(converted, dict):
        raise ValueError(f"{field_name} must be a JSON-compatible dictionary")
    return converted


def _optional_text_list(value: list[str], field_name: str) -> list[str]:
    _ensure_list(value, field_name)
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{index}] must be a string")
    return value


def _mapping(data: Mapping[str, Any], class_name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError(f"{class_name}.from_dict requires a mapping")
    return data


def _nested_list(data: Mapping[str, Any], key: str, item_type: type[Any]) -> list[Any]:
    value = data.get(key, [])
    _ensure_list(value, key)
    return [item_type.from_dict(item) if isinstance(item, Mapping) else item for item in value]


@dataclass(slots=True)
class EvidenceItem:
    source: str
    summary: str
    kind: str = "note"
    weight: float | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source = _required_text(self.source, "EvidenceItem.source")
        self.summary = _required_text(self.summary, "EvidenceItem.summary")
        self.kind = _required_text(self.kind, "EvidenceItem.kind")
        if self.weight is not None and not isinstance(self.weight, (int, float)):
            raise ValueError("EvidenceItem.weight must be numeric when provided")
        self.metadata = _json_mapping(self.metadata, "EvidenceItem.metadata")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "source": self.source,
            "summary": self.summary,
            "kind": self.kind,
            "weight": self.weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceItem":
        data = _mapping(data, "EvidenceItem")
        return cls(
            source=data.get("source", ""),
            summary=data.get("summary", ""),
            kind=data.get("kind", "note"),
            weight=data.get("weight"),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class RiskAssessment:
    leakage: str
    public_overfit: str
    runtime: str
    reproducibility: str
    mitigations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.leakage = _required_text(self.leakage, "RiskAssessment.leakage")
        self.public_overfit = _required_text(self.public_overfit, "RiskAssessment.public_overfit")
        self.runtime = _required_text(self.runtime, "RiskAssessment.runtime")
        self.reproducibility = _required_text(self.reproducibility, "RiskAssessment.reproducibility")
        self.mitigations = _optional_text_list(self.mitigations, "RiskAssessment.mitigations")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "leakage": self.leakage,
            "public_overfit": self.public_overfit,
            "runtime": self.runtime,
            "reproducibility": self.reproducibility,
            "mitigations": list(self.mitigations),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RiskAssessment":
        data = _mapping(data, "RiskAssessment")
        return cls(
            leakage=data.get("leakage", ""),
            public_overfit=data.get("public_overfit", ""),
            runtime=data.get("runtime", ""),
            reproducibility=data.get("reproducibility", ""),
            mitigations=data.get("mitigations", []),
        )


@dataclass(slots=True)
class NodeSpec:
    role: NodeRole | str
    model: str
    instruction: str
    stage: ResearchStage | str | None = None
    name: str = ""
    max_tokens: int = 2200
    temperature: float = 0.35

    def __post_init__(self) -> None:
        self.role = NodeRole(_enum_value(self.role, NodeRole, "NodeSpec.role"))
        default_stage = _stage_for_role(self.role)
        self.stage = ResearchStage(_enum_value(self.stage or default_stage, ResearchStage, "NodeSpec.stage"))
        self.model = _required_text(self.model, "NodeSpec.model")
        self.instruction = _required_text(self.instruction, "NodeSpec.instruction")
        if self.name and not isinstance(self.name, str):
            raise ValueError("NodeSpec.name must be a string")
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("NodeSpec.max_tokens must be a positive integer")
        if not isinstance(self.temperature, (int, float)):
            raise ValueError("NodeSpec.temperature must be numeric")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "role": self.role.value,
            "stage": self.stage.value,
            "name": self.name,
            "model": self.model,
            "instruction": self.instruction,
            "max_tokens": self.max_tokens,
            "temperature": float(self.temperature),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NodeSpec":
        data = _mapping(data, "NodeSpec")
        return cls(
            role=data.get("role", ""),
            stage=data.get("stage"),
            name=data.get("name", ""),
            model=data.get("model", ""),
            instruction=data.get("instruction", ""),
            max_tokens=data.get("max_tokens", 2200),
            temperature=data.get("temperature", 0.35),
        )


@dataclass(slots=True)
class NodeResult:
    role: NodeRole | str
    model: str
    content: str
    stage: ResearchStage | str | None = None
    name: str = ""
    error: str = ""
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.role = NodeRole(_enum_value(self.role, NodeRole, "NodeResult.role"))
        default_stage = _stage_for_role(self.role)
        self.stage = ResearchStage(_enum_value(self.stage or default_stage, ResearchStage, "NodeResult.stage"))
        self.model = _required_text(self.model, "NodeResult.model")
        self.content = _required_text(self.content, "NodeResult.content")
        if self.name and not isinstance(self.name, str):
            raise ValueError("NodeResult.name must be a string")
        if self.error and not isinstance(self.error, str):
            raise ValueError("NodeResult.error must be a string")
        self.metadata = _json_mapping(self.metadata, "NodeResult.metadata")

    def to_panel_result(self) -> dict[str, str]:
        return {"role": self.role.value, "model": self.model, "content": self.content}

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "role": self.role.value,
            "stage": self.stage.value,
            "name": self.name,
            "model": self.model,
            "content": self.content,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NodeResult":
        data = _mapping(data, "NodeResult")
        return cls(
            role=data.get("role", ""),
            stage=data.get("stage"),
            name=data.get("name", ""),
            model=data.get("model", ""),
            content=data.get("content", ""),
            error=data.get("error", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class HypothesisProposal:
    hypothesis_id: str
    title: str
    rationale: str
    implementation_sketch: str
    validation_method: str
    expected_effect: str
    runtime_cost: str
    leakage_risk: str
    proposer_model: str = ""
    evidence: list[EvidenceItem] = field(default_factory=list)
    risks: list[RiskAssessment] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.hypothesis_id = _required_text(self.hypothesis_id, "HypothesisProposal.hypothesis_id")
        self.title = _required_text(self.title, "HypothesisProposal.title")
        self.rationale = _required_text(self.rationale, "HypothesisProposal.rationale")
        self.implementation_sketch = _required_text(
            self.implementation_sketch,
            "HypothesisProposal.implementation_sketch",
        )
        self.validation_method = _required_text(self.validation_method, "HypothesisProposal.validation_method")
        self.expected_effect = _required_text(self.expected_effect, "HypothesisProposal.expected_effect")
        self.runtime_cost = _required_text(self.runtime_cost, "HypothesisProposal.runtime_cost")
        self.leakage_risk = _required_text(self.leakage_risk, "HypothesisProposal.leakage_risk")
        if self.proposer_model and not isinstance(self.proposer_model, str):
            raise ValueError("HypothesisProposal.proposer_model must be a string")
        self.evidence = [item if isinstance(item, EvidenceItem) else EvidenceItem.from_dict(item) for item in self.evidence]
        self.risks = [item if isinstance(item, RiskAssessment) else RiskAssessment.from_dict(item) for item in self.risks]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "rationale": self.rationale,
            "implementation_sketch": self.implementation_sketch,
            "validation_method": self.validation_method,
            "expected_effect": self.expected_effect,
            "runtime_cost": self.runtime_cost,
            "leakage_risk": self.leakage_risk,
            "proposer_model": self.proposer_model,
            "evidence": [item.to_dict() for item in self.evidence],
            "risks": [item.to_dict() for item in self.risks],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HypothesisProposal":
        data = _mapping(data, "HypothesisProposal")
        return cls(
            hypothesis_id=data.get("hypothesis_id", ""),
            title=data.get("title", ""),
            rationale=data.get("rationale", ""),
            implementation_sketch=data.get("implementation_sketch", ""),
            validation_method=data.get("validation_method", ""),
            expected_effect=data.get("expected_effect", ""),
            runtime_cost=data.get("runtime_cost", ""),
            leakage_risk=data.get("leakage_risk", ""),
            proposer_model=data.get("proposer_model", ""),
            evidence=_nested_list(data, "evidence", EvidenceItem),
            risks=_nested_list(data, "risks", RiskAssessment),
        )


@dataclass(slots=True)
class Critique:
    critic_model: str
    target_hypothesis_ids: list[str]
    decision: str
    rationale: str
    weaknesses: list[str] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    risks: list[RiskAssessment] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.critic_model = _required_text(self.critic_model, "Critique.critic_model")
        self.target_hypothesis_ids = _optional_text_list(self.target_hypothesis_ids, "Critique.target_hypothesis_ids")
        if not self.target_hypothesis_ids:
            raise ValueError("Critique.target_hypothesis_ids is required")
        self.decision = _required_text(self.decision, "Critique.decision")
        self.rationale = _required_text(self.rationale, "Critique.rationale")
        self.weaknesses = _optional_text_list(self.weaknesses, "Critique.weaknesses")
        self.guardrails = _optional_text_list(self.guardrails, "Critique.guardrails")
        self.evidence = [item if isinstance(item, EvidenceItem) else EvidenceItem.from_dict(item) for item in self.evidence]
        self.risks = [item if isinstance(item, RiskAssessment) else RiskAssessment.from_dict(item) for item in self.risks]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "critic_model": self.critic_model,
            "target_hypothesis_ids": list(self.target_hypothesis_ids),
            "decision": self.decision,
            "rationale": self.rationale,
            "weaknesses": list(self.weaknesses),
            "guardrails": list(self.guardrails),
            "evidence": [item.to_dict() for item in self.evidence],
            "risks": [item.to_dict() for item in self.risks],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Critique":
        data = _mapping(data, "Critique")
        return cls(
            critic_model=data.get("critic_model", ""),
            target_hypothesis_ids=data.get("target_hypothesis_ids", []),
            decision=data.get("decision", ""),
            rationale=data.get("rationale", ""),
            weaknesses=data.get("weaknesses", []),
            guardrails=data.get("guardrails", []),
            evidence=_nested_list(data, "evidence", EvidenceItem),
            risks=_nested_list(data, "risks", RiskAssessment),
        )


@dataclass(slots=True)
class SelectionDecision:
    selector_model: str
    selected_hypothesis_ids: list[str]
    rejected_hypothesis_ids: list[str]
    rationale: str
    validation_guardrails: list[str]
    implementation_priorities: list[str]
    content: str = ""

    def __post_init__(self) -> None:
        self.selector_model = _required_text(self.selector_model, "SelectionDecision.selector_model")
        self.selected_hypothesis_ids = _optional_text_list(
            self.selected_hypothesis_ids,
            "SelectionDecision.selected_hypothesis_ids",
        )
        if not self.selected_hypothesis_ids:
            raise ValueError("SelectionDecision.selected_hypothesis_ids is required")
        self.rejected_hypothesis_ids = _optional_text_list(
            self.rejected_hypothesis_ids,
            "SelectionDecision.rejected_hypothesis_ids",
        )
        self.rationale = _required_text(self.rationale, "SelectionDecision.rationale")
        self.validation_guardrails = _optional_text_list(
            self.validation_guardrails,
            "SelectionDecision.validation_guardrails",
        )
        self.implementation_priorities = _optional_text_list(
            self.implementation_priorities,
            "SelectionDecision.implementation_priorities",
        )
        if self.content and not isinstance(self.content, str):
            raise ValueError("SelectionDecision.content must be a string")

    def to_node_result(self) -> NodeResult:
        return NodeResult(
            role=NodeRole.SELECTOR,
            model=self.selector_model,
            content=self.content or self.rationale,
            stage=ResearchStage.SELECTION,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "selector_model": self.selector_model,
            "selected_hypothesis_ids": list(self.selected_hypothesis_ids),
            "rejected_hypothesis_ids": list(self.rejected_hypothesis_ids),
            "rationale": self.rationale,
            "validation_guardrails": list(self.validation_guardrails),
            "implementation_priorities": list(self.implementation_priorities),
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelectionDecision":
        data = _mapping(data, "SelectionDecision")
        return cls(
            selector_model=data.get("selector_model", ""),
            selected_hypothesis_ids=data.get("selected_hypothesis_ids", []),
            rejected_hypothesis_ids=data.get("rejected_hypothesis_ids", []),
            rationale=data.get("rationale", ""),
            validation_guardrails=data.get("validation_guardrails", []),
            implementation_priorities=data.get("implementation_priorities", []),
            content=data.get("content", ""),
        )


@dataclass(slots=True)
class WarmStartInstruction:
    model: str
    content: str
    selected_hypothesis_ids: list[str]
    validation_protocol: str
    leakage_warnings: list[str]
    fallback_plan: str

    def __post_init__(self) -> None:
        self.model = _required_text(self.model, "WarmStartInstruction.model")
        self.content = _required_text(self.content, "WarmStartInstruction.content")
        self.selected_hypothesis_ids = _optional_text_list(
            self.selected_hypothesis_ids,
            "WarmStartInstruction.selected_hypothesis_ids",
        )
        self.validation_protocol = _required_text(
            self.validation_protocol,
            "WarmStartInstruction.validation_protocol",
        )
        self.leakage_warnings = _optional_text_list(self.leakage_warnings, "WarmStartInstruction.leakage_warnings")
        self.fallback_plan = _required_text(self.fallback_plan, "WarmStartInstruction.fallback_plan")

    def to_node_result(self) -> NodeResult:
        return NodeResult(
            role=NodeRole.WARM_START_BUILDER,
            model=self.model,
            content=self.content,
            stage=ResearchStage.WARM_START,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "model": self.model,
            "content": self.content,
            "selected_hypothesis_ids": list(self.selected_hypothesis_ids),
            "validation_protocol": self.validation_protocol,
            "leakage_warnings": list(self.leakage_warnings),
            "fallback_plan": self.fallback_plan,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WarmStartInstruction":
        data = _mapping(data, "WarmStartInstruction")
        return cls(
            model=data.get("model", ""),
            content=data.get("content", ""),
            selected_hypothesis_ids=data.get("selected_hypothesis_ids", []),
            validation_protocol=data.get("validation_protocol", ""),
            leakage_warnings=data.get("leakage_warnings", []),
            fallback_plan=data.get("fallback_plan", ""),
        )


@dataclass(slots=True)
class ResearchArtifact:
    path: str
    kind: str
    description: str
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = _required_text(self.path, "ResearchArtifact.path")
        self.kind = _required_text(self.kind, "ResearchArtifact.kind")
        self.description = _required_text(self.description, "ResearchArtifact.description")
        self.metadata = _json_mapping(self.metadata, "ResearchArtifact.metadata")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "path": self.path,
            "kind": self.kind,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResearchArtifact":
        data = _mapping(data, "ResearchArtifact")
        return cls(
            path=data.get("path", ""),
            kind=data.get("kind", ""),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class ResearchRunConfig:
    project_dir: str
    round_id: str
    node_specs: list[NodeSpec]
    enable_rag: bool = False
    rag_path: str = ""
    max_context_chars: int = 50000
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.project_dir = _required_text(self.project_dir, "ResearchRunConfig.project_dir")
        self.round_id = _required_text(self.round_id, "ResearchRunConfig.round_id")
        _ensure_list(self.node_specs, "ResearchRunConfig.node_specs")
        if not self.node_specs:
            raise ValueError("ResearchRunConfig.node_specs is required")
        self.node_specs = [item if isinstance(item, NodeSpec) else NodeSpec.from_dict(item) for item in self.node_specs]
        if not isinstance(self.enable_rag, bool):
            raise ValueError("ResearchRunConfig.enable_rag must be a boolean")
        if self.rag_path and not isinstance(self.rag_path, str):
            raise ValueError("ResearchRunConfig.rag_path must be a string")
        if not isinstance(self.max_context_chars, int) or self.max_context_chars <= 0:
            raise ValueError("ResearchRunConfig.max_context_chars must be a positive integer")
        self.metadata = _json_mapping(self.metadata, "ResearchRunConfig.metadata")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "project_dir": self.project_dir,
            "round_id": self.round_id,
            "node_specs": [item.to_dict() for item in self.node_specs],
            "enable_rag": self.enable_rag,
            "rag_path": self.rag_path,
            "max_context_chars": self.max_context_chars,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResearchRunConfig":
        data = _mapping(data, "ResearchRunConfig")
        return cls(
            project_dir=data.get("project_dir", ""),
            round_id=data.get("round_id", ""),
            node_specs=_nested_list(data, "node_specs", NodeSpec),
            enable_rag=data.get("enable_rag", False),
            rag_path=data.get("rag_path", ""),
            max_context_chars=data.get("max_context_chars", 50000),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class ResearchRunResult:
    config: ResearchRunConfig
    analyst_results: list[NodeResult] = field(default_factory=list)
    hypothesis_results: list[NodeResult] = field(default_factory=list)
    critic_results: list[NodeResult] = field(default_factory=list)
    hypotheses: list[HypothesisProposal] = field(default_factory=list)
    critiques: list[Critique] = field(default_factory=list)
    selection: SelectionDecision | None = None
    warm_start: WarmStartInstruction | None = None
    artifacts: list[ResearchArtifact] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.config, ResearchRunConfig):
            self.config = ResearchRunConfig.from_dict(self.config)  # type: ignore[arg-type]
        self.analyst_results = _coerce_results(self.analyst_results, "ResearchRunResult.analyst_results")
        self.hypothesis_results = _coerce_results(self.hypothesis_results, "ResearchRunResult.hypothesis_results")
        self.critic_results = _coerce_results(self.critic_results, "ResearchRunResult.critic_results")
        self.hypotheses = [
            item if isinstance(item, HypothesisProposal) else HypothesisProposal.from_dict(item)
            for item in self.hypotheses
        ]
        self.critiques = [item if isinstance(item, Critique) else Critique.from_dict(item) for item in self.critiques]
        if self.selection is not None and not isinstance(self.selection, SelectionDecision):
            self.selection = SelectionDecision.from_dict(self.selection)  # type: ignore[arg-type]
        if self.warm_start is not None and not isinstance(self.warm_start, WarmStartInstruction):
            self.warm_start = WarmStartInstruction.from_dict(self.warm_start)  # type: ignore[arg-type]
        self.artifacts = [
            item if isinstance(item, ResearchArtifact) else ResearchArtifact.from_dict(item) for item in self.artifacts
        ]

    def panel_results(self) -> dict[str, list[dict[str, str]]]:
        return {
            "analyst_outputs": [item.to_panel_result() for item in self.analyst_results],
            "hypothesis_outputs": [item.to_panel_result() for item in self.hypothesis_results],
            "critic_outputs": [item.to_panel_result() for item in self.critic_results],
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "config": self.config.to_dict(),
            "analyst_results": [item.to_dict() for item in self.analyst_results],
            "hypothesis_results": [item.to_dict() for item in self.hypothesis_results],
            "critic_results": [item.to_dict() for item in self.critic_results],
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "critiques": [item.to_dict() for item in self.critiques],
            "selection": self.selection.to_dict() if self.selection else None,
            "warm_start": self.warm_start.to_dict() if self.warm_start else None,
            "artifacts": [item.to_dict() for item in self.artifacts],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResearchRunResult":
        data = _mapping(data, "ResearchRunResult")
        return cls(
            config=ResearchRunConfig.from_dict(data.get("config", {})),
            analyst_results=_nested_list(data, "analyst_results", NodeResult),
            hypothesis_results=_nested_list(data, "hypothesis_results", NodeResult),
            critic_results=_nested_list(data, "critic_results", NodeResult),
            hypotheses=_nested_list(data, "hypotheses", HypothesisProposal),
            critiques=_nested_list(data, "critiques", Critique),
            selection=SelectionDecision.from_dict(data["selection"]) if data.get("selection") else None,
            warm_start=WarmStartInstruction.from_dict(data["warm_start"]) if data.get("warm_start") else None,
            artifacts=_nested_list(data, "artifacts", ResearchArtifact),
        )


def _stage_for_role(role: NodeRole) -> ResearchStage:
    return {
        NodeRole.ANALYST: ResearchStage.ANALYSIS,
        NodeRole.HYPOTHESIS: ResearchStage.HYPOTHESIS_PROPOSAL,
        NodeRole.CRITIC: ResearchStage.CRITIQUE,
        NodeRole.SELECTOR: ResearchStage.SELECTION,
        NodeRole.WARM_START_BUILDER: ResearchStage.WARM_START,
    }[role]


def _coerce_results(value: list[Any], field_name: str) -> list[NodeResult]:
    _ensure_list(value, field_name)
    return [item if isinstance(item, NodeResult) else NodeResult.from_dict(item) for item in value]
