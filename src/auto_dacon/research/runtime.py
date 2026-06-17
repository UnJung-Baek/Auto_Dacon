from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from auto_dacon.research.context import load_project_research_context
from auto_dacon.research.openrouter import OpenRouterClient, ResearchClient
from auto_dacon.research.prompts import research_messages
from auto_dacon.research.schemas import (
    NodeResult,
    NodeRole,
    NodeSpec,
    ResearchArtifact,
    ResearchRunConfig,
    ResearchRunResult,
    ResearchStage,
)


DEFAULT_SELECTOR_MAX_TOKENS = 2600
DEFAULT_WARM_START_MAX_TOKENS = 3000
DEFAULT_NODE_MAX_TOKENS = 2200


@dataclass(frozen=True, slots=True)
class ResearchRuntimePaths:
    round_dir: Path
    context_snapshot: Path
    selector_decision: Path
    warm_start: Path
    latest_warm_start: Path


class ResearchRuntime:
    def __init__(self, client: ResearchClient | None = None) -> None:
        self.client = client or OpenRouterClient()

    def run(self, config: ResearchRunConfig) -> ResearchRunResult:
        nodes = list(config.node_specs)
        return self.run_round(
            project_dir=Path(config.project_dir),
            round_id=config.round_id,
            panel_nodes=[node for node in nodes if node.role in _PANEL_ROLES],
            selector_node=_single_node(nodes, NodeRole.SELECTOR),
            warm_start_node=_single_node(nodes, NodeRole.WARM_START_BUILDER),
            metadata=dict(config.metadata),
            config=config,
        )

    def run_round(
        self,
        *,
        project_dir: str | Path,
        round_id: str | None = None,
        panel_nodes: Iterable[NodeSpec],
        selector_node: NodeSpec,
        warm_start_node: NodeSpec,
        metadata: dict[str, Any] | None = None,
        config: ResearchRunConfig | None = None,
    ) -> ResearchRunResult:
        root = Path(project_dir).resolve()
        if not root.exists():
            raise FileNotFoundError(f"project repo not found: {root}")
        round_id = round_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = _runtime_paths(root, round_id)
        paths.round_dir.mkdir(parents=True, exist_ok=True)
        paths.latest_warm_start.parent.mkdir(parents=True, exist_ok=True)

        metadata = _load_metadata(root, metadata)
        context = load_project_research_context(root, metadata)
        _write_text(paths.context_snapshot, json.dumps(context, ensure_ascii=False, indent=2, default=str))

        panel_nodes = [node if isinstance(node, NodeSpec) else NodeSpec.from_dict(node) for node in panel_nodes]
        analyst_results = self.run_panel("analyst", _nodes_for_role(panel_nodes, NodeRole.ANALYST), context, paths.round_dir)
        hypothesis_context = {**context, "analyst_outputs": [result.to_panel_result() for result in analyst_results]}
        hypothesis_results = self.run_panel(
            "hypothesis",
            _nodes_for_role(panel_nodes, NodeRole.HYPOTHESIS),
            hypothesis_context,
            paths.round_dir,
        )
        critic_context = {
            **hypothesis_context,
            "hypothesis_outputs": [result.to_panel_result() for result in hypothesis_results],
        }
        critic_results = self.run_panel("critic", _nodes_for_role(panel_nodes, NodeRole.CRITIC), critic_context, paths.round_dir)

        selector_context = {
            **critic_context,
            "critic_outputs": [result.to_panel_result() for result in critic_results],
        }
        selector_result = self.run_single_node(
            selector_node,
            selector_context,
            paths.selector_decision,
            default_max_tokens=DEFAULT_SELECTOR_MAX_TOKENS,
        )

        warm_start_context = {**selector_context, "selector_decision": selector_result.content}
        warm_start_result = self.run_single_node(
            warm_start_node,
            warm_start_context,
            paths.warm_start,
            default_max_tokens=DEFAULT_WARM_START_MAX_TOKENS,
        )
        shutil.copyfile(paths.warm_start, paths.latest_warm_start)

        node_specs = [*panel_nodes, selector_node, warm_start_node]
        run_config = config or ResearchRunConfig(
            project_dir=str(root),
            round_id=round_id,
            node_specs=node_specs,
            metadata=metadata,
        )
        return ResearchRunResult(
            config=run_config,
            analyst_results=analyst_results,
            hypothesis_results=hypothesis_results,
            critic_results=critic_results,
            selection=None,
            warm_start=None,
            artifacts=[
                _artifact(paths.context_snapshot, root, "context", "Research context snapshot"),
                _artifact(paths.selector_decision, root, "selector", "Selector decision markdown"),
                _artifact(paths.warm_start, root, "warm_start", "Round warm-start instructions"),
                _artifact(paths.latest_warm_start, root, "latest_warm_start", "Latest warm-start copy"),
            ],
        )

    def run_panel(
        self,
        name: str,
        nodes: list[NodeSpec],
        context: dict[str, Any],
        out_dir: Path,
    ) -> list[NodeResult]:
        if not nodes:
            return []
        results: list[NodeResult] = []
        for node in nodes:
            result = self._call_node(node, context, default_max_tokens=DEFAULT_NODE_MAX_TOKENS)
            artifact_path = out_dir / f"{name}_{safe_model_name(node.model)}.md"
            _write_text(artifact_path, result.content)
            results.append(result)
        if results and all(result.content.startswith("ERROR from ") for result in results):
            raise RuntimeError(f"All models failed in research node: {name}")
        return results

    def run_single_node(
        self,
        node: NodeSpec,
        context: dict[str, Any],
        artifact_path: Path,
        *,
        default_max_tokens: int,
    ) -> NodeResult:
        result = self._call_node(node, context, default_max_tokens=default_max_tokens)
        if result.error:
            raise RuntimeError(f"Required research node failed: {node.role.value} {node.model}: {result.error}")
        _write_text(artifact_path, result.content)
        return result

    def _call_node(self, node: NodeSpec, context: dict[str, Any], *, default_max_tokens: int) -> NodeResult:
        max_tokens = default_max_tokens if node.max_tokens == DEFAULT_NODE_MAX_TOKENS else node.max_tokens
        try:
            content = self.client.chat(
                node.model,
                research_messages(node.role.value, context, node.instruction),
                max_tokens=max_tokens,
            )
            error = ""
        except Exception as exc:  # noqa: BLE001 - legacy behavior records per-model failures as content.
            error = f"{type(exc).__name__}: {exc}"
            content = f"ERROR from {node.model}: {error}"
        return NodeResult(
            role=node.role,
            model=node.model,
            content=content,
            stage=node.stage,
            name=node.name,
            error=error,
            metadata={"max_tokens": max_tokens},
        )


def safe_model_name(model: str) -> str:
    safe_model = "".join(ch if ch.isalnum() else "_" for ch in model).strip("_")
    return safe_model or "model"


def _runtime_paths(root: Path, round_id: str) -> ResearchRuntimePaths:
    round_dir = (root / "notes" / "research_rounds" / _safe_round_id(round_id)).resolve()
    _ensure_inside(root, round_dir)
    latest_warm_start = (root / "notes" / "latest_research_warm_start.txt").resolve()
    _ensure_inside(root, latest_warm_start)
    return ResearchRuntimePaths(
        round_dir=round_dir,
        context_snapshot=round_dir / "context_snapshot.json",
        selector_decision=round_dir / "selector_decision.md",
        warm_start=round_dir / "warm_start_for_react.txt",
        latest_warm_start=latest_warm_start,
    )


def _safe_round_id(round_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in round_id).strip("._-")
    if not safe:
        raise ValueError("round_id must contain at least one safe character")
    return safe[:120]


def _ensure_inside(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("research artifact path must stay inside project_dir") from exc


def _load_metadata(root: Path, metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is not None:
        return metadata
    metadata_path = root / "auto_dacon_task.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _nodes_for_role(nodes: list[NodeSpec], role: NodeRole) -> list[NodeSpec]:
    return [node for node in nodes if node.role == role]


def _single_node(nodes: list[NodeSpec], role: NodeRole) -> NodeSpec:
    matches = _nodes_for_role(nodes, role)
    if len(matches) != 1:
        raise ValueError(f"exactly one {role.value} node is required")
    return matches[0]


def _artifact(path: Path, root: Path, kind: str, description: str) -> ResearchArtifact:
    return ResearchArtifact(path=str(path.relative_to(root)), kind=kind, description=description)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


_PANEL_ROLES = {NodeRole.ANALYST, NodeRole.HYPOTHESIS, NodeRole.CRITIC}
