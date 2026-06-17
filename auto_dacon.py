import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
DEFAULT_DATA_ROOT = Path("data") / "dacon"
DEFAULT_WORKSPACE = Path("workspace_dacon")
DEFAULT_OUTPUT_ROOT = Path("outputs") / "dacon"
DEFAULT_LLM = "openrouter/qwen37_plus"
DEFAULT_CODE_LLM = "openrouter/claude_sonnet_46"
DEFAULT_RAG_PATH = Path("C:/Auto_Dacon_RAG/kaggle_cases_db")
AIDE_RAG_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
PROJECT_RUNTIME_DIRNAME = ".auto_dacon_runtime"
DEFAULT_REACT_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_RESEARCH_ANALYST_MODELS = (
    "deepseek/deepseek-v4-pro",
    "google/gemini-3.5-flash",
    "z-ai/glm-5.1",
)
DEFAULT_RESEARCH_HYPOTHESIS_MODELS = (
    "anthropic/claude-sonnet-4.6",
    "deepseek/deepseek-v4-pro",
    "moonshotai/kimi-k2.7-code",
)
DEFAULT_RESEARCH_CRITIC_MODELS = (
    "anthropic/claude-sonnet-4.6",
    "deepseek/deepseek-v4-pro",
    "google/gemini-3.5-flash",
)
DEFAULT_RESEARCH_SELECTOR_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_RESEARCH_WARM_START_MODEL = "anthropic/claude-sonnet-4.6"

from auto_dacon.research.context import (  # noqa: E402
    compact_dataframe_profile as _compact_dataframe_profile,
    compact_research_context_for_prompt as _compact_research_context_for_prompt,
    load_project_research_context as _load_project_research_context,
    read_text_if_exists as _read_text_if_exists,
)
from auto_dacon.research.prompts import research_messages as _research_messages  # noqa: E402
from auto_dacon.research.openrouter import (  # noqa: E402
    OpenRouterClient,
    openrouter_chat as _packaged_openrouter_chat,
)
from auto_dacon.research.runtime import ResearchRuntime  # noqa: E402
from auto_dacon.research.schemas import NodeRole, NodeSpec  # noqa: E402


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def infer_task_id(competition_url: str) -> str:
    path_parts = [part for part in urlparse(competition_url).path.split("/") if part]
    numeric_ids = [part for part in path_parts if part.isdigit()]
    if numeric_ids:
        return f"dacon_{numeric_ids[-1]}"
    return "dacon_competition"


def ensure_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _is_default_path(value: str | Path | None, default: Path) -> bool:
    if value is None:
        return True
    return Path(value) == default or str(value) == str(default)


def project_runtime_dir(project_dir: Path) -> Path:
    return project_dir / PROJECT_RUNTIME_DIRNAME


def auto_dacon_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def project_run_cwd(project_dir: Path) -> Path:
    run_cwd = project_runtime_dir(project_dir) / "cwd"
    run_cwd.mkdir(parents=True, exist_ok=True)
    return run_cwd


def add_auto_dacon_pythonpath(env: dict[str, str]) -> None:
    paths = [ROOT, ROOT / "src"]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in paths) + (
        os.pathsep + existing if existing else ""
    )


def normalize_project_gitignore(project_dir: Path) -> None:
    gitignore = project_dir / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    removed = {"outputs/*.csv", "outputs/", "data/"}
    lines = [line for line in existing if line.strip().replace("\\", "/") not in removed]
    required = [".env", "__pycache__/", f"{PROJECT_RUNTIME_DIRNAME}/"]
    for line in required:
        if line not in lines:
            lines.append(line)
    write_text_file(gitignore, "\n".join(lines).rstrip() + "\n")


def project_args_from_repo(args: argparse.Namespace) -> argparse.Namespace:
    project_dir = ensure_file(args.project_dir, "project repo")
    normalize_project_gitignore(project_dir)
    metadata_path = project_dir / "auto_dacon_task.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"auto_dacon_task.json not found in {project_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    data_dir = project_dir / "data"
    train = data_dir / "train.csv"
    test = data_dir / "test.csv"
    sample = data_dir / "sample_submission.csv"
    for path, label in [(train, "train.csv"), (test, "test.csv"), (sample, "sample_submission.csv")]:
        ensure_file(path, label)

    forwarded = argparse.Namespace(**vars(args))
    forwarded.competition_url = metadata["competition_url"]
    forwarded.train = str(train)
    forwarded.test = str(test)
    forwarded.sample_submission = str(sample)
    forwarded.task_id = args.task_id or metadata.get("task_id")
    forwarded.id_column = args.id_column or metadata.get("id_column")
    forwarded.target_column = args.target_column or metadata.get("target_column")
    forwarded.metric = args.metric or metadata.get("metric") or "MAE"
    forwarded.project_dir = str(project_dir)
    runtime_dir = project_runtime_dir(project_dir)
    if _is_default_path(getattr(args, "data_root", None), DEFAULT_DATA_ROOT):
        forwarded.data_root = str(runtime_dir / "data")
    if _is_default_path(getattr(args, "output_root", None), DEFAULT_OUTPUT_ROOT):
        forwarded.output_root = str(runtime_dir / "outputs")
    if _is_default_path(getattr(args, "workspace_name", None), DEFAULT_WORKSPACE):
        forwarded.workspace_name = str(runtime_dir / "workspace")
    if not getattr(forwarded, "competition_context", None):
        context_candidates = [
            project_dir / "competition_context.md",
            project_dir / "notes" / "competition_context.md",
            project_dir / "notes" / "competition.md",
        ]
        for context_path in context_candidates:
            if context_path.exists():
                forwarded.competition_context = str(context_path)
                break
    return forwarded


def copy_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            dst.chmod(0o666)
        except OSError:
            pass
        dst.unlink()
    shutil.copy2(src, dst)


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            path.chmod(0o666)
        except OSError:
            pass
    path.write_text(content, encoding="utf-8")


def venv_python_path(venv_dir: str | Path) -> Path:
    return Path(venv_dir) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_aide_rag_metadata(rag_path: str | Path) -> None:
    rag_root = Path(rag_path)
    kaggle_db = rag_root / "kaggle_db"
    metadata_path = rag_root / "metadata.json"
    if metadata_path.exists() or not (kaggle_db / "index.faiss").exists():
        return
    metadata = {
        "databases": {
            "kaggle_db": {
                "embedded_field": "code",
                "model_name": AIDE_RAG_EMBEDDING_MODEL,
                "template_path": "data_preprocessing/data_map/code",
                "stage": "input",
                "modality": "tab",
            }
        }
    }
    write_text_file(metadata_path, json.dumps(metadata, indent=2))


def profile_csv(path: Path, nrows: int = 5000) -> dict:
    import pandas as pd

    preview = pd.read_csv(path, nrows=nrows)
    columns = list(preview.columns)
    dtypes = {col: str(dtype) for col, dtype in preview.dtypes.items()}
    missing_ratio = preview.isna().mean().round(4).to_dict()
    return {
        "file": path.name,
        "columns": columns,
        "dtypes": dtypes,
        "sample_rows_profiled": int(len(preview)),
        "missing_ratio_sample": missing_ratio,
    }


def make_descriptions(
        competition_url: str,
        train_profile: dict,
        test_profile: dict,
        sample_profile: dict,
        target_column: str,
        id_column: str,
        metric: str,
        competition_context: str | None = None,
) -> tuple[str, str, str]:
    common_train_cols = set(train_profile["columns"])
    common_test_cols = set(test_profile["columns"])
    shared_feature_cols = sorted((common_train_cols & common_test_cols) - {id_column})
    train_only_cols = sorted(common_train_cols - common_test_cols)
    test_only_cols = sorted(common_test_cols - common_train_cols)
    context_block = (
        competition_context.strip()
        if competition_context
        else "No additional competition context file was provided."
    )

    task_description = f"""DACON competition URL: {competition_url}

Competition context supplied by the project:
{context_block}

Task:
- Build a predictive model from the provided DACON train.csv and test.csv files.
- The target column is `{target_column}`.
- The ID column is `{id_column}` and must be preserved in the final submission.

Final output requirement:
- Create a CSV submission with the same columns and row order as sample_submission.csv.
- The prediction column(s) must match sample_submission.csv exactly.
- Do not add helper columns, drop rows, reorder rows, or leak information from test labels.
"""

    data_description = f"""Data files:
- train.csv: training rows with target `{target_column}`.
- test.csv: test rows without the target column.
- sample_submission.csv: required submission format.

Train columns ({len(train_profile["columns"])}):
{", ".join(train_profile["columns"])}

Test columns ({len(test_profile["columns"])}):
{", ".join(test_profile["columns"])}

Sample submission columns:
{", ".join(sample_profile["columns"])}

Column notes:
- `{id_column}` is the row identifier.
- Shared train/test feature columns ({len(shared_feature_cols)}):
{", ".join(shared_feature_cols)}
- Train-only columns ({len(train_only_cols)}):
{", ".join(train_only_cols)}
- Test-only columns ({len(test_only_cols)}):
{", ".join(test_only_cols)}
- Treat object/string columns as categorical or text features as appropriate.
- Missing values may be present and should be handled robustly.
"""

    metric_description = f"""Evaluation metric: {metric}.

The public leaderboard uses a subset of the test data, and the private leaderboard uses the
remaining hidden test data. Optimize validation for the official metric and avoid leakage from IDs, row order,
or any information unavailable at prediction time.
"""

    return task_description, data_description, metric_description


def prepare_task(args: argparse.Namespace) -> Path:
    task_id = args.task_id or infer_task_id(args.competition_url)
    if getattr(args, "project_dir", None) and _is_default_path(getattr(args, "data_root", None), DEFAULT_DATA_ROOT):
        data_root = project_runtime_dir(ensure_file(args.project_dir, "project repo")) / "data"
    else:
        data_root = Path(args.data_root)
    task_dir = data_root / task_id

    train_path = ensure_file(args.train, "train.csv")
    test_path = ensure_file(args.test, "test.csv")
    sample_path = ensure_file(args.sample_submission, "sample_submission.csv")

    train_profile = profile_csv(train_path)
    test_profile = profile_csv(test_path)
    sample_profile = profile_csv(sample_path)

    id_column = args.id_column or sample_profile["columns"][0]
    target_column = args.target_column or sample_profile["columns"][1]
    metric = args.metric or "MAE"
    competition_context = None
    if getattr(args, "competition_context", None):
        competition_context = ensure_file(args.competition_context, "competition context").read_text(encoding="utf-8")

    task_dir.mkdir(parents=True, exist_ok=True)
    copy_csv(train_path, task_dir / "train.csv")
    copy_csv(test_path, task_dir / "test.csv")
    copy_csv(sample_path, task_dir / "sample_submission.csv")

    raw_task, raw_data, raw_metric = make_descriptions(
        competition_url=args.competition_url,
        train_profile=train_profile,
        test_profile=test_profile,
        sample_profile=sample_profile,
        target_column=target_column,
        id_column=id_column,
        metric=metric,
        competition_context=competition_context,
    )
    write_text_file(task_dir / "raw_task_description.txt", raw_task)
    write_text_file(task_dir / "raw_data_description.txt", raw_data)
    write_text_file(task_dir / "raw_metric_description.txt", raw_metric)

    metadata = {
        "task_id": task_id,
        "competition_url": args.competition_url,
        "id_column": id_column,
        "target_column": target_column,
        "metric": metric,
        "competition_context_file": (
            str(args.competition_context)
            if getattr(args, "competition_context", None)
            else None
        ),
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(task_dir),
        "train_profile": train_profile,
        "test_profile": test_profile,
        "sample_submission_profile": sample_profile,
    }
    write_text_file(task_dir / "auto_dacon_metadata.json", json.dumps(metadata, indent=2))

    if args.project_dir:
        write_project_scaffold(Path(args.project_dir), metadata)

    print(f"Prepared DACON local task: {task_dir}")
    return task_dir


def write_project_scaffold(project_dir: Path, metadata: dict) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "outputs").mkdir(exist_ok=True)
    (project_dir / "notes").mkdir(exist_ok=True)
    normalize_project_gitignore(project_dir)
    write_text_file(project_dir / "auto_dacon_task.json", json.dumps(metadata, indent=2))
    readme = project_dir / "README.md"
    if not readme.exists():
        write_text_file(
            readme,
            f"# {metadata['task_id']}\n\n"
            "This repository stores notes and outputs for a DACON competition run driven by Auto_Dacon.\n\n"
            f"- Competition: {metadata['competition_url']}\n"
            f"- Metric: {metadata['metric']}\n"
            f"- Target: `{metadata['target_column']}`\n",
        )


def run_agent(args: argparse.Namespace) -> None:
    task_dir = prepare_task(args)
    task_id = args.task_id or infer_task_id(args.competition_url)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    add_auto_dacon_pythonpath(env)
    for key, value in read_env_file(ROOT / ".env").items():
        env.setdefault(key, value)
    if args.project_dir:
        for key, value in read_env_file(Path(args.project_dir) / ".env").items():
            env.setdefault(key, value)
    env.setdefault("AUTO_DACON_LLM_TIMEOUT", "120")
    env.setdefault("AUTO_DACON_ENABLE_HYPEROPT", "1" if args.enable_hyperopt else "0")
    if args.openrouter_api_key:
        env["OPENROUTER_API_KEY"] = args.openrouter_api_key
    if "OPENROUTER_API_KEY" not in env:
        raise RuntimeError("OPENROUTER_API_KEY is required. Pass --openrouter-api-key or set the environment variable.")

    cmd = [
        sys.executable,
        str(ROOT / "run_complete_pipeline.py"),
        "--task_id", task_id,
        "--llm", args.llm,
        "--code_llm", args.code_llm or args.llm,
        "--total_time", str(args.total_time),
        "--max_time_per_submission", str(args.max_time_per_submission),
        "--workspace_name", str(args.workspace_name),
        "--max_cpu", str(args.max_cpu),
        "--max_setups", str(args.max_setups),
        "--blend_after_n", str(args.blend_after_n),
        "--alt_raw_data_root", str(Path(args.data_root)),
        "--tabular_task",
        "--is_local_task",
    ]
    if args.enable_agent_rag:
        cmd.append("--enable_agent_rag")
        cmd.extend(["--agent_rag_k", str(args.agent_rag_k)])
        if args.agent_rag_path:
            ensure_aide_rag_metadata(args.agent_rag_path)
            cmd.extend(["--agent_rag_path", args.agent_rag_path])

    print("Running Auto_Dacon Agent_K pipeline:")
    print(" ".join(cmd))
    run_cwd = project_run_cwd(Path(args.project_dir)) if args.project_dir else Path.cwd()
    subprocess.run(cmd, cwd=run_cwd, env=env, check=True)
    collect_submission(
        task_id=task_id,
        workspace_name=Path(args.workspace_name),
        output_root=Path(args.output_root),
        project_dir=Path(args.project_dir) if args.project_dir else None,
    )
    print(f"Input task directory: {task_dir}")


def run_project(args: argparse.Namespace) -> None:
    run_agent(project_args_from_repo(args))


def submission_priority(path: Path) -> tuple[int, float]:
    name = path.name.lower()
    full = str(path).lower()
    if "valid" in name:
        return (99, -path.stat().st_mtime)
    if "final_test_predictions" in full and "bagged_then_blended" in name:
        return (0, -path.stat().st_mtime)
    if "final_test_predictions" in full and "last_blend" in name:
        return (1, -path.stat().st_mtime)
    if "final_test_predictions" in full:
        return (2, -path.stat().st_mtime)
    if "post_scaffold" in full:
        return (3, -path.stat().st_mtime)
    if "main_pipeline" in full:
        return (4, -path.stat().st_mtime)
    if "submission_bagged_test" in name:
        return (5, -path.stat().st_mtime)
    return (10, -path.stat().st_mtime)


def next_numbered_submission_path(project_outputs: Path, prefix: str) -> Path:
    existing = []
    for path in project_outputs.glob(f"{prefix}*.csv"):
        suffix = path.stem.removeprefix(prefix)
        if suffix == "":
            existing.append(1)
        elif suffix.isdigit():
            existing.append(int(suffix))
    next_idx = max(existing, default=0) + 1
    return project_outputs / f"{prefix}{next_idx}.csv"


def mirror_submission_to_project_outputs(src: Path, project_dir: Path, kind: str) -> Path:
    project_outputs = project_dir / "outputs"
    project_outputs.mkdir(parents=True, exist_ok=True)

    if kind == "baseline":
        archived = project_outputs / "submission_baseline.csv"
        if archived.exists():
            archived = next_numbered_submission_path(project_outputs, "submission_baseline")
    elif kind == "react":
        archived = next_numbered_submission_path(project_outputs, "submission_react")
    else:
        safe_kind = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in kind).strip("_")
        archived = next_numbered_submission_path(project_outputs, f"submission_{safe_kind}")

    shutil.copy2(src, archived)
    return archived


def collect_submission(
        task_id: str,
        workspace_name: Path,
        output_root: Path,
        project_dir: Path | None = None,
) -> Path | None:
    workspace = workspace_name if workspace_name.is_absolute() else ROOT / workspace_name
    patterns = [
        workspace / task_id / "seed_*" / "ramp_kit_v*" / "final_test_predictions" / "*.csv",
        workspace
        / task_id
        / "seed_*"
        / "ramp_kit_v*"
        / "submissions"
        / "training_output"
        / "submission_bagged_then_blended_test.csv",
        workspace
        / task_id
        / "seed_*"
        / "ramp_kit_v*"
        / "submissions"
        / "starting_kit"
        / "training_output"
        / "submission_bagged_test.csv",
        workspace
        / task_id
        / "seed_*"
        / "final_unit_test_vtest_n0"
        / "submissions"
        / "starting_kit"
        / "training_output"
        / "submission_bagged_test.csv",
        workspace / task_id / "seed_*" / "main_pipeline" / "**" / "submission.csv",
        workspace / task_id / "post_scaffold" / "**" / "submission*.csv",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path(p) for p in workspace.glob(str(pattern.relative_to(workspace))) if Path(p).is_file())

    if not candidates:
        print("No submission candidate found yet.")
        return None

    non_valid_candidates = [p for p in candidates if "valid" not in p.name.lower()]
    if non_valid_candidates:
        candidates = non_valid_candidates
    candidates.sort(key=submission_priority)
    best = candidates[0]
    out_dir = output_root / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission.csv"
    shutil.copy2(best, out_path)
    if project_dir is not None:
        archived = mirror_submission_to_project_outputs(best, project_dir, kind="baseline")
        print(f"Mirrored submission to project outputs: {archived}")
    print(f"Collected submission: {out_path}")
    return out_path


def collect_react_submission(project_dir: Path, post_scaffold_root: Path) -> Path | None:
    candidates = [
        path
        for pattern in [
            "**/best_submissions/submission_*.csv",
            "**/intermediate_best_submissions/submission_*.csv",
            "**/draft_submissions/submission_*.csv",
            "**/submission/submission.csv",
        ]
        for path in post_scaffold_root.glob(pattern)
        if path.is_file()
    ]
    if not candidates:
        print("No post-scaffold/ReAct submission candidate found yet.")
        return None
    candidates.sort(key=lambda path: (0 if "best_submissions" in str(path).lower() else 1, -path.stat().st_mtime))
    archived = mirror_submission_to_project_outputs(candidates[0], project_dir, kind="react")
    print(f"Mirrored post-scaffold/ReAct submission to project outputs: {archived}")
    return archived


def default_warm_start_summary(project_dir: Path, metadata: dict) -> Path:
    notes_dir = project_dir / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    warm_start = notes_dir / "agentk_warm_start_summary.txt"

    score_summary = notes_dir / "score_summary.md"
    latest_score = notes_dir / "latest_score.md"
    parts = [
        f"Task: {metadata.get('task_id', 'unknown')}",
        f"Metric: {metadata.get('metric', 'MAE')}, lower is better.",
        f"Target column: {metadata.get('target_column')}",
        f"Submission id column: {metadata.get('id_column')}",
        "",
        "Previous Auto_Dacon / Agent_K-style notes:",
    ]
    if latest_score.exists():
        parts.extend(["", "Latest recorded score:", latest_score.read_text(encoding="utf-8")])
    if score_summary.exists():
        parts.extend(["", "Score summary:", score_summary.read_text(encoding="utf-8")])
    if not latest_score.exists() and not score_summary.exists():
        parts.append("- No prior public score has been recorded yet.")
    parts.extend(
        [
            "",
            "Guidance for post-scaffold/ReAct:",
            "- Do not submit automatically.",
            "- Generate a valid submission.csv matching sample_submission.csv exactly.",
            "- Keep 5-fold validation by default for tabular regression.",
            "- Focus on feature engineering, robust preprocessing, validation, and model/ensemble improvements.",
            "- Avoid expensive hyperparameter search unless explicitly enabled by the user.",
            "- On Windows, avoid GPU-only assumptions. If CatBoost is used, prefer CPU "
            "or set an ASCII train_dir; if a model fails, still write a valid "
            "submission.csv from the successful folds/models.",
        ]
    )
    write_text_file(warm_start, "\n".join(parts).rstrip() + "\n")
    return warm_start


def read_text_if_exists(path: Path, max_chars: int = 12000) -> str:
    return _read_text_if_exists(path, max_chars)


def compact_dataframe_profile(path: Path, target_column: str | None = None) -> dict[str, Any]:
    return _compact_dataframe_profile(path, target_column)


def load_project_research_context(project_dir: Path, metadata: dict) -> dict[str, Any]:
    return _load_project_research_context(project_dir, metadata)


def compact_research_context_for_prompt(context: dict[str, Any]) -> dict[str, Any]:
    return _compact_research_context_for_prompt(context)


def research_messages(role: str, context: dict[str, Any], extra: str) -> list[dict[str, str]]:
    return _research_messages(role, context, extra)


def openrouter_chat(model: str, messages: list[dict[str, str]], api_key: str, max_tokens: int = 2200) -> str:
    return _packaged_openrouter_chat(model, messages, api_key, max_tokens=max_tokens)


def run_research_panel(
        name: str,
        models: tuple[str, ...],
        context: dict[str, Any],
        instruction: str,
        api_key: str,
        out_dir: Path,
) -> list[dict[str, str]]:
    nodes = []
    for model in models:
        print(f"Running research node {name}: {model}")
        nodes.append(NodeSpec(role=NodeRole(name), model=model, instruction=instruction))
    results = ResearchRuntime(OpenRouterClient(api_key=api_key)).run_panel(name, nodes, context, out_dir)
    return [result.to_panel_result() for result in results]


def build_research_warm_start(project_dir: Path, args: argparse.Namespace) -> Path:
    project_dir = ensure_file(project_dir, "project repo")
    normalize_project_gitignore(project_dir)
    metadata_path = project_dir / "auto_dacon_task.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"auto_dacon_task.json not found in {project_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    env = os.environ.copy()
    for key, value in read_env_file(ROOT / ".env").items():
        env.setdefault(key, value)
    for key, value in read_env_file(project_dir / ".env").items():
        env.setdefault(key, value)
    if args.openrouter_api_key:
        env["OPENROUTER_API_KEY"] = args.openrouter_api_key
    api_key = env.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required. Pass --openrouter-api-key or set the environment variable.")

    analyst_instruction = (
        "Analyze the latest public-score trend, CV/public mismatch risk, data profile, "
        "and project notes. Identify why recent attempts improved or failed. "
        "Return concise findings and experiment priorities."
    )
    hypothesis_instruction = (
        "Generate 5 to 8 concrete experiment hypotheses. Focus first on EDA-driven "
        "preprocessing and feature engineering, then model/ensemble fit. For each "
        "hypothesis include rationale, implementation sketch, validation method, "
        "expected effect, runtime cost, and leakage risk."
    )
    critic_instruction = (
        "Critique the proposed hypotheses aggressively. Reject leakage-prone, "
        "public-overfit, duplicate, too-expensive, or weakly evidenced ideas. "
        "Keep only robust experiments likely to improve private leaderboard."
    )
    selector_instruction = (
        "Select the best 1 to 2 hypotheses for the next ReAct run. "
        "Give a strict decision: accepted hypotheses, rejected hypotheses, "
        "validation guardrails, and exact implementation priorities. "
        "Do not request hyperparameter search unless the user explicitly enabled it."
    )
    warm_start_instruction = (
        "Write the final warm-start instruction for Auto_Dacon run-react-project. "
        "It must be directly actionable for a coding agent. Include competition "
        "goal, current best public score if known, selected hypotheses, validation "
        "protocol, leakage warnings, output requirement, and a fallback plan that "
        "still writes a valid submission.csv. Keep it focused."
    )
    round_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    panel_nodes = [
        NodeSpec(role=NodeRole.ANALYST, model=model, instruction=analyst_instruction)
        for model in DEFAULT_RESEARCH_ANALYST_MODELS
    ]
    panel_nodes.extend(
        NodeSpec(role=NodeRole.HYPOTHESIS, model=model, instruction=hypothesis_instruction)
        for model in DEFAULT_RESEARCH_HYPOTHESIS_MODELS
    )
    panel_nodes.extend(
        NodeSpec(role=NodeRole.CRITIC, model=model, instruction=critic_instruction)
        for model in DEFAULT_RESEARCH_CRITIC_MODELS
    )
    result = ResearchRuntime(OpenRouterClient(api_key=api_key)).run_round(
        project_dir=project_dir,
        round_id=round_id,
        panel_nodes=panel_nodes,
        selector_node=NodeSpec(
            role=NodeRole.SELECTOR,
            model=DEFAULT_RESEARCH_SELECTOR_MODEL,
            instruction=selector_instruction,
            max_tokens=2600,
        ),
        warm_start_node=NodeSpec(
            role=NodeRole.WARM_START_BUILDER,
            model=DEFAULT_RESEARCH_WARM_START_MODEL,
            instruction=warm_start_instruction,
            max_tokens=3000,
        ),
        metadata=metadata,
    )
    warm_start_path = project_dir / next(artifact.path for artifact in result.artifacts if artifact.kind == "warm_start")
    print(f"Research warm-start written: {warm_start_path}")
    return warm_start_path


def run_research_next(args: argparse.Namespace) -> None:
    warm_start = build_research_warm_start(Path(args.project_dir), args)
    if not args.run_react:
        print(f"Use this with run-react-project --warm-start {warm_start}")
        return
    react_args = argparse.Namespace(
        project_dir=args.project_dir,
        task_id=args.task_id,
        openrouter_api_key=args.openrouter_api_key,
        react_venv=args.react_venv,
        model=args.react_model,
        feedback_model=args.feedback_model,
        total_time=args.total_time,
        exec_timeout=args.exec_timeout,
        steps=args.steps,
        top_n=args.top_n,
        warm_start=str(warm_start),
        enable_rag=args.enable_rag,
        rag_path=args.rag_path,
        skip_research_loop=False,
    )
    run_react_project(react_args)


def run_react_project(args: argparse.Namespace) -> None:
    project_dir = ensure_file(args.project_dir, "project repo")
    normalize_project_gitignore(project_dir)
    metadata_path = project_dir / "auto_dacon_task.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"auto_dacon_task.json not found in {project_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    task_id = args.task_id or metadata.get("task_id")
    if not task_id:
        raise ValueError("task_id is required in auto_dacon_task.json or --task-id.")

    data_dir = project_dir / "data"
    for filename in ["train.csv", "test.csv", "sample_submission.csv"]:
        ensure_file(data_dir / filename, filename)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    add_auto_dacon_pythonpath(env)
    for key, value in read_env_file(ROOT / ".env").items():
        env.setdefault(key, value)
    for key, value in read_env_file(project_dir / ".env").items():
        env.setdefault(key, value)
    if args.openrouter_api_key:
        env["OPENROUTER_API_KEY"] = args.openrouter_api_key
    if "OPENROUTER_API_KEY" not in env:
        raise RuntimeError("OPENROUTER_API_KEY is required. Pass --openrouter-api-key or set the environment variable.")

    aide_path = ROOT / "third_party" / "aideml"
    if not (aide_path / "aide").exists():
        raise FileNotFoundError(f"AIDE package not found: {aide_path}")
    env["PYTHONPATH"] = str(aide_path) + os.pathsep + env.get("PYTHONPATH", "")

    react_python = venv_python_path(auto_dacon_path(args.react_venv))
    if not react_python.exists():
        raise FileNotFoundError(
            f"ReAct venv python not found: {react_python}. Run `python auto_dacon.py bootstrap-react` first."
        )

    post_scaffold_root = project_dir / PROJECT_RUNTIME_DIRNAME / "post_scaffold"
    if args.warm_start:
        warm_start = ensure_file(args.warm_start, "warm-start file")
    elif getattr(args, "skip_research_loop", False):
        warm_start = default_warm_start_summary(project_dir, metadata)
    else:
        warm_start = build_research_warm_start(project_dir, args)
    if args.enable_rag:
        rag_path = Path(args.rag_path).expanduser().resolve()
        ensure_aide_rag_metadata(rag_path)
    else:
        rag_path = None

    cmd = [
        str(react_python),
        "-m", "aide.run",
        f"data_dir={data_dir.as_posix()}",
        f"exp_name={task_id}",
        f"top_n={args.top_n}",
        f"agent.time_limit={args.total_time}",
        f"agent.steps={args.steps}",
        f"exec.timeout={args.exec_timeout}",
        "copy_data=true",
        f"workspace_dir={post_scaffold_root.as_posix()}",
        f"agent.code.model={args.model}",
        f"agent.feedback.model={args.feedback_model or args.model}",
        "agent.k_fold_validation=5",
        "agent.use_agent_k_warm_start=true",
        f"agent.agent_k_submissions={warm_start.as_posix()}",
        f"agent.use_rag={'true' if args.enable_rag else 'false'}",
    ]
    if rag_path is not None:
        cmd.append(f"agent.rag_path={rag_path.as_posix()}")

    print("Running Auto_Dacon post-scaffold/ReAct:")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, cwd=project_run_cwd(project_dir), env=env, check=True)
    except subprocess.CalledProcessError:
        recovered = collect_react_submission(project_dir, post_scaffold_root)
        if recovered is not None:
            print("Recovered a valid ReAct submission despite a post-run wrapper error.")
            return
        raise
    collect_react_submission(project_dir, post_scaffold_root)


def record_score(args: argparse.Namespace) -> None:
    task_id = args.task_id
    if getattr(args, "project_dir", None):
        project_dir = ensure_file(args.project_dir, "project repo")
        normalize_project_gitignore(project_dir)
        notes_dir = project_dir / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "task_id": task_id,
            "public_score": args.public_score,
            "private_score": args.private_score,
            "score_direction": "lower_is_better",
            "metric": args.metric,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "notes": args.notes or "",
        }
        with (notes_dir / "score_history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        latest = (
            f"Task: {task_id}\n"
            f"Metric: {args.metric} lower is better\n"
            f"Public score: {args.public_score}\n"
            f"Private score: {args.private_score}\n"
            f"Notes: {args.notes or ''}\n"
        )
        write_text_file(notes_dir / "latest_score.md", latest)
        print(f"Recorded score in project repo: {notes_dir}")
        return

    if not args.experience_root:
        raise ValueError(
            "record-score must use --project-dir for competition-specific notes, "
            "or an explicit external --experience-root for curated RAG experience."
        )

    exp_dir = Path(args.experience_root) / task_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "task_id": task_id,
        "public_score": args.public_score,
        "private_score": args.private_score,
        "score_direction": "lower_is_better",
        "metric": args.metric,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "notes": args.notes or "",
    }
    (exp_dir / "score.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    summary = (
        f"Task: {task_id}\n"
        f"Metric: {args.metric} lower is better\n"
        f"Public score: {args.public_score}\n"
        f"Private score: {args.private_score}\n"
        f"Notes: {args.notes or ''}\n"
    )
    (exp_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(f"Recorded score experience: {exp_dir}")


def build_aide_rag(args: argparse.Namespace) -> None:
    aide_dir = ROOT / "third_party" / "aideml"
    rag_path = Path(args.rag_path)
    if not (aide_dir / "aide").exists():
        shutil.unpack_archive(ROOT / "third_party" / "aideml.zip", ROOT / "third_party")
    cases_zip = aide_dir / "kaggle_cases.zip"
    cases_dir = aide_dir / "kaggle_cases"
    if cases_zip.exists() and not cases_dir.exists():
        shutil.unpack_archive(cases_zip, aide_dir)
    shim_path = aide_dir / "aide" / "utils" / "langchain_huggingface.py"
    if not shim_path.exists():
        shim_path.write_text(
            "from langchain_community.embeddings import HuggingFaceEmbeddings\n",
            encoding="utf-8",
        )
    cmd = [
        sys.executable,
        str(aide_dir / "aide" / "utils" / "db_faiss.py"),
        "--populate",
        "--data_path", str(cases_dir),
        "--rag_path", str(rag_path),
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    ensure_aide_rag_metadata(rag_path)
    print(f"Built AIDE Kaggle-cases RAG DB: {rag_path}")


def patch_ramp_hyperopt_for_windows() -> None:
    """Apply small local-portability patches after extracting Agent_K archives."""
    ramp_root = ROOT / "third_party" / "ramp-hyperopt"

    def replace_if_present(path: Path, old: str, new: str, marker: str) -> None:
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
        if marker in text:
            return
        if old in text:
            path.write_text(text.replace(old, new), encoding="utf-8")

    actions_path = ramp_root / "ramphy" / "actions.py"
    replace_if_present(
        actions_path,
        (
            "        f_name = actions_dir / str(ramp_action_object.start_time)\n"
            "        ramp_action_object.save(f_name)\n"
        ),
        (
            "        safe_start_time = ramp_action_object.start_time.strftime(\"%Y%m%dT%H%M%S.%f\")\n"
            "        f_name = actions_dir / safe_start_time\n"
            "        ramp_action_object.save(f_name)\n"
        ),
        "safe_start_time = ramp_action_object.start_time.strftime",
    )
    replace_if_present(
        actions_path,
        (
            "        f_name = actions_dir / f\"{ramp_action_object.start_time}\"\n"
            "        ramp_action_object.save(f_name)\n"
        ),
        (
            "        safe_start_time = ramp_action_object.start_time.strftime(\"%Y%m%dT%H%M%S.%f\")\n"
            "        f_name = actions_dir / safe_start_time\n"
            "        ramp_action_object.save(f_name)\n"
        ),
        "safe_start_time = ramp_action_object.start_time.strftime",
    )

    hyperopt_path = ramp_root / "ramphy" / "hyperopt.py"
    replace_if_present(
        hyperopt_path,
        (
            "    tune_name = (\n"
            "        f\"{hyperparameter_experiment.engine.name}__\"\n"
            "        + f'{hyperparameter_experiment.submission_path.split(\"/\")[-1]}__'\n"
            "        + f\"{hyperparameter_experiment.data_label}\"\n"
            "    )\n"
        ),
        (
            "    tune_name = (\n"
            "        f\"{hyperparameter_experiment.engine.name}__\"\n"
            "        + f'{hyperparameter_experiment.submission_path.split(\"/\")[-1]}__'\n"
            "        + f\"{hyperparameter_experiment.data_label}\"\n"
            "    )\n"
            "    tune_name = re.sub(r'[^a-zA-Z0-9_.-]+', '_', tune_name)\n"
            "    tune_hash = hashlib.sha1(tune_name.encode()).hexdigest()[:10]\n"
            "    tune_name = f\"{hyperparameter_experiment.engine.name}_{tune_hash}\"\n"
        ),
        "hashlib.sha1(tune_name.encode())",
    )
    replace_if_present(
        hyperopt_path,
        "        resources_per_trial={\"cpu\": n_cpu_per_run, \"gpu\": n_gpu_per_run},\n",
        (
            "        resources_per_trial={\"cpu\": n_cpu_per_run, \"gpu\": n_gpu_per_run},\n"
            "        trial_dirname_creator=lambda trial: f\"trial_{trial.trial_id}\",\n"
        ),
        "trial_dirname_creator",
    )


def bootstrap(args: argparse.Namespace) -> None:
    py = args.python or sys.executable
    venv_dir = Path(args.venv)
    subprocess.run([py, "-m", "venv", str(venv_dir)], cwd=ROOT, check=True)
    venv_python = venv_python_path(venv_dir)
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run([str(venv_python), "-m", "pip", "install", "-e", "."], cwd=ROOT, check=True)
    subprocess.run(
        [
            str(venv_python), "-m", "pip", "install",
            "-r", str(ROOT / "requirements-agentk-extra.txt"),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-e", str(ROOT / "third_party" / "ds-agent"), "--no-deps"],
        cwd=ROOT,
        check=True,
    )
    if not (ROOT / "third_party" / "ramp-workflow").exists():
        shutil.unpack_archive(ROOT / "third_party" / "ramp-workflow.zip", ROOT / "third_party")
    if not (ROOT / "third_party" / "ramp-hyperopt").exists():
        shutil.unpack_archive(ROOT / "third_party" / "ramp-hyperopt.zip", ROOT / "third_party")
    patch_ramp_hyperopt_for_windows()
    subprocess.run(
        [
            str(venv_python), "-m", "pip", "install",
            "-e", str(ROOT / "third_party" / "ramp-workflow"),
            "-e", str(ROOT / "third_party" / "ramp-hyperopt"),
        ],
        cwd=ROOT,
        check=True,
    )
    (ROOT / "third_party" / "agent_k_python_path.txt").write_text(str(venv_python.resolve()), encoding="utf-8")
    print(f"Bootstrap complete. Python: {venv_python}")


def bootstrap_react(args: argparse.Namespace) -> None:
    py = args.python or sys.executable
    venv_dir = Path(args.venv)
    subprocess.run([py, "-m", "venv", str(venv_dir)], cwd=ROOT, check=True)
    venv_python = venv_python_path(venv_dir)
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-r", str(ROOT / "requirements-react.txt")],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-e", str(ROOT / "third_party" / "aideml"), "--no-deps"],
        cwd=ROOT,
        check=True,
    )
    (ROOT / "third_party" / "react_python_path.txt").write_text(str(venv_python.resolve()), encoding="utf-8")
    print(f"ReAct bootstrap complete. Python: {venv_python}")


def doctor(args: argparse.Namespace) -> None:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Python version 3.11", sys.version_info[:2] == (3, 11), sys.version.split()[0]))
    checks.append(("OpenRouter key set", bool(os.environ.get("OPENROUTER_API_KEY")), "env OPENROUTER_API_KEY"))
    checks.append(
        (
            "Agent_K python path file",
            (ROOT / "third_party" / "agent_k_python_path.txt").exists(),
            "third_party/agent_k_python_path.txt",
        )
    )
    checks.append(
        (
            "RAMP workflow extracted",
            (ROOT / "third_party" / "ramp-workflow").exists(),
            "third_party/ramp-workflow",
        )
    )
    checks.append(
        (
            "RAMP hyperopt extracted",
            (ROOT / "third_party" / "ramp-hyperopt").exists(),
            "third_party/ramp-hyperopt",
        )
    )
    checks.append(
        (
            "AIDE RAG DB exists",
            (Path(args.rag_path) / "kaggle_db" / "index.faiss").exists(),
            str(args.rag_path),
        )
    )

    import_results = []
    for module in [
        "agent",
        "ds_agent",
        "hydra",
        "pandas",
        "sklearn",
        "rampwf",
        "ramphy",
        "langchain",
        "py7zr",
        "rarfile",
        "lightgbm",
    ]:
        try:
            __import__(module)
            import_results.append((module, True))
        except Exception as exc:
            import_results.append((module, False))
            checks.append((f"import {module}", False, str(exc)))
    if all(ok for _, ok in import_results):
        checks.append(("core imports", True, ", ".join(name for name, _ in import_results)))

    for name, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")

    if not all(ok for _, ok, _ in checks):
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DACON wrapper for Agent_K/Auto_Dacon.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_run_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--competition-url", required=True)
        p.add_argument("--train", required=True)
        p.add_argument("--test", required=True)
        p.add_argument("--sample-submission", required=True)
        p.add_argument("--task-id", default=None)
        p.add_argument("--id-column", default=None)
        p.add_argument("--target-column", default=None)
        p.add_argument("--metric", default="MAE")
        p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
        p.add_argument("--project-dir", default=None)
        p.add_argument("--competition-context", default=None)

    prepare = sub.add_parser("prepare", help="Create an Agent_K local-task folder from DACON files.")
    add_common_run_args(prepare)

    run = sub.add_parser("run", help="Prepare data and run the Agent_K pipeline.")
    add_common_run_args(run)
    run.add_argument("--openrouter-api-key", default=None)
    run.add_argument("--llm", default=DEFAULT_LLM)
    run.add_argument("--code-llm", default=DEFAULT_CODE_LLM)
    run.add_argument("--total-time", type=int, default=7200)
    run.add_argument("--max-time-per-submission", type=int, default=1800)
    run.add_argument("--workspace-name", default=str(DEFAULT_WORKSPACE))
    run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    run.add_argument("--max-cpu", type=int, default=0)
    run.add_argument("--max-setups", type=int, default=3)
    run.add_argument("--blend-after-n", type=int, default=3)
    run.add_argument("--enable-agent-rag", action="store_true")
    run.add_argument("--agent-rag-path", default=None)
    run.add_argument("--agent-rag-k", type=int, default=2)
    run.add_argument(
        "--enable-hyperopt",
        action="store_true",
        help="Enable HEBO/Ray hyperparameter search. Base-model training and blending still run by default.",
    )
    project_run = sub.add_parser(
        "run-project",
        help="Run using a DACON project repo with auto_dacon_task.json and data/*.csv.",
    )
    project_run.add_argument("--project-dir", required=True)
    project_run.add_argument("--task-id", default=None)
    project_run.add_argument("--id-column", default=None)
    project_run.add_argument("--target-column", default=None)
    project_run.add_argument("--metric", default=None)
    project_run.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    project_run.add_argument("--competition-context", default=None)
    project_run.add_argument("--openrouter-api-key", default=None)
    project_run.add_argument("--llm", default=DEFAULT_LLM)
    project_run.add_argument("--code-llm", default=DEFAULT_CODE_LLM)
    project_run.add_argument("--total-time", type=int, default=7200)
    project_run.add_argument("--max-time-per-submission", type=int, default=1800)
    project_run.add_argument("--workspace-name", default=str(DEFAULT_WORKSPACE))
    project_run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    project_run.add_argument("--max-cpu", type=int, default=0)
    project_run.add_argument("--max-setups", type=int, default=3)
    project_run.add_argument("--blend-after-n", type=int, default=3)
    project_run.add_argument("--enable-agent-rag", action="store_true")
    project_run.add_argument("--agent-rag-path", default=str(DEFAULT_RAG_PATH))
    project_run.add_argument("--agent-rag-k", type=int, default=2)
    project_run.add_argument(
        "--enable-hyperopt",
        action="store_true",
        help="Enable HEBO/Ray hyperparameter search. Base-model training and blending still run by default.",
    )
    react_project = sub.add_parser(
        "run-react-project",
        help="Run AIDE post-scaffold/ReAct from a DACON project repo without automatic submission.",
    )
    react_project.add_argument("--project-dir", required=True)
    react_project.add_argument("--task-id", default=None)
    react_project.add_argument("--openrouter-api-key", default=None)
    react_project.add_argument("--react-venv", default=".venv-react")
    react_project.add_argument("--model", default=DEFAULT_REACT_MODEL)
    react_project.add_argument("--feedback-model", default=None)
    react_project.add_argument("--total-time", type=int, default=7200)
    react_project.add_argument("--exec-timeout", type=int, default=7200)
    react_project.add_argument("--steps", type=int, default=5000)
    react_project.add_argument("--top-n", type=int, default=1)
    react_project.add_argument("--warm-start", default=None)
    react_project.add_argument(
        "--skip-research-loop",
        action="store_true",
        help="Bypass the multi-model research loop and use the default/project warm-start directly.",
    )
    react_project.add_argument("--enable-rag", action="store_true")
    react_project.add_argument("--rag-path", default=str(DEFAULT_RAG_PATH))

    research_next = sub.add_parser(
        "research-next",
        help="Run multi-model research nodes and build a ReAct warm-start for the next experiment.",
    )
    research_next.add_argument("--project-dir", required=True)
    research_next.add_argument("--task-id", default=None)
    research_next.add_argument("--openrouter-api-key", default=None)
    research_next.add_argument("--run-react", action="store_true")
    research_next.add_argument("--react-venv", default=".venv-react")
    research_next.add_argument("--react-model", default=DEFAULT_REACT_MODEL)
    research_next.add_argument("--feedback-model", default=None)
    research_next.add_argument("--total-time", type=int, default=7200)
    research_next.add_argument("--exec-timeout", type=int, default=7200)
    research_next.add_argument("--steps", type=int, default=5000)
    research_next.add_argument("--top-n", type=int, default=1)
    research_next.add_argument("--enable-rag", action="store_true")
    research_next.add_argument("--rag-path", default=str(DEFAULT_RAG_PATH))

    collect = sub.add_parser("collect", help="Copy the latest generated submission to outputs/dacon/<task_id>.")
    collect.add_argument("--task-id", required=True)
    collect.add_argument("--workspace-name", default=str(DEFAULT_WORKSPACE))
    collect.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    collect.add_argument("--project-dir", default=None)

    score = sub.add_parser("record-score", help="Record a DACON public/private score for later experience reuse.")
    score.add_argument("--task-id", required=True)
    score.add_argument("--project-dir", default=None)
    score.add_argument("--public-score", required=True)
    score.add_argument("--private-score", default=None)
    score.add_argument("--metric", default="MAE")
    score.add_argument("--notes", default=None)
    score.add_argument("--experience-root", default=None)

    rag = sub.add_parser("build-aide-rag", help="Build AIDE's Kaggle-cases FAISS RAG DB from bundled cases.")
    rag.add_argument("--rag-path", default="third_party/aideml/kaggle_cases_db")

    boot = sub.add_parser("bootstrap", help="Create a local venv and install Auto_Dacon dependencies.")
    boot.add_argument("--venv", default=".venv-agentk")
    boot.add_argument("--python", default=None)

    boot_react = sub.add_parser("bootstrap-react", help="Create a separate venv for AIDE post-scaffold/ReAct.")
    boot_react.add_argument("--venv", default=".venv-react")
    boot_react.add_argument("--python", default=None)

    doc = sub.add_parser("doctor", help="Check whether this local machine is ready to run Auto_Dacon.")
    doc.add_argument("--rag-path", default=str(DEFAULT_RAG_PATH))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_task(args)
    elif args.command == "run":
        run_agent(args)
    elif args.command == "run-project":
        run_project(args)
    elif args.command == "run-react-project":
        run_react_project(args)
    elif args.command == "research-next":
        run_research_next(args)
    elif args.command == "collect":
        project_dir = Path(args.project_dir) if args.project_dir else None
        workspace_name = Path(args.workspace_name)
        output_root = Path(args.output_root)
        if project_dir is not None:
            project_dir = ensure_file(project_dir, "project repo")
            runtime_dir = project_runtime_dir(project_dir)
            if _is_default_path(args.workspace_name, DEFAULT_WORKSPACE):
                workspace_name = runtime_dir / "workspace"
            if _is_default_path(args.output_root, DEFAULT_OUTPUT_ROOT):
                output_root = runtime_dir / "outputs"
        collect_submission(
            args.task_id,
            workspace_name,
            output_root,
            project_dir,
        )
    elif args.command == "record-score":
        record_score(args)
    elif args.command == "build-aide-rag":
        build_aide_rag(args)
    elif args.command == "bootstrap":
        bootstrap(args)
    elif args.command == "bootstrap-react":
        bootstrap_react(args)
    elif args.command == "doctor":
        doctor(args)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
