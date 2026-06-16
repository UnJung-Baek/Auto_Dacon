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
DEFAULT_DATA_ROOT = Path("data") / "dacon"
DEFAULT_WORKSPACE = Path("workspace_dacon")
DEFAULT_OUTPUT_ROOT = Path("outputs") / "dacon"
DEFAULT_LLM = "openrouter/qwen37_plus"
DEFAULT_CODE_LLM = "openrouter/claude_sonnet_46"
DEFAULT_RAG_PATH = Path("C:/Auto_Dacon_RAG/kaggle_cases_db")
AIDE_RAG_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
PROJECT_RUNTIME_DIRNAME = ".auto_dacon_runtime"
DEFAULT_REACT_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_RESEARCH_PANEL_MODELS = (
    "deepseek/deepseek-v4-pro",
    "z-ai/glm-5.1",
    "moonshotai/kimi-k2.7-code",
    "openai/gpt-5.5",
    "google/gemini-3.5-flash",
)
DEFAULT_RESEARCH_ANALYST_MODELS = DEFAULT_RESEARCH_PANEL_MODELS
DEFAULT_RESEARCH_HYPOTHESIS_MODELS = DEFAULT_RESEARCH_PANEL_MODELS
DEFAULT_RESEARCH_CRITIC_MODELS = DEFAULT_RESEARCH_PANEL_MODELS
DEFAULT_RESEARCH_SELECTOR_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_RESEARCH_WARM_START_MODEL = "anthropic/claude-sonnet-4.6"


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


def research_messages(role: str, context: dict[str, Any], extra: str) -> list[dict[str, str]]:
    prompt_context = compact_research_context_for_prompt(context)
    compact_context = json.dumps(prompt_context, ensure_ascii=False, indent=2, default=str)
    if len(compact_context) > 50000:
        compact_context = compact_context[:50000] + "\n...<truncated>"
    system = (
        "You are an elite DACON tabular competition research node. "
        "Respect the Auto_Dacon contract: do not change Agent_K core algorithms, "
        "do not assume automatic submission, avoid leakage, and prefer experiments "
        "that can be validated with robust folds. Be concrete and evidence-driven."
    )
    user = (
        f"Role: {role}\n\n"
        f"Project context JSON:\n{compact_context}\n\n"
        f"Task:\n{extra}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def openrouter_chat(model: str, messages: list[dict[str, str]], api_key: str, max_tokens: int = 2200) -> str:
    from openai import OpenAI

    timeout = float(os.getenv("AUTO_DACON_LLM_TIMEOUT", "120"))
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=timeout,
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.35,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"OpenRouter model returned empty content: {model}")
    return content


def run_research_panel(
        name: str,
        models: tuple[str, ...],
        context: dict[str, Any],
        instruction: str,
        api_key: str,
        out_dir: Path,
) -> list[dict[str, str]]:
    results = []
    for model in models:
        print(f"Running research node {name}: {model}")
        try:
            text = openrouter_chat(
                model,
                research_messages(name, context, instruction),
                api_key,
            )
        except Exception as exc:
            text = f"ERROR from {model}: {type(exc).__name__}: {exc}"
        safe_model = "".join(ch if ch.isalnum() else "_" for ch in model).strip("_")
        write_text_file(out_dir / f"{name}_{safe_model}.md", text)
        results.append({"role": name, "model": model, "content": text})
    if results and all(result["content"].startswith("ERROR from ") for result in results):
        raise RuntimeError(f"All models failed in research node: {name}")
    return results


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
    api_key = env.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required. Pass --openrouter-api-key or set the environment variable.")

    context = load_project_research_context(project_dir, metadata)
    round_dir = project_dir / "notes" / "research_rounds" / datetime.now().strftime("%Y%m%d_%H%M%S")
    round_dir.mkdir(parents=True, exist_ok=True)
    write_text_file(round_dir / "context_snapshot.json", json.dumps(context, ensure_ascii=False, indent=2, default=str))

    analyst_results = run_research_panel(
        "analyst",
        DEFAULT_RESEARCH_ANALYST_MODELS,
        context,
        (
            "Analyze the latest public-score trend, CV/public mismatch risk, data profile, "
            "and project notes. Identify why recent attempts improved or failed. "
            "Return concise findings and experiment priorities."
        ),
        api_key,
        round_dir,
    )
    hypothesis_context = {**context, "analyst_outputs": analyst_results}
    hypothesis_results = run_research_panel(
        "hypothesis",
        DEFAULT_RESEARCH_HYPOTHESIS_MODELS,
        hypothesis_context,
        (
            "Generate 5 to 8 concrete experiment hypotheses. Focus first on EDA-driven "
            "preprocessing and feature engineering, then model/ensemble fit. For each "
            "hypothesis include rationale, implementation sketch, validation method, "
            "expected effect, runtime cost, and leakage risk."
        ),
        api_key,
        round_dir,
    )
    critic_context = {**hypothesis_context, "hypothesis_outputs": hypothesis_results}
    critic_results = run_research_panel(
        "critic",
        DEFAULT_RESEARCH_CRITIC_MODELS,
        critic_context,
        (
            "Critique the proposed hypotheses aggressively. Reject leakage-prone, "
            "public-overfit, duplicate, too-expensive, or weakly evidenced ideas. "
            "Keep only robust experiments likely to improve private leaderboard."
        ),
        api_key,
        round_dir,
    )
    selector_context = {**critic_context, "critic_outputs": critic_results}
    selector = openrouter_chat(
        DEFAULT_RESEARCH_SELECTOR_MODEL,
        research_messages(
            "selector",
            selector_context,
            (
                "Select the best 1 to 2 hypotheses for the next ReAct run. "
                "Give a strict decision: accepted hypotheses, rejected hypotheses, "
                "validation guardrails, and exact implementation priorities. "
                "Do not request hyperparameter search unless the user explicitly enabled it."
            ),
        ),
        api_key,
        max_tokens=2600,
    )
    write_text_file(round_dir / "selector_decision.md", selector)

    warm_start = openrouter_chat(
        DEFAULT_RESEARCH_WARM_START_MODEL,
        research_messages(
            "warm_start_builder",
            {**selector_context, "selector_decision": selector},
            (
                "Write the final warm-start instruction for Auto_Dacon run-react-project. "
                "It must be directly actionable for a coding agent. Include competition "
                "goal, current best public score if known, selected hypotheses, validation "
                "protocol, leakage warnings, output requirement, and a fallback plan that "
                "still writes a valid submission.csv. Keep it focused."
            ),
        ),
        api_key,
        max_tokens=3000,
    )
    warm_start_path = round_dir / "warm_start_for_react.txt"
    write_text_file(warm_start_path, warm_start)
    write_text_file(project_dir / "notes" / "latest_research_warm_start.txt", warm_start)
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
