import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("data") / "dacon"
DEFAULT_WORKSPACE = Path("workspace_dacon")
DEFAULT_OUTPUT_ROOT = Path("outputs") / "dacon"
DEFAULT_EXPERIENCE_ROOT = Path("experiences") / "dacon"
DEFAULT_LLM = "openrouter/qwen25_72b"
DEFAULT_RAG_PATH = Path("C:/Auto_Dacon_RAG/kaggle_cases_db")


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


def project_args_from_repo(args: argparse.Namespace) -> argparse.Namespace:
    project_dir = ensure_file(args.project_dir, "project repo")
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
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
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
    context_block = competition_context.strip() if competition_context else "No additional competition context file was provided."

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
        "competition_context_file": str(args.competition_context) if getattr(args, "competition_context", None) else None,
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
    gitignore = project_dir / ".gitignore"
    if not gitignore.exists():
        write_text_file(gitignore, "data/\noutputs/*.csv\n.env\n__pycache__/\n")
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
    for key, value in read_env_file(ROOT / ".env").items():
        env.setdefault(key, value)
    if args.project_dir:
        for key, value in read_env_file(Path(args.project_dir) / ".env").items():
            env.setdefault(key, value)
    ramp_preset = env.get("AUTO_DACON_RAMP_PRESET", "agentk").lower()
    if ramp_preset not in {"agentk", "windows_fast", "fast", "local_fast"}:
        raise ValueError("AUTO_DACON_RAMP_PRESET must be one of: agentk, windows_fast, fast, local_fast")
    env["AUTO_DACON_RAMP_PRESET"] = ramp_preset
    use_fast_ramp_defaults = ramp_preset in {"windows_fast", "fast", "local_fast"}
    env.setdefault("AUTO_DACON_SKIP_RAMP_SETUP_TRAIN", "1" if use_fast_ramp_defaults else "0")
    if args.openrouter_api_key:
        env["OPENROUTER_API_KEY"] = args.openrouter_api_key
    if "OPENROUTER_API_KEY" not in env:
        raise RuntimeError("OPENROUTER_API_KEY is required. Pass --openrouter-api-key or set the environment variable.")

    cmd = [
        sys.executable,
        "run_complete_pipeline.py",
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
        if args.agent_rag_path:
            cmd.extend(["--agent_rag_path", args.agent_rag_path])

    print("Running Auto_Dacon Agent_K pipeline:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    collect_submission(task_id=task_id, workspace_name=Path(args.workspace_name), output_root=Path(args.output_root))
    print(f"Input task directory: {task_dir}")


def run_project(args: argparse.Namespace) -> None:
    run_agent(project_args_from_repo(args))


def collect_submission(task_id: str, workspace_name: Path, output_root: Path) -> Path | None:
    workspace = workspace_name if workspace_name.is_absolute() else ROOT / workspace_name
    patterns = [
        workspace / task_id / "seed_*" / "ramp_kit_v*" / "final_test_predictions" / "*.csv",
        workspace / task_id / "seed_*" / "main_pipeline" / "**" / "submission.csv",
        workspace / task_id / "post_scaffold" / "**" / "submission*.csv",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path(p) for p in workspace.glob(str(pattern.relative_to(workspace))) if Path(p).is_file())

    if not candidates:
        print("No submission candidate found yet.")
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    best = candidates[0]
    out_dir = output_root / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission.csv"
    shutil.copy2(best, out_path)
    (out_dir / "selected_submission_source.txt").write_text(str(best), encoding="utf-8")
    print(f"Collected submission: {out_path}")
    return out_path


def run_lgbm_baseline(args: argparse.Namespace) -> Path:
    forwarded = project_args_from_repo(args) if args.project_dir else args
    prepare_args = argparse.Namespace(**vars(forwarded))
    if args.project_dir:
        prepare_args.project_dir = None
    task_dir = prepare_task(prepare_args)
    task_id = forwarded.task_id or infer_task_id(forwarded.competition_url)

    import numpy as np
    import pandas as pd
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    train = pd.read_csv(task_dir / "train.csv")
    test = pd.read_csv(task_dir / "test.csv")
    sample = pd.read_csv(task_dir / "sample_submission.csv")

    id_column = forwarded.id_column or sample.columns[0]
    target_column = forwarded.target_column or sample.columns[1]
    if target_column not in train.columns:
        raise ValueError(f"Target column not found in train.csv: {target_column}")
    if id_column not in test.columns:
        raise ValueError(f"ID column not found in test.csv: {id_column}")

    features = [col for col in train.columns if col not in {id_column, target_column}]
    missing_in_test = [col for col in features if col not in test.columns]
    if missing_in_test:
        raise ValueError(f"Train feature columns missing in test.csv: {missing_in_test[:10]}")

    x = train[features].copy()
    y = train[target_column].astype(float)
    x_test = test[features].copy()

    cat_cols: list[str] = []
    for col in features:
        if x[col].dtype == "object" or x_test[col].dtype == "object":
            cat_cols.append(col)
            combined = pd.concat([x[col], x_test[col]], axis=0).astype("string").fillna("__MISSING__")
            categories = pd.Categorical(combined).categories
            x[col] = pd.Categorical(x[col].astype("string").fillna("__MISSING__"), categories=categories)
            x_test[col] = pd.Categorical(x_test[col].astype("string").fillna("__MISSING__"), categories=categories)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=args.valid_size,
        random_state=args.seed,
    )
    model = LGBMRegressor(
        objective="regression_l1",
        metric="mae",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        subsample_freq=1,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        n_jobs=args.max_cpu,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="mae",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(args.early_stopping_rounds), log_evaluation(args.log_every)],
    )
    valid_pred = model.predict(x_valid, num_iteration=model.best_iteration_)
    valid_mae = float(mean_absolute_error(y_valid, valid_pred))

    pred = model.predict(x_test, num_iteration=model.best_iteration_)
    if args.clip_min is not None:
        pred = np.maximum(pred, args.clip_min)

    pred_df = pd.DataFrame({id_column: test[id_column].values, target_column: pred})
    submission = sample[[id_column]].merge(pred_df, on=id_column, how="left")
    if submission[target_column].isna().any():
        raise RuntimeError("Missing predictions after aligning to sample_submission IDs.")
    submission = submission[list(sample.columns)]

    output_root = Path(args.output_root)
    output_dir = output_root / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = output_dir / f"submission_lgbm_baseline_{timestamp}.csv"
    latest_path = output_dir / "submission_latest.csv"
    submission.to_csv(submission_path, index=False)
    submission.to_csv(latest_path, index=False)

    report = {
        "task_id": task_id,
        "competition_url": forwarded.competition_url,
        "submission_path": str(submission_path),
        "latest_path": str(latest_path),
        "metric": forwarded.metric or "MAE",
        "valid_mae": valid_mae,
        "best_iteration": int(model.best_iteration_ or args.n_estimators),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "sample_shape": list(sample.shape),
        "id_column": id_column,
        "target_column": target_column,
        "feature_count": len(features),
        "categorical_columns": cat_cols,
        "prediction_min": float(np.min(pred)),
        "prediction_mean": float(np.mean(pred)),
        "prediction_max": float(np.max(pred)),
    }
    report_path = output_dir / f"report_lgbm_baseline_{timestamp}.json"
    write_text_file(report_path, json.dumps(report, indent=2))

    if args.project_dir:
        project_outputs = Path(args.project_dir) / "outputs"
        project_outputs.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest_path, project_outputs / "submission_latest.csv")
        write_text_file(project_outputs / "last_baseline_report.json", json.dumps(report, indent=2))

    print(f"Saved baseline submission: {latest_path}")
    print(f"Validation MAE: {valid_mae:.6f}")
    print(f"Saved report: {report_path}")
    return latest_path


def record_score(args: argparse.Namespace) -> None:
    task_id = args.task_id
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
    print(f"Built AIDE Kaggle-cases RAG DB: {rag_path}")


def patch_ramp_hyperopt_for_windows() -> None:
    """Apply small local-portability patches after extracting Agent_K archives."""
    ramp_root = ROOT / "third_party" / "ramp-hyperopt"
    actions_path = ramp_root / "ramphy" / "actions.py"
    if actions_path.exists():
        text = actions_path.read_text(encoding="utf-8")
        old = (
            "        f_name = actions_dir / str(ramp_action_object.start_time)\n"
            "        ramp_action_object.save(f_name)\n"
        )
        new = (
            "        safe_start_time = ramp_action_object.start_time.strftime(\"%Y%m%dT%H%M%S.%f\")\n"
            "        f_name = actions_dir / safe_start_time\n"
            "        ramp_action_object.save(f_name)\n"
        )
        if old in text and "safe_start_time = ramp_action_object.start_time.strftime" not in text:
            actions_path.write_text(text.replace(old, new), encoding="utf-8")

    setup_path = ramp_root / "ramphy" / "ramp_setup" / "scripts" / "setup.py"
    if setup_path.exists():
        text = setup_path.read_text(encoding="utf-8")
        if "import os" not in text.splitlines()[:5]:
            text = text.replace("import json\n", "import json\nimport os\n", 1)
        old = (
            "    rh.actions.train(\n"
            "        ramp_kit_dir = ramp_kit_dir,\n"
            "        submission = 'starting_kit',\n"
            "#        fold_idxs = range(900, 903),\n"
            "        fold_idxs = range(3),\n"
            "        force_retrain = True,\n"
            "    )\n"
        )
        new = (
            "    # Local DACON runs use this command as a kit materialization step before the\n"
            "    # actual hyperopt race. The original Agent_K path also trains the starting\n"
            "    # kit here, which can hang or consume a large slice of the time budget on\n"
            "    # Windows with wide tabular datasets.\n"
            "    if str(os.environ.get(\"AUTO_DACON_SKIP_RAMP_SETUP_TRAIN\", \"0\")).lower() not in {\"1\", \"true\", \"yes\"}:\n"
            "        rh.actions.train(\n"
            "            ramp_kit_dir = ramp_kit_dir,\n"
            "            submission = 'starting_kit',\n"
            "#            fold_idxs = range(900, 903),\n"
            "            fold_idxs = range(3),\n"
            "            force_retrain = True,\n"
            "        )\n"
        )
        if old in text and "AUTO_DACON_SKIP_RAMP_SETUP_TRAIN" not in text:
            text = text.replace(old, new)
        setup_path.write_text(text, encoding="utf-8")


def bootstrap(args: argparse.Namespace) -> None:
    py = args.python or sys.executable
    venv_dir = Path(args.venv)
    subprocess.run([py, "-m", "venv", str(venv_dir)], cwd=ROOT, check=True)
    venv_python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], cwd=ROOT, check=True)
    subprocess.run([str(venv_python), "-m", "pip", "install", "-e", "."], cwd=ROOT, check=True)
    subprocess.run(
        [
            str(venv_python), "-m", "pip", "install",
            "py7zr",
            "rarfile",
            "jsonlines",
            "textdistance",
            "gensim==4.3.3",
            "sentencepiece==0.2.0",
            "nvidia-ml-py",
            "scikit-posthocs==0.11.4",
            "lightgbm==4.6.0",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run([str(venv_python), "-m", "pip", "install", "-e", str(ROOT / "third_party" / "ds-agent")], cwd=ROOT, check=True)
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


def doctor(args: argparse.Namespace) -> None:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Python version 3.11", sys.version_info[:2] == (3, 11), sys.version.split()[0]))
    checks.append(("OpenRouter key set", bool(os.environ.get("OPENROUTER_API_KEY")), "env OPENROUTER_API_KEY"))
    checks.append(("Agent_K python path file", (ROOT / "third_party" / "agent_k_python_path.txt").exists(), "third_party/agent_k_python_path.txt"))
    checks.append(("RAMP workflow extracted", (ROOT / "third_party" / "ramp-workflow").exists(), "third_party/ramp-workflow"))
    checks.append(("RAMP hyperopt extracted", (ROOT / "third_party" / "ramp-hyperopt").exists(), "third_party/ramp-hyperopt"))
    checks.append(("AIDE RAG DB exists", (Path(args.rag_path) / "kaggle_db" / "index.faiss").exists(), str(args.rag_path)))

    import_results = []
    for module in ["agent", "ds_agent", "hydra", "pandas", "sklearn", "rampwf", "ramphy", "langchain", "py7zr", "rarfile", "lightgbm"]:
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
    run.add_argument("--code-llm", default=None)
    run.add_argument("--total-time", type=int, default=7200)
    run.add_argument("--max-time-per-submission", type=int, default=1800)
    run.add_argument("--workspace-name", default=str(DEFAULT_WORKSPACE))
    run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    run.add_argument("--max-cpu", type=int, default=0)
    run.add_argument("--max-setups", type=int, default=3)
    run.add_argument("--blend-after-n", type=int, default=3)
    run.add_argument("--enable-agent-rag", action="store_true")
    run.add_argument("--agent-rag-path", default=None)

    project_run = sub.add_parser("run-project", help="Run using a DACON project repo with auto_dacon_task.json and data/*.csv.")
    project_run.add_argument("--project-dir", required=True)
    project_run.add_argument("--task-id", default=None)
    project_run.add_argument("--id-column", default=None)
    project_run.add_argument("--target-column", default=None)
    project_run.add_argument("--metric", default=None)
    project_run.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    project_run.add_argument("--competition-context", default=None)
    project_run.add_argument("--openrouter-api-key", default=None)
    project_run.add_argument("--llm", default=DEFAULT_LLM)
    project_run.add_argument("--code-llm", default=None)
    project_run.add_argument("--total-time", type=int, default=7200)
    project_run.add_argument("--max-time-per-submission", type=int, default=1800)
    project_run.add_argument("--workspace-name", default=str(DEFAULT_WORKSPACE))
    project_run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    project_run.add_argument("--max-cpu", type=int, default=0)
    project_run.add_argument("--max-setups", type=int, default=3)
    project_run.add_argument("--blend-after-n", type=int, default=3)
    project_run.add_argument("--enable-agent-rag", action="store_true")
    project_run.add_argument("--agent-rag-path", default=str(DEFAULT_RAG_PATH))

    collect = sub.add_parser("collect", help="Copy the latest generated submission to outputs/dacon/<task_id>.")
    collect.add_argument("--task-id", required=True)
    collect.add_argument("--workspace-name", default=str(DEFAULT_WORKSPACE))
    collect.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))

    baseline = sub.add_parser("baseline", help="Prepare DACON files and create a portable LightGBM baseline submission.")
    add_common_run_args(baseline)
    baseline.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    baseline.add_argument("--valid-size", type=float, default=0.12)
    baseline.add_argument("--seed", type=int, default=42)
    baseline.add_argument("--max-cpu", type=int, default=4)
    baseline.add_argument("--n-estimators", type=int, default=3000)
    baseline.add_argument("--learning-rate", type=float, default=0.03)
    baseline.add_argument("--num-leaves", type=int, default=63)
    baseline.add_argument("--min-child-samples", type=int, default=40)
    baseline.add_argument("--subsample", type=float, default=0.85)
    baseline.add_argument("--colsample-bytree", type=float, default=0.85)
    baseline.add_argument("--reg-alpha", type=float, default=0.05)
    baseline.add_argument("--reg-lambda", type=float, default=0.2)
    baseline.add_argument("--early-stopping-rounds", type=int, default=100)
    baseline.add_argument("--log-every", type=int, default=50)
    baseline.add_argument("--clip-min", type=float, default=0.0)

    baseline_project = sub.add_parser("baseline-project", help="Create a LightGBM baseline from a DACON project repo.")
    baseline_project.add_argument("--project-dir", required=True)
    baseline_project.add_argument("--task-id", default=None)
    baseline_project.add_argument("--id-column", default=None)
    baseline_project.add_argument("--target-column", default=None)
    baseline_project.add_argument("--metric", default=None)
    baseline_project.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    baseline_project.add_argument("--competition-context", default=None)
    baseline_project.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    baseline_project.add_argument("--valid-size", type=float, default=0.12)
    baseline_project.add_argument("--seed", type=int, default=42)
    baseline_project.add_argument("--max-cpu", type=int, default=4)
    baseline_project.add_argument("--n-estimators", type=int, default=3000)
    baseline_project.add_argument("--learning-rate", type=float, default=0.03)
    baseline_project.add_argument("--num-leaves", type=int, default=63)
    baseline_project.add_argument("--min-child-samples", type=int, default=40)
    baseline_project.add_argument("--subsample", type=float, default=0.85)
    baseline_project.add_argument("--colsample-bytree", type=float, default=0.85)
    baseline_project.add_argument("--reg-alpha", type=float, default=0.05)
    baseline_project.add_argument("--reg-lambda", type=float, default=0.2)
    baseline_project.add_argument("--early-stopping-rounds", type=int, default=100)
    baseline_project.add_argument("--log-every", type=int, default=50)
    baseline_project.add_argument("--clip-min", type=float, default=0.0)

    score = sub.add_parser("record-score", help="Record a DACON public/private score for later experience reuse.")
    score.add_argument("--task-id", required=True)
    score.add_argument("--public-score", required=True)
    score.add_argument("--private-score", default=None)
    score.add_argument("--metric", default="MAE")
    score.add_argument("--notes", default=None)
    score.add_argument("--experience-root", default=str(DEFAULT_EXPERIENCE_ROOT))

    rag = sub.add_parser("build-aide-rag", help="Build AIDE's Kaggle-cases FAISS RAG DB from bundled cases.")
    rag.add_argument("--rag-path", default="third_party/aideml/kaggle_cases_db")

    boot = sub.add_parser("bootstrap", help="Create a local venv and install Auto_Dacon dependencies.")
    boot.add_argument("--venv", default=".venv-agentk")
    boot.add_argument("--python", default=None)

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
    elif args.command == "collect":
        collect_submission(args.task_id, Path(args.workspace_name), Path(args.output_root))
    elif args.command in {"baseline", "baseline-project"}:
        run_lgbm_baseline(args)
    elif args.command == "record-score":
        record_score(args)
    elif args.command == "build-aide-rag":
        build_aide_rag(args)
    elif args.command == "bootstrap":
        bootstrap(args)
    elif args.command == "doctor":
        doctor(args)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
