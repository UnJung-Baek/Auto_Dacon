# Auto_Dacon Windows Quickstart

This project is the reusable DACON automation engine. Each DACON competition can live in
its own repository with `auto_dacon_task.json` and `data/train.csv`, `data/test.csv`,
`data/sample_submission.csv`.

## 1. Clone

```powershell
git clone https://github.com/UnJung-Baek/Auto_Dacon.git
cd Auto_Dacon
$env:AUTO_DACON_HOME=(Get-Location).Path
```

## 2. Bootstrap

Use Python 3.11.

```powershell
py -3.11 auto_dacon.py bootstrap
```

## 3. Build the bundled Kaggle-cases RAG DB

Use an ASCII-only path on Windows because FAISS can fail on non-ASCII paths.

```powershell
.\.venv-agentk\Scripts\python.exe auto_dacon.py build-aide-rag `
  --rag-path "C:\Auto_Dacon_RAG\kaggle_cases_db"
```

## 4. Check the machine

```powershell
$env:OPENROUTER_API_KEY="..."
.\.venv-agentk\Scripts\python.exe auto_dacon.py doctor `
  --rag-path "C:\Auto_Dacon_RAG\kaggle_cases_db"
```

## 5. Run a competition project repo

Run this from the competition repository:

```powershell
cd "C:\path\to\your-dacon-project"
& "$env:AUTO_DACON_HOME\.venv-agentk\Scripts\python.exe" "$env:AUTO_DACON_HOME\auto_dacon.py" run-project `
  --project-dir . `
  --enable-agent-rag `
  --agent-rag-path "C:\Auto_Dacon_RAG\kaggle_cases_db"
```

Runtime files are written under `<competition_repo>\.auto_dacon_runtime\`, and
submission archives are copied to the competition repo outputs directory:

- Agent_K/RAMP runs: `<competition_repo>\outputs\submission_baseline.csv`,
  then `submission_baseline2.csv`, `submission_baseline3.csv`, ...
- Post-scaffold/ReAct runs: `<competition_repo>\outputs\submission_react1.csv`,
  then `submission_react2.csv`, `submission_react3.csv`, ...
