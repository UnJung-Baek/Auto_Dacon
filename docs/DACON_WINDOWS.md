# Auto_Dacon Windows Quickstart

This project is the reusable DACON automation engine. Each DACON competition can live in
its own repository with `auto_dacon_task.json` and `data/train.csv`, `data/test.csv`,
`data/sample_submission.csv`.

## 1. Clone

```powershell
git clone https://github.com/UnJung-Baek/Auto_Dacon.git
cd Auto_Dacon
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

```powershell
.\.venv-agentk\Scripts\python.exe auto_dacon.py run-project `
  --project-dir "C:\path\to\Smart-Warehouse-Shipment-Delay-Prediction" `
  --workspace-name "C:\Auto_Dacon_Workspace" `
  --output-root "C:\Auto_Dacon_Outputs" `
  --enable-agent-rag `
  --agent-rag-path "C:\Auto_Dacon_RAG\kaggle_cases_db"
```

The final submission is copied to `C:\Auto_Dacon_Outputs\<task_id>\submission.csv`
when a candidate submission is produced.
