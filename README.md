# Auto_Dacon

Auto_Dacon is a Windows-friendly DACON wrapper around Agent_K.

The long-term goal is to keep Agent_K's competition-solving loop: read a competition,
use the LLM for setup/EDA/code generation, search prior cases through RAG, run model
experiments, collect a submission, record the public score, and reuse that experience
later.

Repository roles:

- `Auto_Dacon`: reusable engine for many DACON competitions.
- One competition repo per DACON project: data, `auto_dacon_task.json`,
  `competition_context.md`, notes, and local outputs.

Current DACON flow:

1. Prepare local CSV files into an Agent_K-style local task.
2. Run the Agent_K pipeline with OpenRouter.
3. Let Agent_K/RAMP create the starting kit, base predictors, hyperopt/blend run,
   and final submission.
4. If the full RAMP race is unstable on Windows, optionally create a reproducible
   LightGBM emergency fallback submission with the same Auto_Dacon CLI.
5. Record the DACON public score as experience for later reuse.

## Windows Setup

Use Python 3.11.

```powershell
git clone https://github.com/UnJung-Baek/Auto_Dacon.git
cd Auto_Dacon
py -3.11 auto_dacon.py bootstrap
$env:OPENROUTER_API_KEY="..."
.\.venv-agentk\Scripts\python.exe auto_dacon.py doctor
```

## Run From A Competition Repo

The competition repo should contain:

- `auto_dacon_task.json`
- `competition_context.md` or `notes/competition_context.md`
- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`

Full Agent_K/Auto_Dacon run:

```powershell
.\.venv-agentk\Scripts\python.exe auto_dacon.py run-project `
  --project-dir "C:\path\to\Smart-Warehouse-Shipment-Delay-Prediction" `
  --total-time 7200 `
  --max-cpu 4
```

Auto_Dacon defaults to `AUTO_DACON_RAMP_PRESET=agentk`. In this mode, the setup
stage lets the LLM create the Agent_K starting kit/baseline code, RAMP setup runs
the starting kit, and the main RAMP hyperopt/blend search uses Agent_K-like
defaults.

Only set `windows_fast` when you intentionally want a smaller local smoke run:

```powershell
$env:AUTO_DACON_RAMP_PRESET="windows_fast"
```

You can also override individual RAMP settings:

```powershell
$env:AUTO_DACON_BASE_PREDICTORS="lgbm,xgboost,catboost"
$env:AUTO_DACON_N_FOLDS_HYPEROPT="3"
$env:AUTO_DACON_N_FOLDS_FINAL_BLEND="30"
$env:AUTO_DACON_DATA_PREPROCESSORS="drop_id,drop_columns,base_columnwise,col_in_train_only,rm_constant_col"
```

Agent_K setup RAG can be enabled after the local pipeline is stable:

```powershell
.\.venv-agentk\Scripts\python.exe auto_dacon.py run-project `
  --project-dir "C:\path\to\Smart-Warehouse-Shipment-Delay-Prediction" `
  --enable-agent-rag `
  --agent-rag-path "C:\Auto_Dacon_RAG\kaggle_cases_db"
```

Portable emergency fallback baseline run:

```powershell
.\.venv-agentk\Scripts\python.exe auto_dacon.py baseline-project `
  --project-dir "C:\path\to\Smart-Warehouse-Shipment-Delay-Prediction" `
  --output-root "C:\Auto_Dacon_Outputs" `
  --max-cpu 4
```

This is not the Agent_K LLM-generated baseline. It is a separate reproducible
fallback for cases where the full Agent_K/RAMP run is blocked on a local Windows
machine.

The latest fallback submission is written to:

```text
C:\Auto_Dacon_Outputs\<task_id>\submission_latest.csv
```

and copied into the competition repo:

```text
<competition_repo>\outputs\submission_latest.csv
```

## Record Public Score

```powershell
.\.venv-agentk\Scripts\python.exe auto_dacon.py record-score `
  --task-id smart_warehouse_shipment_delay_prediction `
  --public-score 11.2360866528 `
  --notes "First DACON baseline submission."
```

Secrets and data are local only. Do not commit `.env`, raw data, workspace outputs,
or generated submission CSV files.
