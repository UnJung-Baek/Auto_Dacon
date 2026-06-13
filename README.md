# Auto_Dacon

Auto_Dacon is a Windows-friendly DACON wrapper around Agent_K.

The long-term goal is to keep Agent_K's competition-solving loop: read a competition,
use the LLM for setup/EDA/code generation, search prior cases through RAG, run model
experiments, collect a submission, record the public score, and reuse that experience
later.

Current DACON flow:

1. Prepare local CSV files into an Agent_K-style local task.
2. Run the Agent_K pipeline with OpenRouter.
3. If the full RAMP race is unstable on Windows, create a reproducible LightGBM
   baseline submission with the same Auto_Dacon CLI.
4. Record the DACON public score as experience for later reuse.

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

Portable baseline run:

```powershell
.\.venv-agentk\Scripts\python.exe auto_dacon.py baseline-project `
  --project-dir "C:\path\to\Smart-Warehouse-Shipment-Delay-Prediction" `
  --output-root "C:\Auto_Dacon_Outputs" `
  --max-cpu 4
```

The latest baseline submission is written to:

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
