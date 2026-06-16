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
3. Let Agent_K/RAMP create the starting kit, base predictors, blend run, and
   final submission. Hyperparameter search runs only when explicitly enabled.
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

Requirements are split by runtime role:

- `requirements.txt`: base Auto_Dacon / Agent_K / RAMP environment.
- `requirements-agentk-extra.txt`: Windows DACON tabular extras used by bootstrap.
- `requirements-react.txt`: separated AIDE post-scaffold/ReAct environment.
- `requirements-post-scaffold.txt`: pip entry point for the separated ReAct env.

Use a separate ReAct environment before running post-scaffold:

```powershell
.\.venv-agentk\Scripts\python.exe auto_dacon.py bootstrap-react
```

## Run From A Competition Repo

The competition repo should contain:

- `auto_dacon_task.json`
- `competition_context.md` or `notes/competition_context.md`
- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`

Run from the competition repo so local runtime artifacts stay with that
competition, not in Auto_Dacon:

```powershell
$env:AUTO_DACON_HOME="C:\path\to\Auto_Dacon"
cd "C:\path\to\your-dacon-project"
```

Full Agent_K/Auto_Dacon run:

```powershell
& "$env:AUTO_DACON_HOME\.venv-agentk\Scripts\python.exe" "$env:AUTO_DACON_HOME\auto_dacon.py" run-project `
  --project-dir . `
  --total-time 7200 `
  --max-cpu 4
```

Default OpenRouter model roles:

- `agent.llm`: `openrouter/qwen37_plus` -> `qwen/qwen3.7-plus`
- `agent.code_llm`: `openrouter/claude_sonnet_46` -> `anthropic/claude-sonnet-4.6`

Auto_Dacon fixes the RAMP preset to Agent_K mode. The setup stage lets the LLM
create the Agent_K starting kit/baseline code, RAMP setup runs the starting kit,
and the main RAMP run keeps Agent_K-like model/blend behavior. HEBO/Ray
hyperparameter search is off by default and runs only with `--enable-hyperopt`.

Agent_K setup RAG can be enabled after the local pipeline is stable:

```powershell
& "$env:AUTO_DACON_HOME\.venv-agentk\Scripts\python.exe" "$env:AUTO_DACON_HOME\auto_dacon.py" run-project `
  --project-dir . `
  --enable-agent-rag `
  --agent-rag-path "C:\Auto_Dacon_RAG\kaggle_cases_db"
```

Post-scaffold/ReAct runs after the Agent_K/RAMP scaffold and does not rerun setup.
By default it first runs the multi-model research loop, builds a fresh warm-start,
then runs Claude ReAct:

```powershell
& "$env:AUTO_DACON_HOME\.venv-agentk\Scripts\python.exe" "$env:AUTO_DACON_HOME\auto_dacon.py" run-react-project `
  --project-dir . `
  --enable-rag `
  --rag-path "C:\Auto_Dacon_RAG\kaggle_cases_db"
```

The ReAct code and feedback model defaults to OpenRouter
`anthropic/claude-sonnet-4.6`. Auto_Dacon only writes submission files; it does
not submit to DACON automatically.

To bypass the research loop and run ReAct directly with the default/project
warm-start:

```powershell
& "$env:AUTO_DACON_HOME\.venv-agentk\Scripts\python.exe" "$env:AUTO_DACON_HOME\auto_dacon.py" run-react-project `
  --project-dir . `
  --enable-rag `
  --skip-research-loop
```

## Multi-Model Research Loop

After recording a public score, Auto_Dacon runs a separate research layer before
the next ReAct experiment. This does not modify Agent_K/RAMP internals. It reads the
competition repo's notes, data profile, score history, and submission list, then runs:

- Analyst nodes: DeepSeek, GLM, Kimi, GPT, Gemini
- Hypothesis nodes: DeepSeek, GLM, Kimi, GPT, Gemini
- Critic nodes: DeepSeek, GLM, Kimi, GPT, Gemini
- Selector: Claude Sonnet 4.6
- Warm-start Builder: Claude Sonnet 4.6

Create only the next warm-start without running ReAct:

```powershell
& "$env:AUTO_DACON_HOME\.venv-agentk\Scripts\python.exe" "$env:AUTO_DACON_HOME\auto_dacon.py" research-next `
  --project-dir .
```

`run-react-project` already does this automatically, but the explicit equivalent is:

```powershell
& "$env:AUTO_DACON_HOME\.venv-agentk\Scripts\python.exe" "$env:AUTO_DACON_HOME\auto_dacon.py" research-next `
  --project-dir . `
  --enable-rag `
  --run-react
```

Research outputs are written under
`<competition_repo>\notes\research_rounds\...`, and the latest warm-start is also
copied to `<competition_repo>\notes\latest_research_warm_start.txt`.

## Record Public Score

```powershell
& "$env:AUTO_DACON_HOME\.venv-agentk\Scripts\python.exe" "$env:AUTO_DACON_HOME\auto_dacon.py" record-score `
  --task-id your_dacon_task_id `
  --project-dir . `
  --public-score 11.2360866528 `
  --notes "First DACON baseline submission."
```

Secrets and data are local only. Auto_Dacon keeps reusable agent code only; competition
data, notes, scores, runtime files, and generated submissions belong in the competition
repo or in an explicit external RAG/experience path.
