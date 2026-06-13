# 🧠 Running ReAct Agent with Chain of Thought of Agent K

The `ReAct Agent` is based on  `AIDE` agent from the [AIDE: AI-Driven Exploration in the Space of Code](https://arxiv.org/abs/2502.13138). In our experiments
[MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://arxiv.org/abs/2410.07095) version of `AIDE` is used.
## ⚙️ Setting Up ReAct Agent

* Create a conda environment with python version 3.11.
```bash
conda create -n reactagent python=3.11 -y
conda activate reactagent
```

* Install ReAct Agent
```bash
unzip third_party/aideml.zip -d third_party/
pip install -e ./third_party/aideml
```


## 🧾 Location of the Agent K Chain of Thought
The Chain of Thoughts from Agent K runs are automatically saved during the main pipeline execution. These COTs can be used to guide ReAct to create solutions for the task in hand.
The file is called summary.txt and its location is:

```bash
  /path/to/your/workspace/
  └── <task_id>/                   
      └── seed_*/                   
          └── summary.txt  
```



## 🚀 Running ReAct Agent
To run ReAct Agent use the following command.

```bash
TASK_ID=... 
TOP_N=... 
TIME_LIMIT_SECONDS=...
TIME_OUT_SECONDS=... 
WORKSPACE_DIR=... 
MODEL_ID=... 
USE_AGENT_K_WARM_START=... 
AGENT_K_SUBMISSIONS=... 
DATA_DIR=... 
aide \
      data_dir=${DATA_DIR}/${TASK_ID} \
      exp_name="${TASK_ID}" \
      top_n="${TOP_N}" \
      agent.time_limit="${TIME_LIMIT_SECONDS}" \
      exec.timeout="${TIME_OUT_SECONDS}" \
      copy_data=false \
      workspace_dir="${WORKSPACE_DIR}" \
      agent.code.model="${MODEL_ID}" \
      agent.feedback.model="${MODEL_ID}" \
      agent.use_agent_k_warm_start="${USE_AGENT_K_WARM_START}" \
      agent.agent_k_submissions="${AGENT_K_SUBMISSIONS}"
```

## 🚀 Running ReAct Agent with RAG

Create a RAG database from the already collected Kaggle cases
```bash
RAG_DB_PATH=./third_party/aideml/kaggle_cases_db/  # Can be set to some other path
unzip third_party/aideml/kaggle_cases.zip -d third_party/aideml
python third_party/aideml/aide/utils/db_faiss.py  --populate --data_path third_party/aideml/kaggle_cases --rag_path "${RAG_DB_PATH}"
```

To run ReAct Agent use the following command.

```bash
TASK_ID=... 
TOP_N=... 
TIME_LIMIT_SECONDS=...
TIME_OUT_SECONDS=... 
WORKSPACE_DIR=... 
MODEL_ID=... 
USE_AGENT_K_WARM_START=... 
AGENT_K_SUBMISSIONS=... 
DATA_DIR=... 
RAG_DB_PATH=./third_party/aideml/kaggle_cases_db/  # should match the path used to create the database
aide \
      data_dir=${DATA_DIR}/${TASK_ID} \
      exp_name="${TASK_ID}" \
      top_n="${TOP_N}" \
      agent.time_limit="${TIME_LIMIT_SECONDS}" \
      exec.timeout="${TIME_OUT_SECONDS}" \
      copy_data=false \
      workspace_dir="${WORKSPACE_DIR}" \
      agent.code.model="${MODEL_ID}" \
      agent.feedback.model="${MODEL_ID}" \
      agent.use_agent_k_warm_start="${USE_AGENT_K_WARM_START}" \
      agent.agent_k_submissions="${AGENT_K_SUBMISSIONS}" \
      agent.use_rag=True \
      agent.rag_path="${RAG_DB_PATH}"
```

### Environment Variables
| Variable                  | Description                                                                                          |
|---------------------------|------------------------------------------------------------------------------------------------------|
| `TASK_ID`                 | Name of the competition/task.                                                                        |
| `TOP_N`                   | Number of best solutions to keep throughout the AIDE run (final output will have `TOP_N` solutions). |
| `TIME_LIMIT_SECONDS`      | Maximum allowed running time.                                                                        |
| `TIME_OUT_SECONDS`        | Execution timeout for individual runs.                                                               |
| `WORKSPACE_DIR`           | Path to the working directory.                                                                       |
| `MODEL_ID`                | Model ID used for both code generation and feedback.                                                 |
| `USE_AGENT_K_WARM_START`  | Whether to use CoT (Chain-of-Thought) warm start from Agent K.                                       |
| `AGENT_K_SUBMISSIONS`     | Path to the summary.txt of Agent K                                                                   |
| `DATA_DIR`                | Directory where the data for the competition/task is stored.                                         |
| `RAG_DB_PATH`             | Directory where the RAG database of Kaggle examples are stored.                                      |
