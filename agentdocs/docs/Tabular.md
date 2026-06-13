# 📊 Running Tabular Competition with Agent K

This guide outlines the steps required to run a Kaggle tabular competition using Agent K.  
> **Note:** This documentation is specific to Kaggle competitions. For instructions on running custom tasks, refer to the [Running Custom Task](custom_task.md) guide.

---

## 🛠️ Prerequisites

- Ensure the agent is properly set up. Refer to the [Installation Guide](index.md) for details.

---

## 🧩 Execution Modes

Agent K supports two modes for running tabular tasks:

1. **Full Pipeline Execution**  
   Runs both the setup and main pipeline in a single step, automatically generating the final solution.

2. **Separate Execution**  
   Allows you to run the setup and main pipeline independently, offering more control over intermediate steps.

### Running the full pipeline
To run the full pipeline use the following command while in your `root` directory.

```bash
ALT_RAW_DATA_ROOT=... 
CPU_RANGE=...  
TASK_ID=... 
MODEL_ID=... 
TIME_LIMIT_SECONDS=... 
ATTEMPT=... 
WORKSPACE_NAME=... 
taskset -c $CPU_RANGE python run_complete_pipeline.py \
  --task_id $TASK_ID \
  --llm $MODEL_ID \
  --code-llm $MODEL_ID \
  --total_time $TIME_LIMIT_SECONDS \
  --tabular-task \
  --attempt $ATTEMPT \
  --max_cpu 8 \
  --workspace_name $WORKSPACE_NAME \
  --alt_raw_data_root $ALT_RAW_DATA_ROOT
```
#### ✅ Required Environment Variables and arguments for Tabular Tasks   

| Variable             | Description                                                           |
| -------------------- |-----------------------------------------------------------------------|
| `ALT_RAW_DATA_ROOT`  | Path where raw competition data is stored                             |
| `CPU_RANGE`          | Range of CPU cores to use (e.g., `0-7`)                               |
| `TASK_ID`            | Identifier for the competition task                                   |
| `MODEL_ID`           | LLM model identifier                                                  |
| `TIME_LIMIT_SECONDS` | Maximum time allowed for task execution (in sec)                      |
| `ATTEMPT`            | Attempt number                                                        |
| `WORKSPACE_NAME`     | Name of the workspace where you would like the solution to be created |

### Running setup pipeline and main pipeline separately.
* To run the setup pipeline use the command to run full pipeline but with a 
`--run_setup_only` flag. This will create the workspace for the task and it is ready to run the full pipeline.
* After setup inside `.../competition_workspace/seed_*/` there will be a text document called `main_pipeline.txt` which will contain three commands required to run the main pipeline.

# 📤 Submissions

### 🗃️ For Tabular Tasks
If your tabular tasks ran successfully, you’ll find the final submission files in:  
`/path/to/your/workspace/{competition_name}/seed_*/ramp_kit_v{LLM_Name}/final_test_predictions/`. 