# 🚀 Running the Full Pipeline

Running the full pipeline involves the following steps:

1. 🧠 **Run Agent K (Scaffold + Post-Scaffold)** – Agent K consists of two stages:  
   - **Scaffold Stage**: analyzes the dataset and generates a problem-specific scaffold.  
   - **Post-Scaffold Stage**: runs the agent based on the ReAct framework using the scaffold to refine and improve solutions.  

2. 🔁 **Iterative Task Solving** – The ReAct-based Post-Scaffold continues solving the task iteratively on top of the scaffolded structure.  

3. 📤 **Make Final Submission** – Generate and submit the final output based on the agent’s results.  

## Command to use
```bash
MAX_TIME_PER_SUBMISSION_SECONDS=...
ALT_RAW_DATA_ROOT=...
TASK_ID=...
MODEL_ID=...
TIME_LIMIT_SECONDS=...
ATTEMPT_SPEC=...
ATTEMPT=...
BLEND_AFTER_N=...
POST_SCAFFOLD_TOP_N=...
POST_SCAFFOLD_TIMEOUT=...
POST_SCAFFOLD_LLM=...
LEADERBOARD_DIR=...
CUDA_VISIBLE_DEVICES=... python script/run_complete_pipeline_with_react.py \
--task_id=$TASK_ID \
--llm=$MODEL_ID \
--code_llm=$MODEL_ID \
--total_time=$TIME_LIMIT_SECONDS \
--attempt=$ATTEMPT \
--attempt_spec=$ATTEMPT_SPEC \
--max_time_per_submission=$MAX_TIME_PER_SUBMISSION_SECONDS \
--alt_raw_data_root=$ALT_RAW_DATA_ROOT \
--blend_after_n=$BLEND_AFTER_N \
--post_scaffold_top_n=$POST_SCAFFOLD_TOP_N \
--post_scaffold_timeout=$POST_SCAFFOLD_TIMEOUT \
--post_scaffold_llm=$POST_SCAFFOLD_LLM \
--leaderboards_dir=$LEADERBOARD_DIR
```

| VARIABLE                          | DESCRIPTION                                                    |
| --------------------------------- |----------------------------------------------------------------|
| `TASK_ID`                         | Competition slug or name of the task                           |
| `MODEL_ID`                        | ID of the LLM to be used                                       |
| `TIME_LIMIT_SECONDS`              | Total time budget for the full run (in seconds)                |
| `ATTEMPT`                         | Attempt number                                                 |
| `ATTEMPT_SPEC`                    | Specific configuration for the attempt                         |
| `MAX_TIME_PER_SUBMISSION_SECONDS` | Maximum time budget allowed per submission (in seconds)        |
| `ALT_RAW_DATA_ROOT`               | Path to the directory where raw data for the task is stored    |
| `BLEND_AFTER_N`                   | Number of submissions after which blending starts              |
| `POST_SCAFFOLD_TOP_N`             | Number of best solutions to keep after the scaffold stage      |
| `POST_SCAFFOLD_TIMEOUT`           | Execution timeout for post-scaffold phase.                     |
| `POST_SCAFFOLD_LLM`               | LLM to be used in the post-scaffold phase                      |
| `LEADERBOARD_DIR`                 | Directory path where leaderboard results are stored            |
| `CUDA_VISIBLE_DEVICES`            | GPU device(s) to be used for execution                         |
