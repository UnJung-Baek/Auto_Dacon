# 🔁 Reproducing Results
Here are some guidelines to reproduce the results from the paper. Below the commands, there is a table of values that were used to produce the results in the paper.

**⚠️ Note: If values for some arguments are not provided, it’s up to the user to define them — they do not impact the final results.**

* Each of the experiments were run for two attempts.
* For Agent K non tabular classification competition were run with class imabalance and once without class imbalance.

## 🤖 Agent K Runs
For Agent K total 81 competitions were run. The list of competitions can be seen on the [Benchmark set](benchmark.md).
Here are the details about the configuration of the Agent K runs.
### 📊 Tabular Tasks
Around 55% of the total tasks are tabular tasks. In the [Benchmark table](benchmark.md) these tasks are the ones with only "Tab" in the modality column.
Each task was run with a maximum time limit of 24 hours on 8 cpus. Here is how to run the tabular tasks.

```bash
ALT_RAW_DATA_ROOT=... 
CPU_RANGE=...  # we used 8 cpus for our experiments
TASK_ID=... 
ATTEMPT=... 
WORKSPACE_NAME=... 
TIME_LIMIT_SECONDS=86400
MODEL_ID=qwen2.5-72b 
taskset -c $CPU_RANGE python run_complete_pipeline.py \
  --task_id $TASK_ID \
  --llm $MODEL_ID \
  --code-llm $MODEL_ID \
  --total_time $TIME_LIMIT_SECONDS \
  --tabular-task \
  --attempt $ATTEMPT \
  --max_cpu 8 \
  --workspace_name $WORKSPACE_NAME
```

🖼️📚 Running Non-Tabular (CV/NLP/Multimodal) Competitions
```bash
MAX_TIME_PER_SUBMISSION_SECONDS=... 
ALT_RAW_DATA_ROOT=... 
TASK_ID=... 
ATTEMPT_SPEC=... 
ATTEMPT=... 
TIME_LIMIT_SECONDS=172800
MODEL_ID=qwen2.5-72b
BLEND_AFTER_N=3
CUDA_VISIBLE_DEVICES=...
python run_complete_pipeline.py \
  --task_id=$TASK_ID \
  --llm=$MODEL_ID \
  --code-llm=$MODEL_ID \
  --total_time=$TIME_LIMIT_SECONDS \
  --attempt_spec=$ATTEMPT_SPEC \
  --attempt=$ATTEMPT \
  --alt_raw_data_root=$ALT_RAW_DATA_ROOT \
  --max_time_per_submission=$MAX_TIME_PER_SUBMISSION_SECONDS \
  --blend_after_n=$BLEND_AFTER_N

```

🤖 Running Agent K (Scaffold + Post-Scaffold + Submission)
```bash
MAX_TIME_PER_SUBMISSION_SECONDS=
ALT_RAW_DATA_ROOT=
TASK_ID=
MODEL_ID=qwen2.5-72b
TIME_LIMIT_SECONDS=172800(CV/NLP competitions) or 86400(Tabular competitions)
ATTEMPT_SPEC=2days
ATTEMPT=0
BLEND_AFTER_N=3
POST_SCAFFOLD_TOP_N=3
POST_SCAFFOLD_TIMEOUT=1000
POST_SCAFFOLD_LLM=qwen2.5-72b
LEADERBOARD_DIR=
CUDA_VISIBLE_DEVICES= SUDO_PASSWORD= python script/run_complete_pipeline_with_react.py \
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


**Note: Use the flag `--use_ci_handling` to handle class imbalance**