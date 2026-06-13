# 🖼️📚 Running Non-Tabular (CV/NLP) Competitions with Agent K

This guide outlines the steps required to run a **Kaggle non-tabular competition** (Computer Vision or Natural Language
Processing) using Agent K.
> **Note:** This documentation is specific to Kaggle competitions. For instructions on running custom tasks, refer to
> the [Running Custom Task](custom_task.md) guide.

---

## 🛠️ Prerequisites

- Ensure the agent is properly set up. Refer to the [Installation Guide](index.md) for details.

---

## Running Agent K

To run a CV/NLP task using Agent K use the following command from the  `root` directory.
As multi-gpu training is not supported by Agent K , it is important to set the value for `CUDA_VISIBLE_DEVICES=0`
variable while running the command.

```bash
MAX_TIME_PER_SUBMISSION_SECONDS=... 
ALT_RAW_DATA_ROOT=... 
TASK_ID=... 
MODEL_ID=... 
TIME_LIMIT_SECONDS=... 
ATTEMPT_SPEC=... 
ATTEMPT=... 
BLEND_AFTER_N=...
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

#### ✅ Required Environment Variables and arguments for Non-Tabular Tasks

| Variable                          | Description                                                 |
|-----------------------------------|-------------------------------------------------------------|
| `MAX_TIME_PER_SUBMISSION_SECONDS` | Time budget per submission                                  |
| `ALT_RAW_DATA_ROOT`               | Path to the directory where raw data for the task is stored |
| `USE_CI_HANDLING`                 | Enable/disable CI (Class Imbalance) handling                |
| `TASK_ID`                         | Identifier for the competition task                         |
| `MODEL_ID`                        | LLM model identifier                                        |
| `TIME_LIMIT_SECONDS`              | Time limit for the full pipeline execution                  |
| `ATTEMPT_SPEC`                    | Specific configuration for the attempt                      |
| `ATTEMPT`                         | ATTEMPT number                                              |
| `BLEND_AFTER_N`                   | create a blend solution after n success submissions         |
| `CUDA_VISIBLE_DEVICES`            | CUDA device ID(s)                                           |

## 📤 Submissions

If the non tabular task has run succesfully the submission files can be found in
`/path/to/you/workspace/attempt- /competition_name/main_pipeline/submissions/date_and_time_of_the_run/`.
The submission files are one or more of the following

- `submission.csv`
- `submission_alt.csv`
- `cv-submission.csv`
- `cv-submission_alt.csv`
- `tta-submission.csv`
- `tta-submission_alt.csv`
