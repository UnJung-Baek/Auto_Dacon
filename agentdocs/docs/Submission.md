# 🚀 Submit to Kaggle
This guide will help you submit to Kaggle competitions using both Agent K 🧠 and ReAct Agent 🤖.
Whether you’re working on a single competition or an entire workspace of projects, follow these steps to get your submissions up and running smoothly.

## 🏆 Submitting a Single Competition

Step-by-step instructions for uploading your results to a specific Kaggle competition. Perfect when you’ve just finished running an experiment and want to get those leaderboard points! 📈
from the `project root`.
```bash
WORKSPACE_ROOT_AGENT=...
WORKSPACE_ROOT_REACT=...
TASK_ID=...
MESSAGE=...
LEADERBOARDS_DIR=...
python ./third_party/data_science/kaggle_submission/submit_kaggle.py \
--workspace_root_agent=$WORKSPACE_ROOT_AGENT \
--workspace_root_aide=$WORKSPACE_ROOT_REACT \
--task_id=$TASK_ID \
--message=$MESSAGE \
--leaderboards_dir=$LEADERBOARDS_DIR
```

## 📂 Submitting an Entire Workspace (Multiple Competitions)

Step-by-step instructions to submit all competitions in your workspace in one go. 
```bash
WORKSPACE_ROOT_AGENT=...
WORKSPACE_ROOT_REACT=...
MESSAGE=...
LEADERBOARDS_DIR=...
python ./third_party/data_science/kaggle_submission/submit_kaggle.py \
--workspace_root_agent=$WORKSPACE_ROOT_AGENT \
--workspace_root_aide=$WORKSPACE_ROOT_REACT \
--message=$MESSAGE \
--leaderboards_dir=$LEADERBOARDS_DIR
```

| Variable             | Description                                            |
|----------------------|--------------------------------------------------------|
| `WORKSPACE_ROOT_AGENT` | Path to the workspace containing Agent K runs.         |
| `WORKSPACE_ROOT_REACT` | Path to the workspace containing ReAct Agent runs.     |
| `TASK_ID`            | The specific Kaggle competition name to submit.        |
| `MESSAGE`            | Message or description associated with the submission. |
| `LEADERBOARDS_DIR`   | Path to the leaderboards directory.                    |

**Notes**

1. To submit all tasks from a workspace, provide the path to either WORKSPACE_ROOT_AGENT, WORKSPACE_ROOT_AIDE, or both—depending on which agent(s) were used to run the tasks.

2. To submit a specific task only, set the TASK_ID environment variable and omit the workspace paths. This will submit just that individual task.

3. If you need to change permissions to create a new file in your folder, you can set `SUDO_PASSWORD` variable.