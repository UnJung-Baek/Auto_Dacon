# 🚀 Running a Custom Task Using Agent K

To execute a data science task not associated with a Kaggle competition, ensure the following setup is completed before initiating the process.

## 📁 1. Raw Data Directory Structure
Your raw data directory must follow the structure below:
```bash
  /root/path/to/your/raw/data/dir/task_id
├── raw_data_description.txt
├── raw_metric_description.txt
├── raw_task_description.txt
├── sample_submission.csv
└── ... other files
```
  An example of the raw data directory:
```bash
  /root/path/to/your/raw/data/dir/task_id
├── train/                         # Only for image-based tasks
├── test/                          # Only for image-based tasks
├── raw_data_description.txt
├── raw_metric_description.txt
├── raw_task_description.txt
├── train.csv
├── test.csv
└── sample_submission.csv
```

## 📄 2. File Descriptions
* `raw_data_description`: Provides a detailed overview of the dataset, including its context, source, and intended use.
* `raw_metric_description`: Describes the evaluation metric(s) that will be used to assess model performance on the given task.
*  `raw_task_description`: Specifies the task definition, including its objectives, constraints, and expected outputs.
* `train.csv` and `test.csv` Represent the primary tabular datasets used for training and evaluation.
  * Note: These files may be absent for image-based tasks. Additionally, test.csv may contain labels depending on the task design.
* `sample_submission.csv`: Provides the expected format for predictions submitted by the model.

Now for running the tasks depending upon the modality you follow either [Tabular](Tabular.md) or [CV/NLP](NLP.md) guides. The commands remain the same but are run with a 
`--is_local_task` flag.

**Make sure the name of the Raw Data directory for the task is same as the task_id.**




