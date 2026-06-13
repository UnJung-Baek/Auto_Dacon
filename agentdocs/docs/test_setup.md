# Test Kaggle Interaction

This guide explains how to test Kaggle interaction from the Agent setup you have created.

> **Note:** Make sure you have a successful Agent setup by following the installation [guide](index.md)!

---

## ▶️ Run the test

To verify the setup, run the command below from the project root folder:

```bash
python tests/utils/test_kaggle_fetch.py
```

By running this test, you can check whether the agent can access the specified Kaggle competition.

### ▶️ Test a Specific Competition

You can test against a specific Kaggle competition by providing the --task_id argument:

```bash
python tests/utils/test_kaggle_fetch.py --task_id playground-series-s3e5
```


### 📁 Keep Downloaded Data

By default, downloaded files are stored in a temporary directory and removed after the test. To keep the data, add the --keep flag:

```bash
python tests/utils/test_kaggle_fetch.py --task_id playground-series-s3e5 --keep
```

### 📂 Specify Download Directory
You can specify a custom directory for downloaded data using the --dir_name option. Note: This option must be used with --keep to retain the data.

```bash
python tests/utils/test_kaggle_fetch.py --task_id playground-series-s3e5 --dir_name custom_dir --keep
```

## 🧪 Test LLM Query

To verify a Language Model (LLM) config (that would be added in ./config/llm), run the following command from the project root:

```bash
LLM_CONFIG=...  # (e.g. hf/example_openchat-3.5)
python tests/utils/test_llm.py llm=LLM_CONFIG

```

This test sends a sample query to the selected LLM backend and checks if a valid response is received.

# Dry Run and WorkFlow Test
You can test the workflow using the command below. It will run three tasks - one from each modality: Tabular, CV, and NLP.


## ✅ Complete Pipeline Test

Run the below command to test setup and main pipeline test.

```bash
python tests/utils/test_tasks.py
```
By default, it uses the first available GPU on your machine.
If you want to specify a GPU, you can do so by setting the CUDA_VISIBLE_DEVICES environment variable.

```bash
CUDA_VISIBLE_DEVICES=0 python tests/utils/test_tasks.py
```

## 🛠️ Setup Pipeline Test
To test a setup-only pipeline, use the `--setup_pipeline` flag

```bash
python tests/utils/test_tasks.py --setup_pipeline
```

As with the complete pipeline test, the script will use the first available GPU unless specified

```bash
CUDA_VISIBLE_DEVICES=0 python tests/utils/test_tasks.py --setup_pipeline
```