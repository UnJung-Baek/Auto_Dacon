# 🔍 TabPFN and Variants

TabPFN and its variants include:

- **TabPFN**
- **TabPFN-Extension**
- **TabICL**

---

## 📦 Creating the Environment

1. Create an environment named `tabpfn`.

    ```bash
    conda create -n tabpfn python=3.11 -y
    ```

2. Activate environment.

    ```bash
    conda activate tabpfn
    ```

3. Install dependency packages.
   ```bash
   pip install -r third_party/tabpfn/requirements.txt
   
   pip install git+https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/ramp-workflow.git@generative_regression_clean
   
   pip install git+https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/ramp-hyperopt.git@fe
   ```

#### Install TabPFN

```bash
pip install "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git@98e9fc84346ce71dcb2b45bd73dffdc8dc999337"
```

#### Install TabPFN-extension

```bash
pip install "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git@c49687fd30d9f7a2aa23260a9bcc2db79305381d"
```

#### Install TabICL

```bash
pip install git+https://github.com/soda-inria/tabicl.git@568edf8e1af29f3fe2c146cb550262e166edca83
```

## 📂 Dataset

All TabPFN variants operate on **Agent K -generated setups**. For more details, refer to
the [Tabular Documentation](Tabular.md).

Once the setup is successfully created, it expects the following file structure:

```text
📁 ./bike-sharing-demand
  ├── 📁 seed_0
  │   ├── 📁 ramp_kit_*
  │   │   ├── 📄 problem.py
  │   │   └── 📁 data
  │   │       ├── 📄 metadata.json
  │   │       ├── 📄 sample_submission.csv
  │   │       ├── 📄 test.csv
  │   │       └── 📄 train.csv
```

## ⚙️ Experiments

### 🧪 Trials

We conducted **three trials** for each TabPFN variant, using **three separate seeds**  created setups by Agent K. These
setups are then used by AgentK in its main pipeline runs.

### ⏱️ Time Allocation

Each experiment is allocated a **2-day time budget**.

### 📏 Context Length

By default, all variants use a **context length of 10,000**, which is the maximum supported by TabPFN.

- **TabPFN** and **TabPFN-Extension**: 10,000 context length.
- **TabICL**: Up to 100,000 context length depending on the competition and system configuration.

### ▶️ Run TabPFN for a competition

```bash
python third_party/tabpfn/run_tabpfn.py --competition_name <competition name> --workspace_path <workspace name> --setup_path <setup path>
```

#### 💡 Example

```bash
python third_party/tabpfn/run_tabpfn.py --competition_name playground-series-s3e11 --workspace_path workspace_test --setup_path /path/to/setup/playground-series-s3e11/seed_0
```

You can specify the context window size using the --context_length option. The default value for TabPFN is 10,000.

```bash
python third_party/tabpfn/run_tabpfn.py --competition_name <competition name> --workspace_path <workspace name> --setup_path <setup path> --context_length <context window size>
```

### ▶️ Run TabPFN-extension for a competition

```bash
python third_party/tabpfn/run_tabpfn_extentions.py --competition_name <competition name> --workspace_path <workspace name> --setup_path <setup path>
```

You can specify the context window size using the --context_length option. The default value for TabPFN-extension is
10,000.

```bash
python third_party/tabpfn/run_tabpfn_extentions.py --competition_name <competition name> --workspace_path <workspace name> --setup_path <setup path> --context_length <context window size>
```

### ▶️ Run TabICL for a competition

```bash
python third_party/tabpfn/run_tabicl.py --competition_name <competition name> --workspace_path <workspace name> --setup_path <setup path>
```

You can specify the context window size using the --context_length option.
In our experiments, we used a default context length of 100,000, but reduced it for some competitions due to hardware
restrictions.
Specifically:

- cat-in-the-dat-ii : 90000
- santander-customer-satisfaction : 20000

```bash
python third_party/tabpfn/run_tabicl.py --competition_name <competition name> --workspace_path <workspace name> --setup_path <setup path> --context_length <context window size>
```