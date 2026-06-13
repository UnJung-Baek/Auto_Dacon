# 🧠 Installing Agent K

Clone the [Agent K repository](https://github.com/huawei-noah/HEBO/tree/dev-agent/Agent_K):

```bash
git clone --depth=1 --branch dev-agent https://github.com/huawei-noah/HEBO.git
cd HEBO/Agent_K
```

### 📦 Creating the Environment

1. Create an environment named `agent`.

    ```bash
    conda create -n agent python=3.11 -y
    ```

2. Save Python executable path.

    ```bash
    conda activate agent
    which python > ./third_party/agent_k_python_path.txt
    ```

3. Install the required packages.

    ```bash
    pip install -e .[datascience]
    pip install -e ./third_party/ds-agent/
    ```
   
#### Packages required for running tabular tasks.

1. Installing ramp-hyperopt
   ```bash
   unzip third_party/ramp-hyperopt.zip -d third_party/
   pip install -e ./third_party/ramp-hyperopt/
   ```

2. Installing ramp-workflow
    ```bash
    unzip third_party/ramp-workflow.zip -d third_party/
    pip install -e ./third_party/ramp-workflow/
    ```

#### Installing geckodriver - Needed to navigate Kaggle

- Download a release of geckodriver using the following command.

```bash
wget https://github.com/mozilla/geckodriver/releases/download/v0.33.0/geckodriver-v0.33.0-linux64.tar.gz
```


- unzip the file using the following command and add path to .bashrc

```bash
tar -xvzf geckodriver-v0.32.0-linux64.tar.gz
echo 'export PATH=$PATH:'"$(pwd)" >> ~/.bashrc
source ~/.bashrc
```

- Check installation 

```bash
which geckodriver  
# Should output a path
```

Install firefox version 113.0.2

### 🏅 Setting Up Kaggle

1. Create an API Token on https://www.kaggle.com/<username>/account. This will trigger download for kaggle.json file.
2. Place this file in the appropriate directory depending on your operating system. For linux the default path is
   `~/.kaggle/kaggle.json`.
3. For more follow the steps in installation and API credentials section
   on https://github.com/Kaggle/kaggle-api/tree/main/docs.
4. Test:
   ```
   kaggle c list
   ```
   
### 🔐 Adding Login Details
   Inside `./third_party/data_preprocessing` create a json file named `kaggle_login_details.json` in the following
   format.

    ```
    {
       "username": "",
       "email": "",
       "pwd": ""
    }
    ```

### 📂 Setting Up Raw Data Paths

This is the directory where when trying a new competition Agent K downloads the raw data for the competition.
This data will be later used during the setup stage and in the main pipeline.

- In Agent K directory, create a file `root_path_to_raw_ds_data.txt` that contains the path to the
  directory where raw competition data will be saved.

```shell
RAW_DATA_PATH=...
echo $RAW_DATA_PATH > ./root_path_to_raw_ds_data.txt
```

## ⚙️ Setting Up Agent K (post-scaffold phase)

Agent K runs following a scaffold stage and a scaffold-free stage, where it uses a different environment (the same environment used to run the ReAct Agent baseline).

Create a conda environment with python version 3.11.
```bash
conda create -n reactagent python=3.11 -y
conda activate reactagent
```

Install ReAct Agent
```bash
unzip third_party/aideml.zip -d third_party/
pip install -e ./third_party/aideml
```