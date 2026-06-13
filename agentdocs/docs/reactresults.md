# 🦾 ReAct Agent Runs

### 🤖 ReAct Agent (No RAG) 
```bash
TASK_ID=... 
TOP_N=... 
MIN_EXEC_TIME_SECONDS=... 
TIME_LIMIT_SECONDS=172800 (Tabular), 345600 (CV/NLP)
TIME_OUT_SECONDS=32400 (Tabular), 64800 (CV/NLP)
WORKSPACE_DIR=... 
MODEL_ID=Qwen/Qwen2.5-72B-Instruct 
DATA_DIR=... 
TOKENIZERS_PARALLELISM=False aide \
      data_dir=${DATA_DIR}/${TASK_ID} \
      exp_name="${TASK_ID}" \
      top_n="${TOP_N}" \
      agent.time_limit="${TIME_LIMIT_SECONDS}" \
      agent.min_exec_time="${MIN_EXEC_TIME_SECONDS}" \
      exec.timeout="${TIME_OUT_SECONDS}" \
      copy_data=false \
      workspace_dir="${WORKSPACE_DIR}" \
      agent.code.model="${MODEL_ID}" \
      agent.feedback.model="${MODEL_ID}" \
      agent.use_rag=False
```
### 📚 ReAct Agent + RAG

To set up a RAG database, refer to the [ReactAgent](reactagent.md) section.
```bash
TASK_ID=... 
TOP_N=... 
MIN_EXEC_TIME_SECONDS=... 
TIME_LIMIT_SECONDS=172800 (Tabular), 345600 (CV/NLP)
TIME_OUT_SECONDS=32400 (Tabular), 64800 (CV/NLP)
WORKSPACE_DIR=... 
MODEL_ID=Qwen/Qwen2.5-72B-Instruct  
DATA_DIR=... 
RAG_DB_PATH=...
TOKENIZERS_PARALLELISM=False aide \
      data_dir=${DATA_DIR}/${TASK_ID} \
      exp_name="${TASK_ID}" \
      top_n="${TOP_N}" \
      agent.time_limit="${TIME_LIMIT_SECONDS}" \
      agent.min_exec_time="${MIN_EXEC_TIME_SECONDS}" \
      exec.timeout="${TIME_OUT_SECONDS}" \
      copy_data=false \
      workspace_dir="${WORKSPACE_DIR}" \
      agent.code.model="${MODEL_ID}" \
      agent.feedback.model="${MODEL_ID}" \
      agent.use_rag=True \
      agent.rag_path="${RAG_DB_PATH}"
```

### 💡 ReAct Agent from COT
```bash
TASK_ID=... 
TOP_N=... 
MIN_EXEC_TIME_SECONDS=... 
TIME_LIMIT_SECONDS=172800 (Tabular), 345600 (CV/NLP)
TIME_OUT_SECONDS=32400 (Tabular), 64800 (CV/NLP)
WORKSPACE_DIR=... 
MODEL_ID=Qwen/Qwen2.5-72B-Instruct 
DATA_DIR=... 
$USE_AGENT_K_WARM_START=...
$AGENT_K_SUBMISSIONS=...
TOKENIZERS_PARALLELISM=False aide \
      data_dir=${DATA_DIR}/${TASK_ID} \
      exp_name="${TASK_ID}" \
      top_n="${TOP_N}" \
      agent.time_limit="${TIME_LIMIT_SECONDS}" \
      agent.min_exec_time="${MIN_EXEC_TIME_SECONDS}" \
      exec.timeout="${TIME_OUT_SECONDS}" \
      copy_data=false \
      workspace_dir="${WORKSPACE_DIR}" \
      agent.code.model="${MODEL_ID}" \
      agent.feedback.model="${MODEL_ID}" \
      agent.use_agent_k_warm_start=$USE_AGENT_K_WARM_START \
      agent.agent_k_submissions=$AGENT_K_SUBMISSIONS \
      agent.use_rag=False
```

### ⚡ ReAct Agent (DeepSeek)
From your `Project Root` and inside the `reactagent` environment
```bash
TASK_ID=... 
TOP_N=... 
MIN_EXEC_TIME_SECONDS=... 
TIME_LIMIT_SECONDS=172800 (Tabular), 345600 (CV/NLP)
TIME_OUT_SECONDS=32400 (Tabular), 64800 (CV/NLP)
WORKSPACE_DIR=... 
MODEL_ID=deepseek-reasoner 
DATA_DIR=... 
DEEPSEEK_API_KEY=<your-api-key> TOKENIZERS_PARALLELISM=False aide \
      data_dir=${DATA_DIR}/${TASK_ID} \
      exp_name="${TASK_ID}" \
      top_n="${TOP_N}" \
      agent.time_limit="${TIME_LIMIT_SECONDS}" \
      agent.min_exec_time="${MIN_EXEC_TIME_SECONDS}" \
      exec.timeout="${TIME_OUT_SECONDS}" \
      copy_data=false \
      workspace_dir="${WORKSPACE_DIR}" \
      agent.code.model="${MODEL_ID}" \
      agent.feedback.model="${MODEL_ID}" \
      agent.use_rag=False
```


For descriptions of each of the variables and arguments and additional information refer to [running ReAct](reactagent.md).