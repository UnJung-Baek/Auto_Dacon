```shell
root_log_path=...  # path to where an output.jsonl is
streamlit run ./dashboard/logs_analysis/simple_app.py --server.fileWatcherType none --server.headless True -- --root_log_path $root_log_path
```