# Dashboard visualizer for Agent K
This is a simple dashboard allowing to visualize the logs of Agent K

## Installation
You need to install the package to launch the dashboard:

    pip install -r requirements.txt

## Launching the dashboard
To launch the dashboard you need to provide the path of the logs as parameters and use this command in `logs_analysis` subdirectory:

    streamlit run app.py <log_path> 

*<log_path>* is the parent logs dir in agent (.i.e logs) 