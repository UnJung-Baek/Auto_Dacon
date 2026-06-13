import glob
from pathlib import Path

import fire
import jsonlines
import streamlit as st
import yaml


def sidebar(experiments: dict):
    st.sidebar.header('Agent_K')

    agent_names = experiments.keys()
    option_agent_name = st.sidebar.selectbox(
        'Agent', agent_names, key='agent_name'
    )

    option_task_name = st.sidebar.selectbox(
        'Task', experiments[option_agent_name], key='task_name'
    )

    option_run_name = st.sidebar.selectbox(
        'Run', experiments[option_agent_name][option_task_name].keys(), key='run_name'
    )

    log_path = experiments[option_agent_name][option_task_name][option_run_name]
    history_chat = get_chat_log(log_path)

    if not history_chat == {}:
        st.session_state['history_chat'] = history_chat

        option_step_id = st.sidebar.selectbox(
            'Step', history_chat.keys(), key='step_id'
        )


def body(experiments: dict):
    if 'step_id' in st.session_state:
        display_chat()


def display_chat():
    st.subheader("Chat Log")
    history = st.session_state['history_chat'][st.session_state.step_id]
    with st.expander("See log", expanded=False):
        for message in history:
            with st.chat_message(message['role']):
                st.write(message['content'])


def get_chat_log(log_path: Path) -> dict:
    history = {}
    try:
        conv_history_path = log_path / 'output.jsonl'
        step_index = 1
        with jsonlines.open(conv_history_path) as f:
            conv = list(f)
            for i, row in enumerate(conv):
                if 'templates' in row.keys():
                    template_path = Path(row['templates'][0])
                    template_name = f"{template_path.parent.parent.name}_{template_path.parent.name}_{template_path.stem}"
                    if 'llm:input' in conv[i + 1] and 'llm:output' in conv[i + 2]:
                        llm_output = {"role": "assistant", "content": conv[i + 2]['llm:output']}
                        full_conv = [query for query in conv[i + 1]['llm:input'] if query['role'] != 'assistant']
                        full_conv.append(llm_output)
                        history[f"{step_index} - {template_name}"] = full_conv
                        step_index += 1

                    elif 'templates' in row.keys() and 'error' in conv[i + 1]:
                        error = conv[i + 1]['error']
                        history[f"{step_index} - {template_name}_ERROR"] = [{'role': 'system', 'content': error}]
                        step_index += 1
                    else:
                        continue


    except FileNotFoundError:
        st.write("No conversation history found")

    return history


def read_agent_logs(
        log_path: Path = None
) -> dict:
    experiments = {}
    agent_config_path = glob.glob(f'{str(log_path)}/**/.hydra/config.yaml', recursive=True)

    for log_file_path in agent_config_path:
        log_file_path = Path(log_file_path)
        with open(log_file_path, 'r') as file:
            config = yaml.safe_load(file)
            agent_name = config['agent']['llm']['model_id']
            try:
                task_id = config['task']['task_id']
            except KeyError as e:
                task_id = config['task']['name']
            log_dir_name = log_file_path.parent.parent.name

            if agent_name not in experiments:
                experiments[agent_name] = {}
            if task_id not in experiments[agent_name]:
                experiments[agent_name][task_id] = {}

            experiments[agent_name][task_id][log_dir_name] = log_file_path.parent.parent

    return experiments


def main(
        log_path: str = None
):
    log_path = Path(log_path)

    st.set_page_config(
        page_title="Agent K",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if log_path is None:
        st.write("You need to specify the log path")
        st.stop()
        return None

    run_log_path = log_path / 'runs'

    if not run_log_path.exists():
        st.write(f"The log path you provide do not contain runs dir: {run_log_path}")
        st.stop()
        return None

    experiments = read_agent_logs(run_log_path)
    st.session_state['experiments'] = experiments
    sidebar(experiments)
    body(experiments)


if __name__ == "__main__":
    fire.Fire(main)
