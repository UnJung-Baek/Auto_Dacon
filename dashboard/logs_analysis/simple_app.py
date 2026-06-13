import glob
import sys
from pathlib import Path

import fire
import streamlit as st

ROOT_PROJECT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from dashboard.logs_analysis.app import get_chat_log


def sidebar(log_paths: list[Path]):
    st.sidebar.header('Agent_K')

    option_log_path = st.sidebar.selectbox(
        'Log path', log_paths, key='agent_name')

    history_chat = get_chat_log(log_path=option_log_path.parent)

    if not history_chat == {}:
        st.session_state['history_chat'] = history_chat

        option_step_id = st.sidebar.selectbox(
            'Step', history_chat.keys(), key='step_id')


def body():
    if 'step_id' in st.session_state:
        display_chat()


def display_chat():
    st.subheader("Chat Log")
    history = st.session_state['history_chat'][st.session_state.step_id]
    with st.expander("See log", expanded=False):
        for message in history:
            with st.chat_message(message['role']):
                st.write(message['content'])


def read_agent_logs(
        log_path: Path = None
) -> list[Path]:
    if "output.jsonl" in str(log_path):
        log_path = log_path.parent
    log_paths = list(map(Path, glob.iglob(f'{str(log_path)}/**/output.jsonl', recursive=True)))
    if len(log_paths) == 0:
        raise RuntimeError(f"No log_path detected in {log_path}")
    return log_paths

def main(
        root_log_path: str = None
) -> None:
    if root_log_path is None:
        st.write("You need to specify the log path")
        st.stop()
        return None

    root_log_path = Path(root_log_path)

    st.set_page_config(
        page_title="Agent K",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    log_paths = read_agent_logs(log_path=root_log_path)
    sidebar(log_paths=log_paths)
    body()


if __name__ == "__main__":
    fire.Fire(main)
