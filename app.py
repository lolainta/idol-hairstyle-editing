import streamlit as st
import os


def main():
    os.makedirs("data/streamlit", exist_ok=True)
    os.makedirs("data/streamlit/results", exist_ok=True)
    st.set_page_config(
        page_title="Deep Learning Final Project",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    pg = st.navigation(
        [
            st.Page("subpages/images.py", title="Image Database", icon="🖼️"),
            st.Page("subpages/pipeline.py", title="Pipeline", icon="🔧"),
        ]
    )
    pg.run()


if __name__ == "__main__":
    main()
