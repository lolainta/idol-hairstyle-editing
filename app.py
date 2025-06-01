import streamlit as st


def main():
    st.set_page_config(
        page_title="Deep Learning Final Project",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    pg = st.navigation(
        [
            st.Page("pages/images.py", title="Image Database", icon="🖼️"),
            st.Page("pages/pipeline.py", title="Pipeline", icon="🔧"),
        ]
    )
    pg.run()


if __name__ == "__main__":
    main()
