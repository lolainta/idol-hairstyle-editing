import streamlit as st
from components.image_selector import image_selector


@st.fragment
def prompt_generate():
    st.header("Prompt Generation")
    st.session_state.update({"prompt": "A cute girl with chin-length hair"})

    key = "prompt_image"
    image = image_selector(key)
    if image is not None:
        st.image(image, caption="Selected Image", use_container_width=True)
        st.session_state.update({key: image})
