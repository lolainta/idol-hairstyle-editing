import streamlit as st
from components.image_selector import image_selector
from modules.prompt_refine import prompt_refine
import os
import random
from PIL import Image


def prompt_generate():
    st.header("Prompt Generation")

    coarse_prompt = st.text_input(
        "Coarse Prompt",
        value=st.session_state.get(
            "coarse_prompt", "A cute girl with chin-length hair"
        ),
    )

    key = "prompt_image"
    image = image_selector(key)
    st.image(
        st.session_state.get(key),
        caption="Selected Image",
        use_container_width=True,
    )
    st.session_state.update({key: image})

    if st.button("Generate Prompt"):
        fine_prompt = prompt_refine(
            image=st.session_state.get(key),
            prompt=coarse_prompt,
        )
        st.session_state.update({"fine_prompt": fine_prompt})
        st.success(f"Generated Prompt: {fine_prompt}")
