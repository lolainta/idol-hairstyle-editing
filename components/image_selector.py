import os
from PIL import Image
import streamlit as st
import random


def image_selector(key: str):
    if key not in st.session_state:
        st.session_state[key] = None
    if st.session_state[key] is not None:
        print(f"Image already selected: {st.session_state[key]}")
        return st.session_state[key]

    images = os.listdir("data/streamlit")
    images = [img for img in images if img.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)
    # print(f"Images in database: {images}")

    if not images:
        st.warning("No images available in the database.")
        return None

    selected_image = st.selectbox("Select an image:", images, key=f"{key}_selectbox")
    print(f"Selected image: {selected_image}")
    image_path = os.path.join("data/streamlit", selected_image)
    image = Image.open(image_path)
    st.session_state[key] = image
    return image
