import streamlit as st
from PIL import Image
import os
from utils import preprocess_image, hash_image


def upload():
    images = st.file_uploader(
        "Upload Images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    if images:
        for img in images:
            image = Image.open(img).convert("RGB")
            image = preprocess_image(image)
            image_hash = hash_image(image)
            file_ext = img.name.split(".")[-1]
            image.save(f"data/streamlit/{image_hash}.{file_ext}")
            st.success(f"Uploaded and processed {img.name} successfully!")


def database():
    images = os.listdir("data/streamlit")
    images = [
        img
        for img in images
        if img.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    images = sorted(
        images,
        key=lambda x: Image.open(os.path.join("data/streamlit", x)).size[0] * 512
        + Image.open(os.path.join("data/streamlit", x)).size[1],
        reverse=True,
    )
    # print(f"Images in database: {images}")
    cols = st.columns(4)
    for i, img_name in enumerate(images):
        if i % 4 == 0 and i != 0:
            cols = st.columns(4)
        with cols[i % 4]:
            st.write("---")
            image_path = os.path.join("data/streamlit", img_name)
            ext = img_name.split(".")[-1]
            image = Image.open(image_path)
            w, h = image.size
            btn_cols = st.columns(2)
            with btn_cols[0]:
                with open(image_path, "rb") as img_file:
                    st.download_button(
                        label="Download",
                        data=img_file,
                        file_name=img_name,
                        mime=f"image/{ext.lower()}",
                        key=f"download_{img_name}",
                        help="Click to download this image.",
                    )
            with btn_cols[1]:
                if st.button("Use image", key=f"pipeline_{img_name}"):
                    st.session_state.image = image
                    st.session_state.result = None
                    st.session_state.remapped_result = None
                    st.session_state.upload_image = None
                    st.session_state.mask = None
                    st.session_state.points = None
                    st.session_state.masked_image = None
                    st.session_state.masked_image_hash = None
                    st.switch_page("subpages/pipeline.py")
            st.image(image, caption=f"{img_name} ({w}x{h})", use_container_width=True)


def main():
    st.title("Image Database")
    st.write("This page is under construction. Please check back later.")
    upload()
    database()


main()
