import streamlit as st
from PIL import Image
import os
from utils import preprocess_image, hash_image


def upload():
    images = st.file_uploader(
        "Upload Images",
        type=["jpg", "jpeg", "png"],
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
    print(f"Images in database: {images}")
    cols = st.columns(3)
    for i, img_name in enumerate(images):
        if i % 3 == 0 and i != 0:
            cols = st.columns(3)
        with cols[i % 3]:
            image_path = os.path.join("data/streamlit", img_name)
            ext = img_name.split(".")[-1]
            if ext.lower() not in ["jpg", "jpeg", "png"]:
                st.warning(f"Unsupported file type: {ext}")
                continue
            image = Image.open(image_path)
            w, h = image.size
            st.image(image, caption=f"{img_name} ({w}x{h})", use_container_width=True)
            with open(image_path, "rb") as img_file:
                st.download_button(
                    label="Download",
                    data=img_file,
                    file_name=img_name,
                    mime=f"image/{ext.lower()}",
                    key=f"download_{img_name}",
                    help="Click to download this image.",
                )
        pass


def main():
    st.title("Image Database")
    st.write("This page is under construction. Please check back later.")
    upload()
    database()


main()
