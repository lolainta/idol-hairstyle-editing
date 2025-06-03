import streamlit as st
import os
import glob
from PIL import Image
import zipfile
import random


@st.fragment
def download_button(uid):
    btn_cols = st.columns(2)

    if btn_cols[0].button(
        "zip",
        key=f"download_{uid}_{random.randint(0, 10000)}",
        help="Download all results for this ID as a ZIP file. (Includes original, result, mask, baseline, and prompt files. The ZIP file will be created on the fly.)",
    ):
        origin_path = glob.glob(f"data/streamlit/logs/origin/{uid}.*")[0]
        result_path = glob.glob(f"data/streamlit/logs/result/{uid}.*")[0]
        mask_path = glob.glob(f"data/streamlit/logs/mask/{uid}.*")[0]
        baseline_path = glob.glob(f"data/streamlit/logs/baseline/{uid}.*")[0]
        prompt_path = f"data/streamlit/logs/prompt/{uid}.txt"
        zip_filename = f"data/streamlit/logs/{uid}.zip"
        with zipfile.ZipFile(zip_filename, "w") as zipf:
            zipf.write(
                origin_path, "origin." + os.path.basename(origin_path).split(".")[-1]
            )
            zipf.write(
                result_path, "result." + os.path.basename(result_path).split(".")[-1]
            )
            zipf.write(mask_path, "mask." + os.path.basename(mask_path).split(".")[-1])
            zipf.write(
                baseline_path,
                "baseline." + os.path.basename(baseline_path).split(".")[-1],
            )
            zipf.write(
                prompt_path, "prompt." + os.path.basename(prompt_path).split(".")[-1]
            )
        with open(zip_filename, "rb") as zip_file:
            btn_cols[1].download_button(
                label="Download ZIP",
                data=zip_file,
                file_name=os.path.basename(zip_filename),
                mime="application/zip",
                key=f"download_zip_{os.path.basename(zip_filename)}_{random.randint(0, 10000)}",
                help="Click to download the results as a ZIP file.",
            )
        os.remove(zip_filename)


@st.fragment
def card(uid):
    with st.expander(f"Result ID: {uid}", expanded=False):
        with open(f"data/streamlit/logs/prompt/{uid}.txt", "r") as f:
            prompt = f.read()
        origin_path = glob.glob(f"data/streamlit/logs/origin/{uid}.*")[0]
        result_path = glob.glob(f"data/streamlit/logs/result/{uid}.*")[0]
        origin_image = Image.open(origin_path)
        result_image = Image.open(result_path)

        st.write(f"**Prompt:** {prompt}")
        cols = st.columns(2)
        with cols[0]:
            st.image(origin_image, caption="Original Image", use_container_width=True)
        with cols[1]:
            st.image(result_image, caption="Processed Image", use_container_width=True)
        download_button(uid)


@st.fragment
def gallery_page(ids, col_num=2):
    st.write(f"Showing {len(ids)} results:")
    cols = st.columns(col_num)
    for i, uid in enumerate(ids):
        if i % col_num == 0 and i != 0:
            cols = st.columns(col_num)
        with cols[i % col_num]:
            card(uid)


def gallery():
    ids = os.listdir("data/streamlit/logs/result")
    ids = [uid.split(".")[0] for uid in ids]
    ids = sorted(ids, reverse=True)
    st.write(f"Total results: {len(ids)}")
    cols = st.columns(2)
    with cols[0]:
        page_size = st.selectbox(
            "Results per page", options=[20, 50, 100, 200, 500, 1000], index=0
        )
    page_num = len(ids) // page_size + (1 if len(ids) % page_size > 0 else 0)
    with cols[1]:
        page = st.number_input(
            "Page number",
            min_value=1,
            max_value=page_num,
            value=1,
            step=1,
            help="Select the page number to view results.",
        )
    st.write("---")
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    ids_to_show = ids[start_index:end_index]
    gallery_page(ids_to_show, col_num=2)
    st.write(f"Showing page {page} of {page_num}")


def query():
    st.write("Query results by ID:")
    uid = st.text_input("Enter result ID:", "")
    if uid:
        if os.path.exists(f"data/streamlit/logs/result/{uid}.png"):
            card(uid)
        else:
            st.error(f"No results found for ID: {uid}")
    else:
        st.warning("Please enter a result ID to query.")


def main():
    st.title("Results")
    st.header("Quick Access")
    query()
    st.header("Gallery View")
    gallery()


main()
