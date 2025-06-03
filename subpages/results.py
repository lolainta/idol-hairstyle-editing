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
def card(uid, full=False, expanded=False):
    with st.expander(f"Result ID: {uid}", expanded=expanded):
        with open(f"data/streamlit/logs/prompt/{uid}.txt", "r") as f:
            prompt = f.read()
        origin_path = glob.glob(f"data/streamlit/logs/origin/{uid}.*")[0]
        result_path = glob.glob(f"data/streamlit/logs/result/{uid}.*")[0]
        origin_image = Image.open(origin_path)
        result_image = Image.open(result_path)

        st.write(f"**Prompt:** {prompt}")
        if not full:
            cols = st.columns(2)
            with cols[0]:
                st.image(
                    origin_image, caption="Original Image", use_container_width=True
                )
            with cols[1]:
                st.image(
                    result_image, caption="Processed Image", use_container_width=True
                )
        else:
            cols = st.columns(4)
            with cols[0]:
                st.image(
                    origin_image, caption="Original Image", use_container_width=True
                )
            with cols[1]:
                mask_path = glob.glob(f"data/streamlit/logs/mask/{uid}.*")[0]
                st.image(
                    Image.open(mask_path) if mask_path else None,
                    caption="Mask Image",
                    use_container_width=True,
                )
            with cols[2]:
                baseline_path = glob.glob(f"data/streamlit/logs/baseline/{uid}.*")
                if baseline_path:
                    st.image(
                        Image.open(baseline_path[0]),
                        caption="Baseline Image",
                        use_container_width=True,
                    )
                else:
                    st.write("No baseline image available.")
            with cols[3]:
                st.image(
                    result_image, caption="Processed Image", use_container_width=True
                )
        download_button(uid)


@st.fragment
def gallery_page(ids):
    cols = st.columns(6)
    with cols[0]:
        col_num = st.number_input(
            "Number of columns",
            min_value=1,
            max_value=6,
            value=3,
            step=1,
            help="Select the number of columns to display the result cards.",
        )
    with cols[5]:
        expand_all = st.checkbox(
            "Expand all cards",
            value=False,
            help="Check to expand all result cards.",
        )
    st.write(f"Showing {len(ids)} results:")
    card_cols = st.columns(col_num)
    for i, uid in enumerate(ids):
        if i % col_num == 0 and i != 0:
            card_cols = st.columns(col_num)
        with card_cols[i % col_num]:
            card(uid, full=False, expanded=expand_all)


def gallery():
    ids = os.listdir("data/streamlit/logs/result")
    ids = [uid.split(".")[0] for uid in ids]
    ids = sorted(ids, reverse=True)
    st.write(f"Total results: {len(ids)}")
    cols = st.columns(6)
    with cols[0]:
        page_size = st.selectbox(
            "Results per page", options=[20, 50, 100, 200, 500, 1000], index=0
        )
    page_num = len(ids) // page_size + (1 if len(ids) % page_size > 0 else 0)
    with cols[1]:
        page = st.number_input(
            f"Page number (1 to {page_num}):",
            min_value=1,
            max_value=page_num,
            value=1,
            step=1,
            help="Select the page number to view results.",
        )
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    ids_to_show = ids[start_index:end_index]
    gallery_page(ids_to_show)
    st.write(f"Showing page {page} of {page_num}")


def query():
    st.write("Query results by ID:")
    uid = st.text_input("Enter result ID:", "")
    if uid:
        if os.path.exists(f"data/streamlit/logs/result/{uid}.png"):
            card(uid, full=True)
        else:
            st.error(f"No results found for ID: {uid}")
    else:
        st.warning("Please enter a result ID to query.")


def main():
    st.title("Results")
    st.header("Quick Access")
    query()
    st.write("---")
    st.header("Gallery View")
    gallery()


main()
