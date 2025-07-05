import streamlit as st
import os
import glob
from PIL import Image
import zipfile


def download_button(uid, key: str):
    if st.button(
        "zip",
        key=f"download_{uid}_{key}",
        help="Download all results for this ID as a ZIP file. (Includes original, result, mask, baseline, and prompt files. The ZIP file will be created on the fly.)",
    ):
        print(f"Downloading results for ID: {uid}")
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
            if st.download_button(
                label="Download ZIP",
                data=zip_file,
                file_name=os.path.basename(zip_filename),
                mime="application/zip",
                key=f"download_zip_{os.path.basename(zip_filename)}_{key}",
                help="Click to download the results as a ZIP file.",
            ):
                os.remove(zip_filename)


def card(uid, key, full=False, expanded=False):
    with st.expander(f"Result ID: {uid}", expanded=expanded):
        with open(f"data/streamlit/logs/prompt/{uid}.txt", "r") as f:
            prompt = f.read()
        origin_path = glob.glob(f"data/streamlit/logs/origin/{uid}.*")[0]
        result_path = glob.glob(f"data/streamlit/logs/result/{uid}.*")[0]
        origin_image = Image.open(origin_path)
        result_image = Image.open(result_path)

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
        st.write(f"**Prompt:** {prompt}")
        download_button(uid, key)


def gallery_page(ids):
    col_num = st.sidebar.number_input(
        "Number of columns",
        min_value=1,
        max_value=6,
        value=3,
        step=1,
        help="Select the number of columns to display the result cards.",
    )
    expand_all = st.sidebar.checkbox(
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
            card(uid, key=f"{uid}_pagecard", full=False, expanded=expand_all)


def gallery():
    ids = os.listdir("data/streamlit/logs/result")
    ids = [uid.split(".")[0] for uid in ids]
    ids = sorted(ids)
    st.write(f"Total results: {len(ids)}")
    page_size = st.sidebar.selectbox(
        "Results per page", options=[20, 50, 100, 200, 500, 1000], index=0
    )
    page_num = len(ids) // page_size + (1 if len(ids) % page_size > 0 else 0)
    page = st.sidebar.number_input(
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
    st.write("To show more details about a specific result, enter the result ID below.")
    uid = st.text_input("Enter result ID:", "")
    if uid:
        if os.path.exists(f"data/streamlit/logs/result/{uid}.png"):
            card(uid, key=f"{uid}_query", full=True, expanded=True)
        else:
            if res:=glob.glob(f"data/streamlit/logs/result/*{uid}*"):
                if len(res) == 1:
                    selected_uid = os.path.basename(res[0]).split(".")[0]
                    st.write(f"Found result for ID: {selected_uid}")
                    card(selected_uid, key=f"{selected_uid}_query", full=True, expanded=True)
                else:
                    st.warning(f"{len(res)} results found for: \"{uid}\". Please select one to view.")
                    selected_uid = st.selectbox(
                        "Select the result to view:", options=[os.path.basename(r).split(".")[0] for r in res]
                    )
                    card(selected_uid, key=f"{selected_uid}_query", full=True, expanded=True)
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
