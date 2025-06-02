import datetime
from PIL import Image
import numpy as np
import streamlit as st
from components.mask_gen import gen_mask
from modules.stable_diff_inpaint import prompt_inpaint
from components.canva import get_segmented_image
from components.prompt_gen import prompt_generate
from components.image_selector import image_selector
from utils import hash_image, preprocess_image, log_info
import random


@st.fragment
def sam2():
    assert st.session_state.get("image", None) is not None, "Image is not ready."
    image = st.session_state.image
    face_col, hair_col = st.columns(2)
    with face_col:
        st.header("Face Segment")
        if "face_mask" not in st.session_state:
            get_segmented_image(image.copy(), "face")
    with hair_col:
        st.header("Hair Segment")
        if "hair_mask" not in st.session_state:
            get_segmented_image(image.copy(), "hair")
    if "face_mask" in st.session_state and "hair_mask" in st.session_state:
        st.success("Both face and hair masks are ready!")
        mask_generation()


@st.fragment
def mask_generation():
    assert (
        st.session_state.get("face_mask", None) is not None
    ), "Face mask is not ready."
    assert (
        st.session_state.get("hair_mask", None) is not None
    ), "Hair mask is not ready."
    st.header("Mask Generation")
    hair_mask = st.session_state.hair_mask
    face_mask = st.session_state.face_mask
    row1, row2 = gen_mask(hair_mask, face_mask)

    hcol, fcol = st.columns(2)
    with hcol:
        st.subheader("Hair Mask Generation")
        hair_cols = st.columns(3)
        hair_cols[0].image(
            row1[0],
            caption="Original Hair Mask",
            use_container_width=True,
        )
        hair_cols[1].image(
            row1[1],
            caption="Dilated Hair Mask",
            use_container_width=True,
        )
        hair_cols[2].image(
            row1[2],
            caption="Dilated Down Hair Mask",
            use_container_width=True,
        )
    with fcol:
        st.subheader("Face Mask Generation")
        face_cols = st.columns(3)
        face_cols[0].image(
            row2[0],
            caption="Original Face Mask",
            use_container_width=True,
        )
        face_cols[1].image(
            row2[1],
            caption="Filled Face Mask",
            use_container_width=True,
        )
        face_cols[2].image(
            row2[2],
            caption="Final Mask (Face Subtracted)",
            use_container_width=True,
        )
    print(
        f"Hair mask shape: {row1[0].shape}, Face mask shape: {row2[0].shape}, Final mask shape: {row2[2].shape}"
    )
    st.session_state.update({"filled_face_mask": row2[1], "inpaint_mask": row2[2]})
    if "inpaint_mask" in st.session_state:
        cols = st.columns(2)
        with cols[0]:
            prompt_generate()
        with cols[1]:
            inpaint()


def inpaint():
    st.header("Inpainting")
    assert (
        st.session_state.get("inpaint_mask", None) is not None
    ), "Inpaint mask is not ready."
    inpaint_mask = st.session_state.inpaint_mask
    image = st.session_state.image
    inpaint_mask = Image.fromarray(inpaint_mask.astype(np.uint8))

    with st.form("Inpainting Settings"):
        prompt = st.text_area(
            "Inpainting Prompt",
            st.session_state.get("fine_prompt"),
            height=200,
        )
        cols = st.columns(2)
        with cols[0]:
            scale = st.slider(
                "Inpainting Scale",
                min_value=0.1,
                max_value=30.0,
                value=7.5,
                step=0.1,
                help="Scale for the inpainting model.",
            )
        with cols[1]:
            ddim_steps = st.slider(
                "Inpainting DDIM Steps",
                min_value=0,
                max_value=80,
                value=50,
                step=1,
                help="Number of DDIM steps for the inpainting model, it will take longer to generate the image with more steps.",
            )
        # give a running feedback onclick when processing
        if st.form_submit_button("Inpaint"):
            with st.spinner("Inpainting in progress..."):
                result = prompt_inpaint(
                    image=image,
                    mask=inpaint_mask,
                    prompt=prompt,
                    scale=scale,
                    ddim_steps=ddim_steps,
                    w=st.session_state.image.width,
                    h=st.session_state.image.height,
                )
                result = preprocess_image(result)
                result_hash = hash_image(result)
                path = f"data/streamlit/results/{result_hash}.png"
                result.save(path)
                st.session_state.update(
                    {
                        "result": result,
                        "result_path": path,
                        "result_hash": result_hash,
                        "prompt": prompt,
                    }
                )
                if "result" in st.session_state:
                    remap_face()
    if (
        st.session_state.get("result", None) is not None
        and st.session_state.get("result_path", None) is not None
        and st.session_state.get("result_hash", None) is not None
    ):
        st.success("Inpainting completed!")
        with open(st.session_state.result_path, "rb") as img_file:
            st.download_button(
                label="Download Inpainted Image",
                data=img_file,
                file_name=f"{st.session_state.result_hash}.png",
                mime="image/png",
                key="download_inpainted_image",
                help="Click to download the inpainted image.",
            )


@st.fragment
def remap_face():
    st.header("Remap Face Segment")
    assert (
        st.session_state.get("filled_face_mask", None) is not None
    ), "Face mask is not ready."
    assert (
        st.session_state.get("result", None) is not None
    ), "Inpainted image is not ready."
    filled_face_mask = st.session_state.filled_face_mask
    result = st.session_state.result
    print(
        f"Face mask shape: {filled_face_mask.shape}, Result shape: {result.size}, Image shape: {st.session_state.image.size}"
    )
    filled_face_mask_3d = np.stack([filled_face_mask] * 3, axis=-1)
    remap_result = np.where(
        filled_face_mask_3d > 0,
        st.session_state.image,
        np.array(result),
    )
    remap_result = Image.fromarray(remap_result.astype(np.uint8))
    st.session_state.update({"remapped_result": remap_result})

    cols = st.columns(3)
    with cols[0]:
        st.image(
            st.session_state.image,
            caption="Original Image",
            use_container_width=True,
        )
    with cols[1]:
        st.image(
            st.session_state.result,
            caption="Inpainted Image",
            use_container_width=True,
        )
    with cols[2]:
        st.image(
            st.session_state.remapped_result,
            caption="Remapped Face Segment",
            use_container_width=True,
        )

    uid = log_info(
        input=np.array(st.session_state.image),
        mask=st.session_state.filled_face_mask,
        inpaint=np.array(st.session_state.result),
        output=np.array(st.session_state.remapped_result),
        prompt=st.session_state.prompt,
    )
    st.info(
        f"Results logged with ID: {uid}. You can find the logs in the `data/streamlit/logs` directory."
    )


def main():
    print(
        "=" * 50
        + datetime.datetime.now().strftime(" %Y-%m-%d %H:%M:%S ")
        + " - Starting Streamlit app... "
        + "=" * 50
    )
    for k, v in st.session_state.items():
        print(f"Session State - {k}: {type(v)}")

    st.title("Deep Learning Final Project")
    st.markdown(
        "This is a Streamlit app for the final project of the Deep Learning course. "
        "We will use Segment Anything Model (SAM) to draw points on an image and generate masks."
    )
    if st.button("Reset", help="Click to reset the mask and use the same image."):
        image = st.session_state.get("image", None)
        st.session_state.clear()
        st.cache_data.clear()
        st.session_state.update({"image": image})
        st.rerun()

    image = image_selector("image")
    st.session_state.update({"image": image})
    st.session_state.canvas_key = random.randint(0, 10000)
    st.session_state.seed = random.randint(0, 1000000)

    sam2()


# if __name__ == "__main__":
main()
