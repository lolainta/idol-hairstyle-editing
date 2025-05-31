import datetime
import hashlib
from PIL import Image
import numpy as np
import streamlit as st
from mask_gen import (
    dilate_mask,
    dilate_mask_down,
    subtract_face_from_mask,
    fill_holes_and_close,
)
from stable_diff_inpaint import prompt_inpaint
from components.sam import get_segmented_image


def hash_image(image):
    sha1 = hashlib.sha1()
    sha1.update(image.read())
    return sha1.hexdigest()[:8]


def sam2():
    global upload_image
    global MAX_SIZE
    image = Image.open(upload_image).convert("RGB")
    file_ext = upload_image.name.split(".")[-1]
    image_hash = hash_image(upload_image)
    image.save(f"data/streamlit/{image_hash}.{file_ext}")
    w, h = image.size
    if w > MAX_SIZE or h > MAX_SIZE:
        factor = MAX_SIZE / max(w, h)
        w = int(factor * w)
        h = int(factor * h)
        w = w // 64 * 64
        h = h // 64 * 64
        image = image.resize((w, h))
        st.warning(f"Image resized to {w}x{h} to fit within {MAX_SIZE}px limit.")
    st.session_state.update({"image": image})
    face_col, hair_col = st.columns(2)
    with face_col:
        st.header("Face Segment")
        if "face_mask" not in st.session_state:
            get_segmented_image(image, "face")
    with hair_col:
        st.header("Hair Segment")
        if "hair_mask" not in st.session_state:
            get_segmented_image(image, "hair")


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
    hair_cols = st.columns(len(row1))
    for i, mask in enumerate(row1, start=1):
        hair_cols[i - 1].image(mask, caption=f"Hair Mask {i}", use_container_width=True)
    face_cols = st.columns(len(row2))
    for i, mask in enumerate(row2, start=1):
        face_cols[i - 1].image(mask, caption=f"Face Mask {i}", use_container_width=True)
    st.session_state.update(
        {
            "inpaint_mask": row2[-1],
        }
    )


def gen_mask(hair_mask: np.ndarray, face_mask: np.ndarray) -> list:
    row1 = []
    row2 = []
    mask = hair_mask
    row1.append(mask)

    kd1_cols = st.columns(2)
    with kd1_cols[0]:
        ks1 = st.slider(
            "Dilate Kernel Size",
            min_value=1,
            max_value=15,
            value=5,
            step=1,
            help="Kernel size for dilating the hair mask.",
        )
    with kd1_cols[1]:
        di1 = st.slider(
            "Dilate Dilation Iterations",
            min_value=1,
            max_value=15,
            value=5,
            step=1,
            help="Number of iterations for dilation of the hair mask.",
        )

    mask = dilate_mask(mask, kernel_size=ks1, dilation_iter=di1)
    row1.append(mask * 255)

    kd2_cols = st.columns(2)
    with kd2_cols[0]:
        ks2 = st.slider(
            "Dilate Down Kernel Size",
            min_value=1,
            max_value=50,
            value=20,
            step=1,
            help="Kernel size for dilating down the mask",
        )
    with kd2_cols[1]:
        di2 = st.slider(
            "Dilate Down Dilation Iterations",
            min_value=1,
            max_value=50,
            value=15,
            step=1,
            help="Number of iterations for dilation down of the mask.",
        )
    mask = dilate_mask_down(mask, kernel_size=ks2, dilation_iter=di2)
    row1.append(mask * 255)
    row2.append(face_mask)
    fill_ks = st.slider(
        "Face Mask Threshold",
        min_value=1,
        max_value=20,
        value=7,
        step=1,
        help="Threshold for the face mask.",
    )
    face_mask_fill = fill_holes_and_close(face_mask, kernal_size=fill_ks)
    row2.append(face_mask_fill * 255)
    mask = subtract_face_from_mask(mask, face_mask_fill)
    row2.append(mask * 255)
    return row1, row2


@st.fragment
def inpaint():
    st.header("Inpainting")
    assert (
        st.session_state.get("inpaint_mask", None) is not None
    ), "Inpaint mask is not ready."
    inpaint_mask = st.session_state.inpaint_mask
    image = st.session_state.image
    inpaint_mask = Image.fromarray(inpaint_mask.astype(np.uint8))
    prompt = st.text_area("Inpainting Prompt", "A cute girl with chin-length hair")
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
            help="Number of DDIM steps for the inpainting model.",
        )
    result = prompt_inpaint(
        image=image,
        mask=inpaint_mask,
        prompt=prompt,
        scale=scale,
        ddim_steps=ddim_steps,
        w=st.session_state.image.width,
        h=st.session_state.image.height,
    )
    st.session_state.update({"result": result})
    if "result" in st.session_state:
        remap_face()


@st.fragment
def remap_face():
    st.header("Remap Face Segment")
    assert (
        st.session_state.get("face_segment", None) is not None
    ), "Face segment is not ready."
    assert (
        st.session_state.get("result", None) is not None
    ), "Inpainted image is not ready."
    face_segment = st.session_state.face_segment
    result = st.session_state.result
    face_segment_3d = np.stack([face_segment] * 3, axis=-1)
    remap_result = np.where(
        face_segment_3d > 0,
        st.session_state.image,
        np.array(result),
    )
    remap_result = Image.fromarray(remap_result.astype(np.uint8))
    st.session_state.update({"remapped_result": remap_result})

    cols = st.columns(3)
    cols[0].image(
        st.session_state.image,
        caption="Original Image",
        use_container_width=True,
    )
    cols[1].image(
        st.session_state.result,
        caption="Inpainted Image",
        use_container_width=True,
    )
    cols[2].image(
        st.session_state.remapped_result,
        caption="Remapped Face Segment",
        use_container_width=True,
    )


def main():
    print(
        "=" * 50
        + datetime.datetime.now().strftime(" %Y-%m-%d %H:%M:%S ")
        + " - Starting Streamlit app... "
        + "=" * 50
    )
    print(st.session_state)

    st.title("Deep Learning Final Project")
    st.markdown(
        "This is a Streamlit app for the final project of the Deep Learning course. "
        "We will use Segment Anything Model (SAM) to draw points on an image and generate masks."
    )
    st.button(
        "Reset",
        on_click=lambda: (
            st.session_state.clear(),
            st.cache_data.clear(),
            st.rerun(),
        ),
    )

    global MAX_SIZE
    MAX_SIZE = 640
    global upload_image
    upload_image = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "webp"]
    )

    if upload_image:
        sam2()
    if (
        st.session_state.get("face_mask", None) is not None
        and st.session_state.get("hair_mask", None) is not None
    ):
        st.success("Both face and hair masks are ready for further processing.")
        mask_generation()

    if st.session_state.get("inpaint_mask", None) is not None:
        st.success("Inpaint mask is ready for inpainting.")
        inpaint()
        st.button(
            "Rerun",
            key="rerun_button",
            help="Click to generate again with the same image and settings.",
            icon="🔄",
            on_click=lambda: st.cache_data.clear(),
        )


if __name__ == "__main__":
    main()
