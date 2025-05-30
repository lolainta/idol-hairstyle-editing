import datetime
import hashlib
from PIL import Image
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from sam import segment
from mask_gen import dilate_mask, dilate_mask_down, subtract_face_from_mask
from inpaint import inpaint_mask


def hash_image(image):
    sha1 = hashlib.sha1()
    sha1.update(image.read())
    return sha1.hexdigest()


def sam2():
    global upload_image
    global MAX_SIZE
    image = Image.open(upload_image)
    # check sha1sum and save
    image_hash = hash_image(upload_image)
    image.save(f"data/streamlit/{image_hash}.png")
    w, h = image.size
    if w > MAX_SIZE or h > MAX_SIZE:
        factor = MAX_SIZE / max(w, h)
        w = int(factor * w)
        h = int(factor * h)
        image = image.resize((w, h))
        st.warning(f"Image resized to {w}x{h} to fit within {MAX_SIZE}px limit.")
    # st.image(image, caption="Uploaded Image")

    face_col, hair_col = st.columns(2)
    with face_col:
        st.header("Face Segment")
        if not st.session_state.get("face_point", False):
            get_face_points(image)
        else:
            # st.info(f"Face point: {st.session_state.face_point[-1]}")
            face_segment = segment(image, st.session_state.face_point)
            if not st.session_state.get("selected_face_mask", False):
                cols = st.columns(3)
                for i, mask in enumerate(face_segment, start=1):
                    cols[i - 1].image(mask, caption=f"Mask {i}")
                    cols[i - 1].button(
                        f"Select {i} Mask",
                        key=f"select_face_mask_{i}",
                        on_click=lambda i=i: st.session_state.update(
                            {
                                "selected_face_mask": i,
                                "face_segment": face_segment[i - 1],
                            }
                        ),
                    )
                st.warning("Please select a mask to proceed.")
            else:
                st.info(
                    f"You selected mask {st.session_state.selected_face_mask} for the face segment."
                )
                st.image(
                    st.session_state.face_segment,
                    caption=f"Selected Mask {st.session_state.selected_face_mask}",
                )
    with hair_col:
        st.header("Hair Segment")
        if not st.session_state.get("hair_point", False):
            get_hair_points(image)
        else:
            # st.info(f"Hair point: {st.session_state.hair_point[-1]}")
            hair_segment = segment(image, st.session_state.hair_point)
            if not st.session_state.get("selected_hair_mask", False):
                cols = st.columns(3)
                for i, mask in enumerate(hair_segment, start=1):
                    cols[i - 1].image(mask, caption=f"Mask {i}")
                    cols[i - 1].button(
                        f"Select {i} Mask",
                        key=f"select_hair_mask_{i}",
                        on_click=lambda i=i: st.session_state.update(
                            {
                                "selected_hair_mask": i,
                                "hair_segment": hair_segment[i - 1],
                            }
                        ),
                    )
                st.warning("Please select a mask to proceed.")
            else:
                st.info(
                    f"You selected mask {st.session_state.selected_hair_mask} for the hair segment."
                )
                st.image(
                    st.session_state.hair_segment,
                    caption=f"Selected Mask {st.session_state.selected_hair_mask}",
                )


def get_face_points(image):
    scale = 2
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=3,
        stroke_color="rgba(255, 0, 0, 1.0)",
        background_image=image,
        update_streamlit=True,
        height=image.height // scale,
        width=image.width // scale,
        drawing_mode="point",
        key="face_canvas",
    )
    if canvas_result.json_data is not None:
        results = canvas_result.json_data["objects"]
        points = [(r["left"] * scale, r["top"] * scale) for r in results]
        if not points:
            st.warning("Please draw at least one point on the image.")
        else:
            st.success(f"Points drawn: {points[-1]}")
            st.session_state.face_point = points
            st.rerun()


def get_hair_points(image):
    scale = 2
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=3,
        stroke_color="rgba(255, 0, 0, 1.0)",
        background_image=image,
        update_streamlit=True,
        height=image.height // scale,
        width=image.width // scale,
        drawing_mode="point",
        key="hair_canvas",
    )
    if canvas_result.json_data is not None:
        results = canvas_result.json_data["objects"]
        points = [(r["left"] * scale, r["top"] * scale) for r in results]
        if not points:
            st.warning("Please draw at least one point on the image.")
        else:
            st.success(f"Points drawn: {points[-1]}")
            st.session_state.hair_point = points
            st.rerun()


@st.fragment
def mask_generation():
    assert (
        st.session_state.get("face_segment", None) is not None
    ), "Face segment is not ready."
    assert (
        st.session_state.get("hair_segment", None) is not None
    ), "Hair segment is not ready."
    st.header("Mask Generation")
    hair_mask = st.session_state.hair_segment
    face_mask = st.session_state.face_segment
    masks = gen_mask(hair_mask, face_mask)
    cols = st.columns(len(masks))
    for i, mask in enumerate(masks, start=1):
        cols[i - 1].image(mask, caption=f"Mask {i}", use_container_width=True)
    st.session_state.update(
        {
            "inpaint_mask": masks[-1],
        }
    )


def gen_mask(hair_mask: np.ndarray, face_mask: np.ndarray) -> list:
    ret = []
    mask = hair_mask
    ret.append(mask)

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
    ret.append(mask * 255)

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
    ret.append(mask * 255)
    print(f"Mask shape: {mask.shape}")
    print(f"Face mask shape: {face_mask.shape}")
    mask = subtract_face_from_mask(mask, face_mask)
    ret.append(mask * 255)
    return ret


@st.fragment
def inpaint():
    st.header("Inpainting")
    assert (
        st.session_state.get("inpaint_mask", None) is not None
    ), "Inpaint mask is not ready."
    inpaint_mask = st.session_state.inpaint_mask
    image = Image.open(upload_image)
    w, h = image.size


def main():
    print(
        "=" * 50
        + datetime.datetime.now().strftime(" %Y-%m-%d %H:%M:%S ")
        + " - Starting Streamlit app... "
        + "=" * 50
    )

    st.title("Deep Learning Final Project")
    st.markdown(
        "This is a Streamlit app for the final project of the Deep Learning course. "
        "We will use Segment Anything Model (SAM) to draw points on an image and generate masks."
    )
    st.button("reset", on_click=lambda: st.session_state.clear())

    global MAX_SIZE
    MAX_SIZE = 640
    global upload_image
    upload_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if upload_image:
        sam2()
    if (
        st.session_state.get("face_segment", None) is not None
        and st.session_state.get("hair_segment", None) is not None
    ):
        st.success("Both face and hair segments are ready for further processing.")
        mask_generation()

    if st.session_state.get("inpaint_mask", None) is not None:
        st.success("Inpaint mask is ready for inpainting.")
        inpaint()


if __name__ == "__main__":
    main()
