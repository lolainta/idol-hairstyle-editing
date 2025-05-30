import datetime
import hashlib
from PIL import Image
from sam import segment
import streamlit as st
from streamlit_drawable_canvas import st_canvas


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


MAX_SIZE = 640
upload_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


def hash_image(image):
    sha1 = hashlib.sha1()
    sha1.update(image.read())
    return sha1.hexdigest()


@st.fragment
def sam2():
    global upload_image
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


if upload_image:
    sam2()

if (
    st.session_state.get("face_segment", None) is not None
    and st.session_state.get("hair_segment", None) is not None
):
    st.success("Both face and hair segments are ready for further processing.")
