import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from sam import segment


def get_segmented_image(image: Image.Image, key: str):
    get_point_from_canvas(image, key)
    if st.session_state.get(f"{key}_point") is None:
        st.warning("Please draw a point on the image to segment the face.")
    if st.session_state.get(f"{key}_point") is not None:
        chose_mask(image, key)


def chose_mask(image: Image.Image, key: str):
    st.info(
        f"{key.capitalize()} point already drawn. Please select a mask to segment the {key}."
    )
    segments = segment(image, st.session_state[f"{key}_point"])
    if st.session_state.get(f"{key}_mask") is None:
        cols = st.columns(3)
        for i, mask in enumerate(segments, start=1):
            cols[i - 1].image(
                mask,
                caption=f"Mask {i}",
                use_container_width=True,
            )
            cols[i - 1].button(
                f"Select Mask {i}",
                key=f"{key}_mask_{i}",
                help=f"Click to select this mask for {key} segmentation.",
                on_click=lambda m=mask: st.session_state.update({f"{key}_mask": m}),
            )
    else:
        st.image(
            st.session_state[f"{key}_mask"],
            caption=f"Selected {key.capitalize()} Mask",
            use_container_width=True,
        )


def get_point_from_canvas(image: Image.Image, key: str):
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
        key=f"{key}_canvas",
    )
    if canvas_result.json_data is not None:
        results = canvas_result.json_data["objects"]
        points = [(r["left"] * scale, r["top"] * scale) for r in results]
        if points:
            st.session_state.update({f"{key}_point": points})
    # if st.session_state.get(f"{key}_point") is not None:
    #     st.success(f"{key.capitalize()} point drawn successfully!")
    #     chose_mask(image, key)
