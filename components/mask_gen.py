import streamlit as st
import numpy as np
import cv2


def dilate_mask(
    mask: np.ndarray, kernel_size: int = 15, dilation_iter: int = 8
) -> np.ndarray:
    binary_mask = (mask > 0).astype(np.uint8)  # binary 0 or 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=dilation_iter)
    return dilated_mask


def dilate_mask_down(
    mask: np.ndarray, kernel_size: int = 15, dilation_iter: int = 1
) -> np.ndarray:
    binary_mask = (mask > 0.5).astype(np.uint8)

    kernel = np.zeros((kernel_size, 1), dtype=np.uint8)
    kernel[: kernel_size // 2 + 1] = 1

    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=dilation_iter)
    return dilated_mask


def subtract_face_from_mask(dilated_mask, face_mask):
    final_mask = dilated_mask.copy()
    final_mask[face_mask > 0.5] = 0
    return final_mask


def fill_holes_and_close(mask, kernel_size=5):
    mask = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    closed_mask = (closed_mask > 0).astype(np.uint8)
    return closed_mask


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
            "Dilate Iterations",
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
            "Dilate Down Iterations",
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
        "Face Mask Kernel Size",
        min_value=1,
        max_value=20,
        value=7,
        step=1,
        help="Kernel size for the face mask.",
    )
    face_mask_fill = fill_holes_and_close(face_mask, kernel_size=fill_ks)
    row2.append(face_mask_fill * 255)
    mask = subtract_face_from_mask(mask, face_mask_fill)
    row2.append(mask * 255)
    return row1, row2
