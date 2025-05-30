from PIL import Image
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
