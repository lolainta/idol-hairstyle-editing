import hashlib
from PIL import Image
import streamlit as st
import numpy as np
import os
import uuid


def hash_image(image: Image.Image) -> str:
    sha1 = hashlib.sha1()
    sha1.update(image.tobytes())
    return sha1.hexdigest()[:8]


def preprocess_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    w, h = image.size
    if w > max_size or h > max_size:
        factor = max_size / max(w, h)
        w = int(factor * w)
        h = int(factor * h)
    w = (w + 31) // 64 * 64
    h = (h + 31) // 64 * 64
    image = image.resize((w, h))
    return image


@st.cache_data
def log_info(
    input: np.ndarray,
    mask: np.ndarray,
    inpaint: np.ndarray,
    output: np.ndarray,
    prompt: str,
):
    os.makedirs("data/streamlit/logs/baseline", exist_ok=True)
    os.makedirs("data/streamlit/logs/mask", exist_ok=True)
    os.makedirs("data/streamlit/logs/origin", exist_ok=True)
    os.makedirs("data/streamlit/logs/result", exist_ok=True)
    os.makedirs("data/streamlit/logs/prompt", exist_ok=True)
    uid = str(uuid.uuid4())
    Image.fromarray(input).save(f"data/streamlit/logs/origin/{uid}.png")
    mask_image = Image.fromarray(mask.astype(np.uint8))
    mask_image.save(f"data/streamlit/logs/mask/{uid}.png")
    Image.fromarray(inpaint).save(f"data/streamlit/logs/baseline/{uid}.png")
    Image.fromarray(output).save(f"data/streamlit/logs/result/{uid}.png")
    with open(f"data/streamlit/logs/prompt/{uid}.txt", "w") as f:
        f.write(prompt)
