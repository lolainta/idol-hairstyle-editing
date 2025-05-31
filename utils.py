import hashlib
from PIL import Image
import streamlit as st


def hash_image(image: Image.Image) -> str:
    sha1 = hashlib.sha1()
    sha1.update(image.tobytes())
    return sha1.hexdigest()[:8]


def preprocess_image(image: Image.Image, max_size: int = 640) -> Image.Image:
    w, h = image.size
    if w > max_size or h > max_size:
        factor = max_size / max(w, h)
        w = int(factor * w)
        h = int(factor * h)
    w = w // 64 * 64
    h = h // 64 * 64
    image = image.resize((w, h))
    st.warning(f"Image resized to {w}x{h} for processing.")
    return image
