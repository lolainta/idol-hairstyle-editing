import streamlit as st
import sys
import torch
from PIL import Image

sys.path.append("sam2")
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2


@st.cache_resource
def init():
    sam_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam_checkpoint))
    return predictor


def segment(image: Image.Image, points) -> list:
    print(f"Segmenting image with {len(points)} points...")
    point_cords = torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # [1, N, 2]
    point_labels = torch.ones((len(points), 1), dtype=torch.int64)
    print(f"Shape of point_coords: {point_cords.shape}")
    print(f"Shape of point_labels: {point_labels.shape}")
    input_prompts = {
        "point_coords": point_cords,  # [1, N, 2]
        "point_labels": point_labels,  # [1, N, 1]
    }
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor = init()
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(**input_prompts)
    print(
        f"Mask shape: {masks.shape}, Scores shape: {scores.shape}, Logits shape: {logits.shape}"
    )
    return [
        masks[0],
        masks[1],
        masks[2],
    ]  # Return the first mask three times for display
