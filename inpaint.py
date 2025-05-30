import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat

# from main import instantiate_from_config
import sys

# sys.path.append("runway-stable-diffusion-inpainting/scripts")
sys.path.append("runway-stable-diffusion-inpainting")
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_tensor
import torch
import os
import cv2


def inpaint_mask():
    pass
