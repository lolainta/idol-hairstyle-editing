import gc
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
import importlib
import streamlit as st


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


@st.cache_resource
def load_model_from_config(config_path, ckpt_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu")["state_dict"], strict=False
    )
    model.to(get_device()).eval()
    return DDIMSampler(model)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_batch_sd(image, mask, prompt, device, num_samples=1):
    image = image.convert("RGB")
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)

    image_np = np.array(image).astype(np.float32) / 127.5 - 1.0
    image_np = image_np.transpose(2, 0, 1)[None, ...]
    image_tensor = torch.from_numpy(image_np).to(device)

    mask_np = np.array(mask).astype(np.float32) / 255.0
    mask_np = (mask_np > 0.5).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_np[None, None, ...]).to(device)

    debug_mask = (mask_np * 255).astype(np.uint8)
    Image.fromarray(debug_mask).save("debug_input_mask.png")

    masked_image = image_tensor * (1 - mask_tensor)

    batch = {
        "image": repeat(image_tensor, "1 ... -> n ...", n=num_samples),
        "mask": repeat(mask_tensor, "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image, "1 ... -> n ...", n=num_samples),
        "txt": [prompt] * num_samples,
    }
    return batch


@st.cache_data
def prompt_inpaint(
    image,
    mask,
    prompt,
    scale,
    ddim_steps,
    w,
    h,
    seed=0,
    num_samples=1,
):
    sampler = load_model_from_config(
        "runway-stable-diffusion-inpainting/configs/stable-diffusion/v1-inpainting-inference.yaml",
        "sd-v1-5-inpainting.ckpt",
    )
    device = get_device()
    model = sampler.model

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        with torch.autocast("cuda"):
            batch = make_batch_sd(
                image, mask, prompt, device=device, num_samples=num_samples
            )

            c = model.cond_stage_model.encode(batch["txt"])

            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck].float()
                if ck != model.masked_image_key:
                    bchw = [num_samples, 4, h // 8, w // 8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            shape = [model.channels, h // 8, w // 8]
            samples_cfg, intermediates = sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=1.0,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc_full,
                x_T=start_code,
            )
            x_samples_ddim = model.decode_first_stage(samples_cfg)

            result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            result = result.cpu().numpy().transpose(0, 2, 3, 1)
            result = result * 255

    result = [Image.fromarray(img.astype(np.uint8)) for img in result]
    return result[0]
