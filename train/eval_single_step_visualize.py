"""
Visualize + quantify ControlNet single-step reconstruction across 4 model
variants, sweeping all 1000 diffusion timesteps.

For a single test image (npz + mask png), per variant
    (v_prediction, epsilon) × (official ControlNet, our fine-tuned ControlNet)
we:
  1. Run forward diffusion  z_t = scheduler.add_noise(z_0, ε, t)  for t = 0..999.
  2. Single-step denoise via  ẑ_0 = scheduler.step(model_pred, t, z_t).pred_original_sample.
  3. VAE-decode ẑ_0 to pixel-space reconstruction x̂_0.
  4. Run frozen SegMAN on x̂_0 → Dice vs the GT cup / rim mask.

Outputs (per variant):
  - forward_noise_grid.png   : z_t decoded, showing forward noising progression
  - recon_grid.png           : x̂_0 single-step reconstructions at sampled t
  - recon_segmask_grid.png   : SegMAN predictions on the recons (RGB coded)
  - metrics.csv              : per-t Dice(cup), Dice(rim), SSIM vs clean
  - dice_curve.png           : curves of Dice(cup)/Dice(rim)/SSIM vs t
Combined:
  - compare_all_variants.png : Dice(cup) + SSIM vs t across the 4 variants.

This script is standalone — it imports from train/utils.py only `get_reward_model`
and does NOT touch reward_control.py. Training code is untouched.
"""

from __future__ import annotations
import os
from pathlib import Path

# ----- Force HF-hub OFFLINE *before* any huggingface_hub / diffusers / transformers
# ----- import caches the online state. If the required caches are on disk, this
# ----- avoids hf-mirror / hf.co HEAD-retry storms at model load time.
_CACHE_ROOT = Path(os.path.expanduser("~/.cache/huggingface/hub"))
_REQUIRED_CACHES = [
    "models--stabilityai--stable-diffusion-2-1",
    "models--thibaud--controlnet-sd21-ade20k-diffusers",
]
if all((_CACHE_ROOT / r).is_dir() for r in _REQUIRED_CACHES):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
    _OFFLINE_ENABLED = True
else:
    _OFFLINE_ENABLED = False

import argparse
import gc
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image

# Heavy deps — imported AFTER env vars are set
import cv2
from diffusers import (
    AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from utils import (
    get_reward_model,                      # SegMAN loader
    calculate_cup_disc_ratio_from_mask,    # Fu-ellipse vCDR (numpy mask → scalar)
)


# ======================================================================
# Model variants
# ======================================================================
# Paths for our fine-tuned ControlNets live on the production server:
V_PRED_FT_PATH = "./checkpoints/controlnet_vpred"
EPS_FT_CKPT_DIR = "./checkpoints/controlnet_eps"
# subdir selected via CLI: 'controlnet' or 'controlnet_ema'

MODEL_VARIANTS = [
    {
        "name": "v_pred_pretrained",
        "label": "v-prediction + official ControlNet (thibaud)",
        "pred_type": "v_prediction",
        "base_model": "stabilityai/stable-diffusion-2-1",
        "controlnet_path": "thibaud/controlnet-sd21-ade20k-diffusers",
    },
    {
        "name": "v_pred_finetuned",
        "label": "v-prediction + fine-tuned ControlNet",
        "pred_type": "v_prediction",
        "base_model": "stabilityai/stable-diffusion-2-1",
        "controlnet_path": V_PRED_FT_PATH,
    },
    {
        "name": "eps_pretrained",
        "label": "epsilon + official ControlNet (thibaud)",
        "pred_type": "epsilon",
        "base_model": "stabilityai/stable-diffusion-2-1",
        "controlnet_path": "thibaud/controlnet-sd21-ade20k-diffusers",
    },
    {
        "name": "eps_finetuned",
        "label": "epsilon + fine-tuned ControlNet",
        "pred_type": "epsilon",
        "base_model": "stabilityai/stable-diffusion-2-1-base",
        "controlnet_path": EPS_FT_CKPT_DIR,  # will be joined with subdir
    },
]


# ======================================================================
# Data loading
# ======================================================================

def _apply_clahe_per_channel(rgb_uint8, clip_limit=2.0, tile=(8, 8)):
    """Match the training-time CLAHE: applied per channel (same as training)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    out = np.zeros_like(rgb_uint8)
    for c in range(3):
        out[:, :, c] = clahe.apply(rgb_uint8[:, :, c])
    return out


def _render_condition_image(clahe_rgb, mask_signed, mask_only=False):
    """Draw the cup/rim as red/blue on CLAHE background (RGB order).

    mask_signed : (H, W) int with values {0=bg, -1=rim, -2=cup}
    """
    palette_rgb = {-2: (255, 0, 0),    # cup = red
                   -1: (0, 0, 255)}    # rim = blue
    if mask_only:
        cond = np.zeros_like(clahe_rgb)
    else:
        cond = clahe_rgb.copy()
    cond[mask_signed == -1] = palette_rgb[-1]
    cond[mask_signed == -2] = palette_rgb[-2]
    return cond


def load_sample(npz_path, mask_png_path=None, mask_only=False, crop_size=512):
    """Load a fundus sample from npz + mask png, in training-time format.

    Returns a dict with:
      pixel_values           : (1, 3, H, W) in [-1, 1]    CLAHE-enhanced fundus
      conditioning_pixel_values : (1, 3, H, W) in [0, 1]  mask overlay (control input)
      labels                 : (1, H, W) int64 {0=bg, 1=rim, 2=cup}
      original_rgb           : (H, W, 3) uint8 raw fundus (for visualization)
      clahe_rgb              : (H, W, 3) uint8 CLAHE'd fundus (for visualization)
      cond_rgb               : (H, W, 3) uint8 mask-overlay (for visualization)
    """
    data = np.load(npz_path, allow_pickle=True)
    fundus = data['slo_fundus']                              # grayscale in practice

    # slo_fundus is usually (H, W) grayscale; stack to 3 channels like dataloader does
    if fundus.ndim == 2:
        fundus = np.stack([fundus, fundus, fundus], axis=-1)
    fundus_rgb = fundus.astype(np.uint8)

    # ------- mask loading -------
    # Priority:
    #   1. explicit mask_png_path if given AND it exists
    #   2. sibling <dataset_root>/mask/<stem>_predict.png auto-detect
    #   3. npz['disc_cup_mask'] (what FairSeg training uses)
    if mask_png_path is not None and not Path(mask_png_path).exists():
        print(f"[warn] mask_png_path={mask_png_path} does not exist — ignoring")
        mask_png_path = None

    if mask_png_path is None:
        npz = Path(npz_path)
        # Try common mask layouts in priority order:
        #   1. <dataset_root>/mask_png/<name>.png
        #   2. <dataset_root>/mask/<name>_predict.png
        # where <dataset_root> could be npz.parent or npz.parent.parent (if npz is in All/).
        candidates = []
        for guess_root in [npz.parent, npz.parent.parent]:
            candidates.append(guess_root / "mask_png" / (npz.stem + ".png"))
            candidates.append(guess_root / "mask" / (npz.stem + "_predict.png"))
        for cand in candidates:
            if cand.exists():
                mask_png_path = str(cand)
                print(f"[info] auto-detected mask png: {mask_png_path}")
                break
        if mask_png_path is None and 'disc_cup_mask' not in data.files:
            tried = "\n  ".join(str(c) for c in candidates)
            raise FileNotFoundError(
                f"No mask png found — tried:\n  {tried}\n"
                f"and npz has no 'disc_cup_mask' field."
            )

    if mask_png_path is not None:
        m_raw = np.array(Image.open(mask_png_path))
        if m_raw.ndim == 3 and m_raw.shape[-1] >= 3:
            # RGB-coded mask (palette). Red = cup, blue = rim (matches training palette).
            r, g, b = m_raw[..., 0], m_raw[..., 1], m_raw[..., 2]
            mask_signed = np.zeros(m_raw.shape[:2], dtype=np.int64)
            mask_signed[(r > 128) & (g < 96) & (b < 96)] = -2   # red → cup
            mask_signed[(r < 96) & (g < 96) & (b > 128)] = -1   # blue → rim
            enc = "RGB palette"
        else:
            m = m_raw.astype(float) if m_raw.ndim == 2 else m_raw[..., 0].astype(float)
            uniq = set(int(v) for v in np.unique(m))
            if uniq.issubset({0, 1, 2}):
                # class-id encoding: 0=bg, 1=rim, 2=cup
                mask_signed = np.zeros_like(m, dtype=np.int64)
                mask_signed[m == 1] = -1
                mask_signed[m == 2] = -2
                enc = "class-id {0,1,2}"
            elif uniq.issubset({0, 128, 255}):
                # grayscale mask encoding
                mask_signed = np.zeros_like(m, dtype=np.int64)
                mask_signed[m == 128] = -1
                mask_signed[m == 255] = -2
                enc = "grayscale {0,128,255}"
            else:
                raise ValueError(
                    f"Unrecognized mask encoding at {mask_png_path}. "
                    f"Unique pixel values: {sorted(uniq)[:10]}... "
                    f"Expected one of: class-id {{0,1,2}}, grayscale {{0,128,255}}, "
                    f"or RGB (red=cup, blue=rim)."
                )
        print(f"[info] mask png encoding: {enc}")
    else:
        # npz-embedded mask, already in {0, -1, -2} format
        ms = data['disc_cup_mask']
        mask_signed = np.zeros_like(ms, dtype=np.int64)
        mask_signed[ms == -1] = -1
        mask_signed[ms == -2] = -2
        print(f"[info] using mask from npz['disc_cup_mask']")

    # Sanity: count cup/rim pixels
    n_cup = int((mask_signed == -2).sum())
    n_rim = int((mask_signed == -1).sum())
    if n_cup == 0 and n_rim == 0:
        raise RuntimeError(
            f"Mask parsing produced zero cup AND zero rim pixels. "
            f"Mask source: {mask_png_path or 'npz'}. Encoding detection likely failed."
        )
    print(f"[info] mask pixel counts: cup={n_cup}, rim={n_rim}")

    # ------- CLAHE enhancement (per-channel, matches training) -------
    clahe_rgb = _apply_clahe_per_channel(fundus_rgb)

    # ------- render conditioning image (RGB) -------
    cond_rgb = _render_condition_image(clahe_rgb, mask_signed, mask_only=mask_only)

    # ------- center crop to crop_size × crop_size -------
    H, W = clahe_rgb.shape[:2]
    y0 = (H - crop_size) // 2
    x0 = (W - crop_size) // 2
    y1, x1 = y0 + crop_size, x0 + crop_size

    clahe_rgb_c = clahe_rgb[y0:y1, x0:x1]
    cond_rgb_c  = cond_rgb[y0:y1, x0:x1]
    fundus_c    = fundus_rgb[y0:y1, x0:x1]
    mask_c      = mask_signed[y0:y1, x0:x1]

    # ------- to tensors -------
    #   pixel_values           : (1, 3, H, W) in [-1, 1]
    #   conditioning_pixel_values : (1, 3, H, W) in [0, 1]
    pixel_values = (clahe_rgb_c.astype(np.float32) / 127.5) - 1.0
    conditioning_pixel_values = cond_rgb_c.astype(np.float32) / 255.0

    pixel_values_t = torch.from_numpy(pixel_values).permute(2, 0, 1).unsqueeze(0).contiguous()
    conditioning_pixel_values_t = torch.from_numpy(conditioning_pixel_values).permute(2, 0, 1).unsqueeze(0).contiguous()

    labels = np.zeros_like(mask_c, dtype=np.int64)
    labels[mask_c == -1] = 1
    labels[mask_c == -2] = 2
    labels_t = torch.from_numpy(labels).unsqueeze(0)

    return dict(
        pixel_values=pixel_values_t,
        conditioning_pixel_values=conditioning_pixel_values_t,
        labels=labels_t,
        original_rgb=fundus_c,
        clahe_rgb=clahe_rgb_c,
        cond_rgb=cond_rgb_c,
    )


# ======================================================================
# Pipeline load / free
# ======================================================================

def load_pipeline(variant, device, dtype):
    base = variant["base_model"]
    cn_path = variant["controlnet_path"]
    print(f"[load] {variant['name']}  base={base}  controlnet={cn_path}")

    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    if noise_scheduler.config.prediction_type != variant["pred_type"]:
        raise RuntimeError(
            f"Scheduler prediction_type={noise_scheduler.config.prediction_type} "
            f"does not match variant['pred_type']={variant['pred_type']} for base {base}"
        )

    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        base, subfolder="text_encoder", torch_dtype=dtype
    ).to(device).eval()
    vae = AutoencoderKL.from_pretrained(
        base, subfolder="vae", torch_dtype=dtype
    ).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(
        base, subfolder="unet", torch_dtype=dtype
    ).to(device).eval()
    controlnet = ControlNetModel.from_pretrained(
        cn_path, torch_dtype=dtype
    ).to(device).eval()

    for p in list(vae.parameters()) + list(unet.parameters()) + \
             list(text_encoder.parameters()) + list(controlnet.parameters()):
        p.requires_grad_(False)

    return dict(scheduler=noise_scheduler, tokenizer=tokenizer,
                text_encoder=text_encoder, vae=vae, unet=unet, controlnet=controlnet)


def free_pipeline(pipe):
    for k in list(pipe.keys()):
        m = pipe[k]
        if hasattr(m, "to"):
            try:
                m.to("cpu")
            except Exception:
                pass
        del pipe[k]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ======================================================================
# Inference helpers
# ======================================================================

@torch.no_grad()
def encode_prompt(pipe, prompt, device):
    tok = pipe["tokenizer"]
    te  = pipe["text_encoder"]
    ids = tok(prompt, padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt").input_ids.to(device)
    return te(ids)[0]   # (1, 77, D)


@torch.no_grad()
def encode_image(pipe, pixel_values, device, dtype):
    vae = pipe["vae"]
    x = pixel_values.to(device, dtype=dtype)
    lat = vae.encode(x).latent_dist.sample()
    return lat * vae.config.scaling_factor


@torch.no_grad()
def single_step_reconstruct(pipe, clean_latents, cond_rgb_01, text_embeds,
                            t_int, fixed_noise, device, dtype):
    """Run one-step reconstruction at a single diffusion timestep.

    Returns:
        noisy_image: (1, 3, H, W) ∈ [0,1]   (VAE-decoded z_t, for visualization)
        recon_image: (1, 3, H, W) ∈ [0,1]   (VAE-decoded ẑ_0)
    """
    sch = pipe["scheduler"]
    un  = pipe["unet"]
    cn  = pipe["controlnet"]
    vae = pipe["vae"]

    t_tensor = torch.tensor([t_int], device=device, dtype=torch.long)
    noisy_latents = sch.add_noise(clean_latents, fixed_noise.to(dtype), t_tensor)

    cond = cond_rgb_01.to(device, dtype=dtype)

    down_res, mid_res = cn(
        noisy_latents, t_tensor,
        encoder_hidden_states=text_embeds.to(dtype),
        controlnet_cond=cond, return_dict=False,
    )
    model_pred = un(
        noisy_latents, t_tensor,
        encoder_hidden_states=text_embeds.to(dtype),
        down_block_additional_residuals=[d.to(dtype) for d in down_res],
        mid_block_additional_residual=mid_res.to(dtype),
    ).sample

    # Single-step reconstruction via scheduler.step (handles epsilon vs v_prediction internally).
    # NOTE: DDPMScheduler at t=0 is an edge case (previous_timestep=-1); pred_original_sample
    # is still defined mathematically, but guard with try/except so the loop never crashes.
    try:
        step_out = sch.step(model_pred, int(t_int), noisy_latents)
        pred_x0_lat = step_out.pred_original_sample
    except Exception:
        # Fallback: compute pred_x0 from scratch using prediction_type and alpha_cumprod
        alpha_bar = sch.alphas_cumprod.to(device)[t_int].to(dtype)
        sqrt_ab = alpha_bar.sqrt()
        sqrt_1mab = (1.0 - alpha_bar).sqrt()
        if sch.config.prediction_type == "epsilon":
            pred_x0_lat = (noisy_latents - sqrt_1mab * model_pred) / sqrt_ab.clamp(min=1e-6)
        elif sch.config.prediction_type == "v_prediction":
            pred_x0_lat = sqrt_ab * noisy_latents - sqrt_1mab * model_pred
        else:
            raise

    # Decode both noisy latent (for forward visualization) and predicted x_0.
    inv_scale = 1.0 / vae.config.scaling_factor
    recon_image = vae.decode(pred_x0_lat * inv_scale).sample
    recon_image = (recon_image / 2 + 0.5).clamp(0, 1)

    noisy_image = vae.decode(noisy_latents * inv_scale).sample
    noisy_image = (noisy_image / 2 + 0.5).clamp(0, 1)

    return noisy_image, recon_image


# ======================================================================
# SegMAN + Dice
# ======================================================================

@torch.no_grad()
def segman_predict(reward_model, image_01, device):
    """image_01: (1, 3, H, W) ∈ [0,1].  Returns argmax pred mask (1, H, W)."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (image_01.to(device, dtype=torch.float32) - mean) / std
    logits = reward_model(x)
    return logits.argmax(dim=1)


def dice_binary(pred, gt, cls, eps=1e-6):
    p = (pred == cls)
    g = (gt == cls)
    inter = (p & g).sum().item()
    denom = p.sum().item() + g.sum().item()
    if denom == 0:
        return 1.0
    return (2.0 * inter + eps) / (denom + eps)


def vcdr_from_class_mask(mask_2d_classid):
    """Compute Fu-ellipse vCDR from a class-id mask {0=bg, 1=rim, 2=cup}.

    Reuses ``calculate_cup_disc_ratio_from_mask`` (training-time canonical
    definition) which expects {0, -1=disc_rim, -2=cup} encoding.

    Returns a non-negative float; returns 0.0 when the mask is invalid
    (cup not inside disc, or ellipse fit fails).
    """
    if isinstance(mask_2d_classid, torch.Tensor):
        mask_2d_classid = mask_2d_classid.detach().cpu().numpy()
    if mask_2d_classid.ndim == 3:
        mask_2d_classid = mask_2d_classid[0]
    # Convert {0, 1, 2} → {0, -1, -2}
    signed = np.zeros_like(mask_2d_classid, dtype=np.float32)
    signed[mask_2d_classid == 1] = -1.0
    signed[mask_2d_classid == 2] = -2.0
    try:
        return float(calculate_cup_disc_ratio_from_mask(signed))
    except Exception:
        return 0.0


def ssim_np(a, b, data_range=1.0):
    """Best-effort SSIM via skimage; NaN if skimage not available."""
    try:
        from skimage.metrics import structural_similarity as sk
        return float(sk(a, b, channel_axis=2, data_range=data_range))
    except Exception:
        return float("nan")


# ======================================================================
# Visualization
# ======================================================================

def tensor_to_pil(img_chw_01):
    """img_chw_01: (3, H, W) ∈ [0,1]."""
    arr = (img_chw_01.detach().cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr.transpose(1, 2, 0))


def save_image_grid(images, labels, out_path, n_per_row=5):
    import matplotlib.pyplot as plt
    n = len(images)
    n_rows = (n + n_per_row - 1) // n_per_row
    fig, axes = plt.subplots(n_rows, n_per_row, figsize=(n_per_row * 2.4, n_rows * 2.6))
    axes = np.atleast_2d(axes)
    for i in range(n_rows * n_per_row):
        r, c = divmod(i, n_per_row)
        ax = axes[r, c]
        ax.axis("off")
        if i < n:
            ax.imshow(images[i])
            ax.set_title(labels[i], fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


def save_dice_curve(df, clean_dice_cup, clean_dice_rim, title, out_path):
    """Dice curves for both single-step recon and forward-noisy z_t.

    If the dataframe has ``dice_cup_noisy`` / ``dice_rim_noisy`` columns
    (added by the noisy-side SegMAN pass), these are drawn as dashed curves
    so the figure directly contrasts:
        - solid = Dice(SegMAN(ẑ_0), GT)   ← value of single-step recon
        - dashed = Dice(SegMAN(z_t), GT)  ← baseline without recon
    """
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(9, 4))
    # Recon-side Dice (solid)
    ax1.plot(df.t, df.dice_cup, color="#d62728", label="Dice cup (recon)")
    ax1.plot(df.t, df.dice_rim, color="#1f77b4", label="Dice rim (recon)")
    # Noisy-side Dice (dashed) — only if present
    if "dice_cup_noisy" in df.columns:
        ax1.plot(df.t, df.dice_cup_noisy, color="#d62728", ls="--", alpha=0.65,
                 label="Dice cup (noisy z_t)")
    if "dice_rim_noisy" in df.columns:
        ax1.plot(df.t, df.dice_rim_noisy, color="#1f77b4", ls="--", alpha=0.65,
                 label="Dice rim (noisy z_t)")
    # Clean reference lines
    ax1.axhline(clean_dice_cup, color="#d62728", ls=":", alpha=0.5, lw=1,
                label=f"clean Dice cup = {clean_dice_cup:.3f}")
    ax1.axhline(clean_dice_rim, color="#1f77b4", ls=":", alpha=0.5, lw=1,
                label=f"clean Dice rim = {clean_dice_rim:.3f}")
    ax1.set_xlabel("timestep t")
    ax1.set_ylabel("Dice")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title(title, fontsize=10)
    ax1.legend(loc="lower left", fontsize=7, ncol=2)

    ax2 = ax1.twinx()
    ax2.plot(df.t, df.ssim, color="#2ca02c", alpha=0.6, label="SSIM(recon, clean)")
    ax2.set_ylabel("SSIM")
    ax2.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_vcdr_curve(df, vcdr_gt, title, out_path):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(df.t, df.vcdr_pred, color="#9467bd", label="vCDR(recon)")
    ax1.axhline(vcdr_gt, color="k", ls=":", alpha=0.6, lw=1.2,
                label=f"GT vCDR = {vcdr_gt:.4f}")
    ax1.set_xlabel("timestep t")
    ax1.set_ylabel("vCDR")
    ax1.set_ylim(-0.02, max(0.85, df.vcdr_pred.max() * 1.1))
    ax1.set_title(title, fontsize=10)
    ax1.legend(loc="lower left", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(df.t, df.vcdr_err, color="#ff7f0e", alpha=0.7, label="|vCDR err|")
    ax2.set_ylabel("|vCDR error|")
    ax2.set_ylim(0, max(0.5, df.vcdr_err.max() * 1.1))
    ax2.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_path", required=True,
                    help="Input .npz file containing 'slo_fundus' (and optionally 'disc_cup_mask').")
    ap.add_argument("--mask_png", default=None,
                    help="Optional path to mask .png (values {0,128,255}). "
                         "If omitted, tries <npz_dir>/mask/<stem>_predict.png, "
                         "then falls back to npz['disc_cup_mask'].")
    ap.add_argument("--output_dir", required=True,
                    help="Directory to write visualizations and metrics.")
    ap.add_argument("--segman_config",
                    default=str(HERE / ".." / "SegMAN" / "segmentation" / "local_configs" / "segman" / "base" / "segman_b_ade.py"))
    ap.add_argument("--segman_ckpt",
                    default=str(HERE / ".." / "checkpoints" / "segman_b" / "segman_b.pth"))
    ap.add_argument("--eps_ckpt_subdir", default="controlnet_ema",
                    choices=["controlnet", "controlnet_ema"],
                    help="Which subdir under epsilon checkpoint-6000 to load (EMA recommended).")
    ap.add_argument("--vpred_ft_path", default=None,
                    help="Override v-prediction fine-tuned ControlNet path. "
                         "If set, replaces the default V_PRED_FT_PATH for the 'v_pred_finetuned' variant.")
    ap.add_argument("--mask_only", action="store_true",
                    help="Use pure mask (black bg) as the ControlNet conditioning "
                         "instead of CLAHE bg + mask overlay.")
    ap.add_argument("--prompt", default="fundus image with optic disc and optic cup",
                    help="Text prompt for CFG conditioning.")
    ap.add_argument("--n_timesteps", type=int, default=1000,
                    help="Number of timesteps to evaluate (sampled evenly in [0, 999]). "
                         "Default 1000 = full sweep. Each step writes one PNG per image type.")
    ap.add_argument("--save_images_every", type=int, default=1,
                    help="Save a PNG every K timesteps (1 = save all). "
                         "The CSV and dice_curve.png always include every evaluated t.")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="Subset of variant names to run (default: all 4).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Noise seed (same across variants for apples-to-apples comparison).")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if (args.dtype == "fp16" and device.type == "cuda") else torch.float32
    print(f"[device] {device}  dtype={dtype}")
    print(f"[env] HF OFFLINE mode: {_OFFLINE_ENABLED}"
          + ("  (both caches present)" if _OFFLINE_ENABLED else "  (some cache missing; will use network)"))

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # ----- Select variants -----
    variants = [v.copy() for v in MODEL_VARIANTS]
    if args.variants:
        variants = [v for v in variants if v["name"] in args.variants]
    # Append the chosen subdir for eps_finetuned
    for v in variants:
        if v["name"] == "eps_finetuned":
            v["controlnet_path"] = str(Path(v["controlnet_path"]) / args.eps_ckpt_subdir)
        if v["name"] == "v_pred_finetuned" and args.vpred_ft_path:
            v["controlnet_path"] = args.vpred_ft_path
            v["label"] = f"v-prediction + fine-tuned ControlNet (custom: {Path(args.vpred_ft_path).name})"

    # ----- Load sample -----
    print(f"[data] {args.npz_path}")
    sample = load_sample(args.npz_path, mask_png_path=args.mask_png,
                         mask_only=args.mask_only)
    pixel_values = sample["pixel_values"]
    cond = sample["conditioning_pixel_values"]
    gt_labels = sample["labels"].to(device)

    # Save input previews
    Image.fromarray(sample["original_rgb"]).save(out / "00_input_original.png")
    Image.fromarray(sample["clahe_rgb"]).save(out / "00_input_clahe.png")
    Image.fromarray(sample["cond_rgb"]).save(out / "00_input_conditioning.png")

    # ----- Timestep schedules -----
    ts_dense = np.unique(np.linspace(0, 999, args.n_timesteps).round().astype(int))
    # Save PNG every `save_images_every` evaluated t's (1 = save all)
    save_stride = max(1, int(args.save_images_every))
    ts_save = ts_dense[::save_stride]
    print(f"[t] evaluating {len(ts_dense)} timesteps, saving {len(ts_save)} PNGs per image type")

    # ----- Load SegMAN ONCE (reused across variants) -----
    print("[load] SegMAN")
    reward_model_path = f"segman::{args.segman_config}::{args.segman_ckpt}"
    reward_model = get_reward_model("segmentation", reward_model_path, device=str(device))
    reward_model.to(device).eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)

    # ----- Upper-bound Dice: SegMAN on the clean fundus itself -----
    with torch.no_grad():
        clean_image_01 = ((pixel_values.to(device, dtype=torch.float32)) / 2 + 0.5).clamp(0, 1)
        pred_clean = segman_predict(reward_model, clean_image_01, device)
    dc_cup_clean = dice_binary(pred_clean, gt_labels, 2)
    dc_rim_clean = dice_binary(pred_clean, gt_labels, 1)
    # vCDR from GT mask (clinical reference) and from SegMAN-on-clean (upper bound)
    vcdr_gt        = vcdr_from_class_mask(gt_labels[0])
    vcdr_clean_seg = vcdr_from_class_mask(pred_clean[0])
    print(f"[clean-image upper bound] SegMAN Dice: cup={dc_cup_clean:.3f}, rim={dc_rim_clean:.3f}")
    print(f"[vCDR reference] GT mask vCDR = {vcdr_gt:.4f}    SegMAN-on-clean vCDR = {vcdr_clean_seg:.4f}    "
          f"|err| = {abs(vcdr_gt - vcdr_clean_seg):.4f}")

    # Save clean segmentation for reference
    seg_rgb_clean = np.zeros((clean_image_01.shape[-2], clean_image_01.shape[-1], 3), dtype=np.uint8)
    pc = pred_clean[0].cpu().numpy()
    seg_rgb_clean[pc == 1] = (0, 0, 255)
    seg_rgb_clean[pc == 2] = (255, 0, 0)
    Image.fromarray(seg_rgb_clean).save(out / "00_segman_on_clean.png")

    # Precompute a fixed noise tensor in latent shape (same for all variants / timesteps).
    # We need its shape which depends on VAE; cache once after loading first pipe.
    fixed_noise = None
    clean_np = clean_image_01[0].cpu().float().numpy().transpose(1, 2, 0)  # (H, W, 3)

    # ----- Run each variant -----
    all_metrics = {}
    for variant in variants:
        print(f"\n============================================================")
        print(f"  {variant['label']}")
        print(f"============================================================")

        pipe = load_pipeline(variant, device, dtype)

        # Encode ONCE per variant
        latents = encode_image(pipe, pixel_values, device, dtype)
        text_embeds = encode_prompt(pipe, args.prompt, device)

        if fixed_noise is None or fixed_noise.shape != latents.shape:
            g = torch.Generator(device=device).manual_seed(args.seed)
            fixed_noise = torch.randn(latents.shape, generator=g, device=device, dtype=torch.float32)

        rows = []
        ts_save_set = set(ts_save.tolist())

        # Prefix + per-image-type subdirs for the 1000-per-type PNG flood
        prefix_map = {
            "v_pred_pretrained": "A1_vPred_preTrained",
            "v_pred_finetuned":  "A2_vPred_FineTuned",
            "eps_pretrained":    "B1_Eps_preTrained",
            "eps_finetuned":     "B2_Eps_FineTuned",
        }
        prefix = prefix_map.get(variant["name"], variant["name"])
        dir_forward         = out / f"{prefix}__forward";          dir_forward.mkdir(parents=True, exist_ok=True)
        dir_forward_segmask = out / f"{prefix}__forward_segmask";  dir_forward_segmask.mkdir(parents=True, exist_ok=True)
        dir_recon           = out / f"{prefix}__recon";            dir_recon.mkdir(parents=True, exist_ok=True)
        dir_segmask         = out / f"{prefix}__segmask";          dir_segmask.mkdir(parents=True, exist_ok=True)

        for idx, t in enumerate(ts_dense):
            t_int = int(t)
            noisy_img, recon_img = single_step_reconstruct(
                pipe, latents, cond, text_embeds, t_int,
                fixed_noise, device, dtype,
            )

            # ---- SegMAN on single-step reconstruction ẑ_0 ----
            pred = segman_predict(reward_model, recon_img, device)
            dice_cup = dice_binary(pred, gt_labels, 2)
            dice_rim = dice_binary(pred, gt_labels, 1)

            recon_np = recon_img[0].cpu().float().numpy().transpose(1, 2, 0)
            ssim_val = ssim_np(clean_np, recon_np, data_range=1.0)

            # Clinical-style vCDR error: Fu-ellipse vCDR(SegMAN(recon)) vs GT mask vCDR
            vcdr_pred = vcdr_from_class_mask(pred[0])
            vcdr_err  = abs(vcdr_pred - vcdr_gt)

            # ---- SegMAN on forward noisy z_t (to show structure is lost without recon) ----
            pred_noisy = segman_predict(reward_model, noisy_img, device)
            dice_cup_noisy = dice_binary(pred_noisy, gt_labels, 2)
            dice_rim_noisy = dice_binary(pred_noisy, gt_labels, 1)
            vcdr_noisy = vcdr_from_class_mask(pred_noisy[0])
            vcdr_err_noisy = abs(vcdr_noisy - vcdr_gt)

            rows.append(dict(
                t=t_int,
                # --- recon-side (post single-step) ---
                dice_cup=dice_cup, dice_rim=dice_rim, ssim=ssim_val,
                vcdr_pred=vcdr_pred, vcdr_gt=vcdr_gt, vcdr_err=vcdr_err,
                # --- forward-side (noisy z_t decoded, no recon) ---
                dice_cup_noisy=dice_cup_noisy, dice_rim_noisy=dice_rim_noisy,
                vcdr_noisy=vcdr_noisy, vcdr_err_noisy=vcdr_err_noisy,
            ))

            if t_int in ts_save_set:
                # Save 4 PIL images directly (no matplotlib grid)
                tag = f"t_{t_int:04d}"
                tensor_to_pil(noisy_img[0]).save(dir_forward / f"{tag}.png", optimize=False)
                tensor_to_pil(recon_img[0]).save(dir_recon   / f"{tag}.png", optimize=False)

                # segmask on recon
                seg_rgb = np.zeros((pred.shape[-2], pred.shape[-1], 3), dtype=np.uint8)
                pm = pred[0].cpu().numpy()
                seg_rgb[pm == 1] = (0, 0, 255)
                seg_rgb[pm == 2] = (255, 0, 0)
                Image.fromarray(seg_rgb).save(dir_segmask / f"{tag}.png", optimize=False)

                # segmask on forward noisy z_t
                seg_rgb_noisy = np.zeros((pred_noisy.shape[-2], pred_noisy.shape[-1], 3), dtype=np.uint8)
                pmn = pred_noisy[0].cpu().numpy()
                seg_rgb_noisy[pmn == 1] = (0, 0, 255)
                seg_rgb_noisy[pmn == 2] = (255, 0, 0)
                Image.fromarray(seg_rgb_noisy).save(dir_forward_segmask / f"{tag}.png", optimize=False)

            if (idx + 1) % 10 == 0 or idx == len(ts_dense) - 1:
                print(f"  [{idx+1}/{len(ts_dense)}] t={t_int:4d}  "
                      f"recon[dice_cup={dice_cup:.3f} rim={dice_rim:.3f} ssim={ssim_val:.3f} "
                      f"vcdr_err={vcdr_err:.3f}]  noisy[dice_cup={dice_cup_noisy:.3f} "
                      f"rim={dice_rim_noisy:.3f} vcdr_err={vcdr_err_noisy:.3f}]")

        # ----- Save per-variant metrics + dice/vcdr curves -----
        # Individual PNGs were saved inside the t-loop into <prefix>__forward / recon / segmask dirs.
        df = pd.DataFrame(rows)
        df.to_csv(out / f"{prefix}__metrics.csv", index=False)
        save_dice_curve(df, dc_cup_clean, dc_rim_clean,
                        title=variant["label"],
                        out_path=out / f"{prefix}__dice_curve.png")
        save_vcdr_curve(df, vcdr_gt,
                        title=variant["label"],
                        out_path=out / f"{prefix}__vcdr_curve.png")

        all_metrics[variant["name"]] = df
        n_saved = len(ts_save_set)
        print(f"[saved] {prefix}__{{forward,forward_segmask,recon,segmask}}/ × {n_saved} PNGs  "
              f"+ {prefix}__metrics.csv ({len(df)} rows) "
              f"+ dice_curve.png + vcdr_curve.png  in {out}/")

        free_pipeline(pipe)

    print(f"\n[DONE] all results in {out}")


if __name__ == "__main__":
    main()
