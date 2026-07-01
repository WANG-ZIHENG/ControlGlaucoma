#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import os
import math
import time
import kornia
import pickle
import random
import logging
import argparse
import pandas as pd
from glob import glob
import cv2

import torch
import numpy as np
import accelerate
import transformers
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint
from skimage import measure
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, load_from_disk, Dataset
from huggingface_hub import create_repo, upload_folder
from transformers import AutoTokenizer, PretrainedConfig

from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from packaging import version
from torchvision import transforms
from torch.cuda.amp import autocast
from torchvision.transforms.functional import normalize
from scipy.optimize import minimize
from shapely.geometry import Polygon
from skimage.metrics import peak_signal_noise_ratio

from train_CLIP.clip_score.clip_score import initialize_clip_model, calculate_single_image_clip_score
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
try:
    from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
    MEDCLIP_AVAILABLE = True
except ImportError:
    MEDCLIP_AVAILABLE = False
    print("Warning: MedCLIP is not installed. Install it with https://github.com/StefanDenn3r/MedCLIP/tree/patch-1")
from scipy import ndimage
from scipy.ndimage import zoom
from pytorch_fid import fid_score
from pytorch_fid.fid_score import InceptionV3, compute_statistics_of_path, calculate_frechet_distance, calculate_activation_statistics
import json
import lpips
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM



try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.Resampling.BICUBIC

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from utils import image_grid, get_reward_model, get_reward_loss, label_transform, group_random_crop, calculate_cup_disc_ratio_difference_from_outputs, cosine_decay


def create_foreground_mask(labels, predictions, mode='union'):
    if isinstance(labels, torch.Tensor):
        if labels.ndim == 3 and labels.shape[0] == 1:
            labels = labels.squeeze(0)  # [1, H, W] -> [H, W]
        label_mask = (labels != 0).float()  # [H, W]
    else:
        label_mask = torch.tensor(labels != 0, dtype=torch.float32)
    
    if predictions is not None:
        probs = torch.nn.functional.softmax(predictions, dim=1)  # [B, C, H, W]
        pred_classes = torch.argmax(probs, dim=1)  # [B, H, W]
        pred_mask = (pred_classes[0] != 0).float().cpu()  # [H, W]
    else:
        pred_mask = label_mask
    
    if label_mask.shape != pred_mask.shape:
        import torch.nn.functional as F
        pred_mask = F.interpolate(
            pred_mask.unsqueeze(0).unsqueeze(0), 
            size=label_mask.shape, 
            mode='nearest'
        ).squeeze()
    
    if mode == 'union':
        foreground_mask = torch.max(label_mask, pred_mask)
    elif mode == 'intersection':
        foreground_mask = label_mask * pred_mask
    else:
        raise ValueError(f"Unknown mode: {mode}, expected 'union' or 'intersection'")
    
    return foreground_mask


def apply_mask_to_images(images, mask):
    masked_images = []
    mask_np = mask.cpu().numpy()
    
    for img in images:
        img_array = np.array(img)
        mask_3ch = np.stack([mask_np, mask_np, mask_np], axis=-1)
        masked_array = (img_array * mask_3ch).astype(np.uint8)
        masked_img = Image.fromarray(masked_array)
        masked_images.append(masked_img)
    
    return masked_images


from PIL import PngImagePlugin
MaximumDecompressedsize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedsize * MegaByte
Image.MAX_IMAGE_PIXELS = None

InceptionV3_model = None
loss_fn_alex = None

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__)

_global_clip_model_custom = None
_global_clip_model_baseline = None
_global_clip_model_medclip = None
_global_clip_processor_custom = None
_global_clip_processor_baseline = None
_global_clip_processor_medclip = None
_global_clip_tokenizer_custom = None
_global_clip_tokenizer_baseline = None
_global_clip_tokenizer_medclip = None
# Offloading state_dict to CPU to avoid GPU memory boom (only used for FSDP training)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)



def calculate_pairwise_metrics(generated_images, original_image, device='cuda'):
    to_t = transforms.ToTensor()

    gen_tensors = [to_t(img.convert('RGB')) for img in generated_images]
    real_tensor = to_t(original_image.convert('RGB'))

    X = torch.stack(gen_tensors).to(device) * 255.0               # (N, 3, H, W) float [0,255]
    if real_tensor.shape != gen_tensors[0].shape:
        real_tensor = torch.nn.functional.interpolate(
            real_tensor.unsqueeze(0), size=gen_tensors[0].shape[-2:],
            mode='bilinear', align_corners=False).squeeze(0)
    Y_single = real_tensor.to(device) * 255.0                     # (3, H, W)
    Y = Y_single.unsqueeze(0).expand_as(X).contiguous()           # (N, 3, H, W)

    ssim_val = ssim(X, Y, data_range=255, size_average=True)
    ms_ssim_val = ms_ssim(X, Y, data_range=255, size_average=True)

    X_lpips = (X / 255.0) * 2 - 1
    Y_lpips = (Y / 255.0) * 2 - 1
    with torch.no_grad():
        lpips_out = loss_fn_alex(X_lpips, Y_lpips).view(-1)
    lpips_alex = float(lpips_out.mean().cpu().numpy())

    X_np = X.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float64)
    Y_np = Y.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float64)
    psnr_scores = [
        peak_signal_noise_ratio(Y_np[i], X_np[i], data_range=255)
        for i in range(X_np.shape[0])
    ]
    psnr_score = float(np.mean(psnr_scores))

    return {
        'ssim':       float(ssim_val.cpu().numpy()),
        'ms_ssim':    float(ms_ssim_val.cpu().numpy()),
        'lpips_alex': lpips_alex,
        'psnr':       psnr_score,
    }


def calculate_fid_distribution(generated_images, real_images, device='cuda', batch_size=16):
    import tempfile

    n_gen = len(generated_images)
    n_real = len(real_images)
    if n_gen < 2 or n_real < 2:
        logger.warning(
            f"FID needs at least 2 samples on each side (got gen={n_gen}, real={n_real}); "
            f"returning NaN. Increase the validation set or num_validation_images."
        )
        return float('nan')

    with tempfile.TemporaryDirectory() as temp_dir:
        gen_paths = []
        for i, img in enumerate(generated_images):
            p = os.path.join(temp_dir, f"gen_{i}.png")
            img.save(p)
            gen_paths.append(p)
        real_paths = []
        for i, img in enumerate(real_images):
            p = os.path.join(temp_dir, f"real_{i}.png")
            img.save(p)
            real_paths.append(p)

        bsz = max(1, min(batch_size, n_gen, n_real))
        m1, s1 = calculate_activation_statistics(
            gen_paths, InceptionV3_model, bsz, 2048, device, 0)
        m2, s2 = calculate_activation_statistics(
            real_paths, InceptionV3_model, bsz, 2048, device, 0)
        fid = calculate_frechet_distance(m1, s1, m2, s2)
    return float(fid)


def initialize_clip_models(device=None, args=None):
    global _global_clip_model_custom, _global_clip_model_baseline, _global_clip_model_medclip
    global _global_clip_processor_custom, _global_clip_processor_baseline, _global_clip_processor_medclip
    global _global_clip_tokenizer_custom, _global_clip_tokenizer_baseline, _global_clip_tokenizer_medclip
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    need_custom = False
    need_baseline = False
    need_medclip = False
    
    if args is not None:
        if args.clip_model_for_loss == "custom" or args.clip_model_for_metric == "custom":
            need_custom = True
        if args.clip_model_for_loss == "baseline" or args.clip_model_for_metric == "baseline":
            need_baseline = True
        if args.clip_model_for_loss == "medclip" or args.clip_model_for_metric == "medclip":
            need_medclip = True
    else:
        need_custom = True
        need_baseline = True
    
    if need_custom and _global_clip_model_custom is None:
        custom_model_path = None
        if args is not None and hasattr(args, 'custom_clip_model_path'):
            custom_model_path = args.custom_clip_model_path
        
        if not custom_model_path:
            custom_model_path = "./checkpoints/custom_clip.ckpt"
            print("Custom CLIP model initialized (default path).")
        else:
            print(f"Custom CLIP model initialized (path: {custom_model_path}).")
        
        _global_clip_model_custom, _global_clip_processor_custom, _global_clip_tokenizer_custom = initialize_clip_model(
            clip_model_name=custom_model_path, device=device)
        if _global_clip_model_custom is not None:
            for _p in _global_clip_model_custom.parameters():
                _p.requires_grad_(False)
            _global_clip_model_custom.eval()

    if need_baseline and _global_clip_model_baseline is None:
        _global_clip_model_baseline, _global_clip_processor_baseline, _global_clip_tokenizer_baseline = initialize_clip_model(device=device)
        print("Pretrained CLIP model initialized.")
        if _global_clip_model_baseline is not None:
            for _p in _global_clip_model_baseline.parameters():
                _p.requires_grad_(False)
            _global_clip_model_baseline.eval()

    if need_medclip and _global_clip_model_medclip is None:


        print("Initializing MedCLIP model ...")
        _global_clip_processor_medclip = MedCLIPProcessor()
        _global_clip_model_medclip = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        _global_clip_model_medclip.from_pretrained()
        _global_clip_model_medclip.to(device)
        _global_clip_model_medclip.eval()
        for _p in _global_clip_model_medclip.parameters():
            _p.requires_grad_(False)
        _global_clip_tokenizer_medclip = None
        print("MedCLIP model initialized.")

    if _global_clip_model_custom is not None and not _global_clip_tokenizer_custom:
        if _global_clip_tokenizer_baseline is not None:
            _global_clip_tokenizer_custom = _global_clip_tokenizer_baseline
    
    return (_global_clip_model_custom, _global_clip_processor_custom, _global_clip_tokenizer_custom,
            _global_clip_model_baseline, _global_clip_processor_baseline, _global_clip_tokenizer_baseline,
            _global_clip_model_medclip, _global_clip_processor_medclip, _global_clip_tokenizer_medclip)


def calculate_clip_loss(generated_images, original_images, original_texts, device, clip_scale=1.0, clip_timestep_mask=None, args=None):
    global _global_clip_model_custom, _global_clip_processor_custom, _global_clip_tokenizer_custom
    global _global_clip_model_baseline, _global_clip_processor_baseline, _global_clip_tokenizer_baseline
    global _global_clip_model_medclip, _global_clip_processor_medclip, _global_clip_tokenizer_medclip


    # Sourced from: https://github.com/facebookresearch/swav/blob/5e073db0cc69dea22aa75e92bfdd75011e888f28/main_swav.py#L354
    def sinkhorn(out):
        Q = torch.exp(out / 0.05).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(3):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    model_type = "custom"
    if args is not None and hasattr(args, 'clip_model_for_loss'):
        model_type = args.clip_model_for_loss
    
    initialize_clip_models(device=device, args=args)
    
    if model_type == "custom":
        if _global_clip_model_custom is None:
            raise ValueError("Custom CLIP model is required but not loaded. Please set --custom_clip_model_path and ensure --clip_model_for_loss is 'custom'.")
        clip_model = _global_clip_model_custom
        clip_processor = _global_clip_processor_custom
        clip_tokenizer = _global_clip_tokenizer_custom
        is_medclip = False
    elif model_type == "medclip":
        if _global_clip_model_medclip is None:
            raise ValueError("MedCLIP model is required but not loaded. Please ensure --clip_model_for_loss is 'medclip'.")
        clip_model = _global_clip_model_medclip
        clip_processor = _global_clip_processor_medclip
        clip_tokenizer = _global_clip_tokenizer_medclip
        is_medclip = True
    elif model_type=="baseline":  # baseline
        if _global_clip_model_baseline is None:
            raise ValueError("Baseline CLIP model is required but not loaded. Please ensure --clip_model_for_loss is 'baseline'.")
        clip_model = _global_clip_model_baseline
        clip_processor = _global_clip_processor_baseline
        clip_tokenizer = _global_clip_tokenizer_baseline
        is_medclip = False
    
    batch_size = generated_images.shape[0]
    
    if clip_timestep_mask is None:
        clip_timestep_mask = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
    
    if not clip_timestep_mask.any():
        return torch.zeros(batch_size, device=device, dtype=generated_images.dtype)

    with torch.enable_grad():
        def preprocess_images_for_clip(images, is_ckpt_model=False):
            images = torch.clamp(images, 0, 1)
            
            if images.shape[-2:] != (224, 224):
                images = F.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False, antialias=True)
            
            if images.shape[1] != 3:
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                else:
                    images = images[:, :3, :, :]
            
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device).view(1, 3, 1, 1)
            images = (images - mean) / std
            
            if is_ckpt_model:
                return images
            else:
                return {'pixel_values': images}

        if is_medclip:
            def preprocess_for_medclip(images):
                images = torch.clamp(images, 0, 1)
                
                if images.shape[-2:] != (224, 224):
                    images = F.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False, antialias=True)
                
                if images.shape[1] != 3:
                    if images.shape[1] == 1:
                        images = images.repeat(1, 3, 1, 1)
                    else:
                        images = images[:, :3, :, :]
                
                return images
            
            def tensor_to_pil(tensor_images):
                tensor_images = tensor_images.to( dtype=torch.float32)
                pil_images = []
                for i in range(tensor_images.shape[0]):
                    img = tensor_images[i].cpu().clone()
                    img = torch.clamp(img, 0, 1)
                    img = transforms.ToPILImage()(img)
                    pil_images.append(img)
                return pil_images
            
            generated_images_processed = preprocess_for_medclip(generated_images)
            original_images_processed = preprocess_for_medclip(original_images)
            
            generated_pil_images = tensor_to_pil(generated_images_processed)
            original_pil_images = tensor_to_pil(original_images_processed)
            
            generated_inputs = clip_processor(images=generated_pil_images, return_tensors="pt", padding=True)
            original_inputs = clip_processor(images=original_pil_images, return_tensors="pt", padding=True)
            text_inputs = clip_processor(text=original_texts, return_tensors="pt", padding=True)
            
            for key in generated_inputs:
                if isinstance(generated_inputs[key], torch.Tensor):
                    generated_inputs[key] = generated_inputs[key].to(device)
            for key in original_inputs:
                if isinstance(original_inputs[key], torch.Tensor):
                    original_inputs[key] = original_inputs[key].to(device)
            for key in text_inputs:
                if isinstance(text_inputs[key], torch.Tensor):
                    text_inputs[key] = text_inputs[key].to(device)
            
            generated_outputs = clip_model(**generated_inputs)
            original_outputs = clip_model(**original_inputs)
            text_outputs = clip_model(**text_inputs)
            
            generated_image_features = generated_outputs['img_embeds']
            original_image_features = original_outputs['img_embeds']
            text_features = text_outputs['text_embeds']
        else:
            is_ckpt_model = hasattr(clip_model, 'model') and hasattr(clip_model.model, 'encode_image')
            
            generated_images_processed = preprocess_images_for_clip(generated_images, is_ckpt_model)
            original_images_processed = preprocess_images_for_clip(original_images, is_ckpt_model)
            
            if is_ckpt_model:
                generated_image_features = clip_model.model.encode_image(generated_images_processed)
                original_image_features = clip_model.model.encode_image(original_images_processed)
                
                import clip
                text_tokens = clip.tokenize(original_texts, truncate=True).to(device)
                text_features = clip_model.model.encode_text(text_tokens)
            else:
                generated_image_features = clip_model.get_image_features(**generated_images_processed)
                original_image_features = clip_model.get_image_features(**original_images_processed)
                
                if clip_tokenizer is not None:
                    text_tokens = clip_tokenizer(original_texts, padding=True, return_tensors='pt', truncation=True)
                else:
                    if _global_clip_tokenizer_baseline is not None:
                        text_tokens = _global_clip_tokenizer_baseline(original_texts, padding=True, return_tensors='pt', truncation=True)
                    else:
                        raise ValueError("CLIP tokenizer is not available. Please ensure the CLIP model is properly initialized.")
                for key in text_tokens:
                    text_tokens[key] = text_tokens[key].to(device)
                text_features = clip_model.get_text_features(**text_tokens)

        generated_image_features = F.normalize(generated_image_features, dim=1)
        original_image_features = F.normalize(original_image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        original_similarity = (original_image_features * text_features).sum(dim=1)  # [batch_size]
        generated_similarity = (generated_image_features * text_features).sum(dim=1)  # [batch_size]
        
        mse_loss_per_sample = F.mse_loss(
            generated_similarity,
            original_similarity,
            reduction='none'
        )  # [batch_size]
        
        clip_loss_per_sample = mse_loss_per_sample * clip_scale
        
        clip_loss_per_sample = clip_loss_per_sample * clip_timestep_mask.squeeze(1).float()
        
        return clip_loss_per_sample


def compute_loss_weights(timesteps, args, device, noise_scheduler=None):
    batch_size = timesteps.shape[0]
    timesteps_reshaped = timesteps.reshape(-1, 1)
    
    if args.loss_weight_strategy == "fixed_timestep":
        mask = (args.min_timestep_rewarding <= timesteps_reshaped) & (timesteps_reshaped <= args.max_timestep_rewarding)
        weights = mask.float()
        
    elif args.loss_weight_strategy == "cosine_decay":
        if noise_scheduler is None:
            raise ValueError("noise_scheduler is required for cosine_decay weight strategy")
        
        total_steps = noise_scheduler.config.num_train_timesteps
        
        timesteps_float = timesteps_reshaped.float()
        current_steps = total_steps - timesteps_float
        
        current_steps_np = current_steps.cpu().numpy()
        decay_values = cosine_decay(current_steps_np, start=700, end=1000)
        weights = torch.tensor(
            decay_values,
            device=device,
            dtype=torch.float32
        ).reshape(-1, 1)
        
        mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        
    elif args.loss_weight_strategy == "snr":
        if noise_scheduler is None:
            raise ValueError("noise_scheduler is required for SNR weight strategy")
        
        # SNR = alpha_cumprod / (1 - alpha_cumprod)
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
        
        batch_alphas_cumprod = alphas_cumprod[timesteps.long()]
        
        snr = batch_alphas_cumprod / (1.0 - batch_alphas_cumprod + 1e-12)
        
        snr_min = 0.0047
        snr_max = 1175

        snr_normalized = (snr - snr_min) / (snr_max - snr_min + 1e-12)
        
        weights = snr_normalized.reshape(-1, 1)
        
        mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        
    elif args.loss_weight_strategy == "snr*cosine":
        if noise_scheduler is None:
            raise ValueError("noise_scheduler is required for SNR*cosine weight strategy")
        
        total_steps = noise_scheduler.config.num_train_timesteps
        
        timesteps_float = timesteps.float()
        x = timesteps_float / total_steps  # normalized timestep in [0,1]
        
        w_cos = torch.cos(0.5 * torch.pi * x)
        w_cos = torch.clamp(w_cos, min=0.0)  # clip to [0, ∞)
        
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
        batch_alphas_cumprod = alphas_cumprod[timesteps.long()]
        
        snr = batch_alphas_cumprod / (1.0 - batch_alphas_cumprod + 1e-12)
        
        all_snr = alphas_cumprod / (1.0 - alphas_cumprod + 1e-12)
        
        snr_min = torch.quantile(all_snr, 0.10)
        snr_max = torch.quantile(all_snr, 0.90)
        
        w_snr = (snr - snr_min) / (snr_max - snr_min + 1e-12)
        w_snr = torch.clamp(w_snr, min=0.0, max=1.0)
        
        gamma = getattr(args, 'snr_cosine_gamma', 1.0)  # cosine exponent, default 1.0
        beta_snr = getattr(args, 'snr_cosine_beta_snr', 0.5)  # SNR exponent, default 0.5
        
        w_comb = (w_cos ** gamma) * (w_snr ** beta_snr)
        
        weights = w_comb.reshape(-1, 1)
        
        mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        
    elif args.loss_weight_strategy == "piecewise_cosine":
        if noise_scheduler is None:
            raise ValueError("noise_scheduler is required for piecewise_cosine weight strategy")
        
        total_steps = noise_scheduler.config.num_train_timesteps
        T = float(total_steps)
        
        t1 = getattr(args, 'piecewise_cosine_t1', 700.0)
        t1 = float(t1)
        
        timesteps_float = timesteps.float()
        
        weights = torch.where(
            timesteps_float <= t1,
            torch.ones_like(timesteps_float),
            (1.0 + torch.cos(torch.pi * (timesteps_float - t1) / (T - t1))) / 2.0
        )
        
        weights = weights.reshape(-1, 1)
        
        mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        
    else:
        raise ValueError(f"Unknown loss_weight_strategy: {args.loss_weight_strategy}")
    
    return weights, mask






def log_validation(
        vae,
        text_encoder,
        tokenizer,
        unet,
        controlnet,
        ema_controlnet,
        args,
        accelerator,
        weight_dtype,
        step,
        val_dataset):

    (custom_model, custom_processor, custom_tokenizer,
     baseline_model, baseline_processor, baseline_tokenizer,
     medclip_model, medclip_processor, medclip_tokenizer) = initialize_clip_models(device=accelerator.device, args=args)
    
    if args.clip_model_for_metric == "custom":
        if custom_model is None:
            raise ValueError("Custom CLIP model is required for metrics but not loaded. Please set --custom_clip_model_path and ensure --clip_model_for_metric is 'custom'.")
        metric_model = custom_model
        metric_processor = custom_processor
        metric_tokenizer = custom_tokenizer
        is_medclip_metric = False
    elif args.clip_model_for_metric == "medclip":
        if medclip_model is None:
            raise ValueError("MedCLIP model is required for metrics but not loaded. Please ensure --clip_model_for_metric is 'medclip'.")
        metric_model = medclip_model
        metric_processor = medclip_processor
        metric_tokenizer = medclip_tokenizer
        is_medclip_metric = True
    else:  # baseline
        if baseline_model is None:
            raise ValueError("Baseline CLIP model is required for metrics but not loaded. Please ensure --clip_model_for_metric is 'baseline'.")
        metric_model = baseline_model
        metric_processor = baseline_processor
        metric_tokenizer = baseline_tokenizer
        is_medclip_metric = False

    # randomly select some samples to log
    if val_dataset is not None:
        if isinstance(val_dataset, dict):
            val_dataset = val_dataset['validation']
        else:
            val_dataset = val_dataset.select(
                random.sample(range(len(val_dataset)), args.max_val_samples)
            )

    if args.task_name in ['lineart', 'hed', 'segmentation']:
        reward_model = get_reward_model(args.task_name, args.reward_model_name_or_path, device=accelerator.device)
        reward_model.to(accelerator.device)
        reward_model.eval()
    else:
        reward_model = None

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    torch.cuda.empty_cache()
    pipeline.enable_vae_slicing()

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_column = args.image_column
    caption_column = args.caption_column

    if args.conditioning_image_column in ['canny', 'lineart', 'hed']:
        conditioning_image_column = image_column
    else:
        conditioning_image_column = args.conditioning_image_column

    assert val_dataset is not None, "Validation dataset is required for logging validation images."
    
    # Check if it's the FairSeg dataset (returns dict with specific keys)
    if 'fairseg' in args.dataset_name.lower():
        sample = val_dataset[0]
        # FairSeg dataset - extract data from dictionary structure
        logger.info("Detected FairSeg dataset format, using custom data extraction logic")
        validation_images = []
        validation_conditions = []
        validation_prompts = []
        validation_labels = []

        max_samples = len(val_dataset)
        logger.info(f"Processing {max_samples} validation samples from FairSeg dataset")

        for item in range(max_samples):
            sample = val_dataset[item]

            # Get labels for uncertainty calculation
            if 'labels' in sample:
                label = sample['labels']
                if isinstance(label, torch.Tensor):
                    label = label.to(memory_format=torch.contiguous_format).float()
                else:
                    label = torch.tensor(label, dtype=torch.float32)
                validation_labels.append(label)

            # Convert pixel_values (target image) from tensor to PIL Image
            pixel_values = sample['pixel_values']  # Shape: [3, H, W], range [-1, 1]
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values.cpu().numpy()

            # Denormalize from [-1, 1] to [0, 255] and convert to PIL
            pixel_values = ((pixel_values + 1.0) * 127.5).astype(np.uint8)
            pixel_values = np.transpose(pixel_values, (1, 2, 0))  # [H, W, 3]
            validation_image = Image.fromarray(pixel_values)
            validation_images.append(validation_image)

            # Convert conditioning_pixel_values (conditioning image) from tensor to PIL Image
            conditioning_pixel_values = sample['conditioning_pixel_values']  # Shape: [3, H, W], range [0, 1]
            if isinstance(conditioning_pixel_values, torch.Tensor):
                conditioning_pixel_values = conditioning_pixel_values.cpu().numpy()

            # Denormalize from [0, 1] to [0, 255] and convert to PIL
            conditioning_pixel_values = (conditioning_pixel_values * 255.0).astype(np.uint8)
            conditioning_pixel_values = np.transpose(conditioning_pixel_values, (1, 2, 0))  # [H, W, 3]
            validation_condition = Image.fromarray(conditioning_pixel_values)
            validation_conditions.append(validation_condition)

            # Get prompt
            validation_prompts.append(sample['prompt'])
    else:
        # Fallback to original logic
        logger.info("Using fallback dataset format")
        validation_images = [val_dataset[item][image_column] for item in range(len(val_dataset))]
        validation_conditions = [val_dataset[item][conditioning_image_column] for item in range(len(val_dataset))]
        validation_prompts = [val_dataset[item][caption_column] for item in range(len(val_dataset))]
        validation_labels = []

    # Avoid some problems caused by grayscale images
    validation_conditions = [x.convert('RGB') for x in validation_conditions]

    if args.conditioning_image_column == "canny":
        low_threshold = 0.1 # low_threshold = random.uniform(0, 1)
        high_threshold = 0.2 # high_threshold = random.uniform(low_threshold, 1)
        with autocast():
            validation_conditions = [torchvision.transforms.functional.pil_to_tensor(x) for x in validation_conditions]
            validation_conditions = [x / 255. for x in validation_conditions]
            validation_conditions = kornia.filters.canny(torch.stack(validation_conditions), low_threshold, high_threshold)[1]
            validation_conditions = torch.chunk(validation_conditions, len(validation_conditions), dim=0)
            validation_conditions = [torchvision.transforms.functional.to_pil_image(x.squeeze(0), 'L') for x in validation_conditions]
    elif args.conditioning_image_column in ['lineart', 'hed']:
        with autocast():
            validation_conditions = [torchvision.transforms.functional.pil_to_tensor(x) for x in validation_conditions]
            validation_conditions = [x / 255. for x in validation_conditions]
            validation_conditions = [torchvision.transforms.functional.resize(x, (512, 512), interpolation=Image.BICUBIC) for x in validation_conditions]
            with torch.no_grad():
                validation_conditions = reward_model(torch.stack(validation_conditions).to(accelerator.device))
            validation_conditions = 1 - validation_conditions if args.task_name == 'lineart' else validation_conditions
            validation_conditions = torch.chunk(validation_conditions, len(validation_conditions), dim=0)
            validation_conditions = [torchvision.transforms.functional.to_pil_image(x.squeeze(0), 'L') for x in validation_conditions]


    image_logs = []

    fid_gen_accum_nonema   = []
    fid_real_accum_nonema  = []
    masked_fid_gen_accum_nonema  = []
    masked_fid_real_accum_nonema = []
    fid_gen_accum_ema      = []
    fid_real_accum_ema     = []
    # EMA / masked
    masked_fid_gen_accum_ema  = []
    masked_fid_real_accum_ema = []

    gen_vcdr_abs_errors_nonema = []
    gen_vcdr_gt_list_nonema    = []
    gen_vcdr_pred_list_nonema  = []
    cup_inside_disc_flags_nonema = []  # bool per generated image
    gen_vcdr_abs_errors_ema = []
    gen_vcdr_gt_list_ema    = []
    gen_vcdr_pred_list_ema  = []
    cup_inside_disc_flags_ema = []

    def _bbox_vcdr(labels_np):
        disc_rows_mask = (labels_np == 1) | (labels_np == 2)   # rim or cup
        cup_rows_mask  = (labels_np == 2)
        if not disc_rows_mask.any() or not cup_rows_mask.any():
            return 0.0, False
        disc_rows = np.where(disc_rows_mask.any(axis=1))[0]
        cup_rows  = np.where(cup_rows_mask.any(axis=1))[0]
        if len(disc_rows) < 2 or len(cup_rows) < 2:
            return 0.0, False
        disc_vd = float(disc_rows[-1] - disc_rows[0])
        cup_vd  = float(cup_rows[-1]  - cup_rows[0])
        if disc_vd <= 0:
            return 0.0, False
        valid = (cup_rows[0]  >= disc_rows[0]  - 1) and \
                (cup_rows[-1] <= disc_rows[-1] + 1)
        return cup_vd / disc_vd, valid

    def _compute_vcdrs_from_predictions(preds_tensor, gt_mask_tensor):
        # pred masks: argmax over class → (B, H, W) int64
        with torch.no_grad():
            pred_labels = preds_tensor.argmax(dim=1).to(torch.uint8).cpu().numpy()
        gt_labels = gt_mask_tensor.detach().cpu().numpy()
        if gt_labels.ndim == 3 and gt_labels.shape[0] == 1:
            gt_labels = gt_labels[0]
        elif gt_labels.ndim == 3:
            gt_labels = gt_labels[0]
        gt_labels = gt_labels.astype(np.int32)

        gt_vcdr, _ = _bbox_vcdr(gt_labels)

        pred_vcdrs = []
        cup_valids = []
        for b in range(pred_labels.shape[0]):
            vcdr, valid = _bbox_vcdr(pred_labels[b].astype(np.int32))
            pred_vcdrs.append(float(vcdr))
            cup_valids.append(bool(valid))
        return float(gt_vcdr), pred_vcdrs, cup_valids

    logger.info(f"Running validation with {len(validation_prompts)} prompts... ")
    if validation_labels:
        validation_iter = zip(validation_prompts, validation_conditions, validation_images, validation_labels)
    else:
        validation_iter = zip(validation_prompts, validation_conditions, validation_images)
    
    for idx, item in enumerate(validation_iter):
        if validation_labels:
            validation_prompt, validation_condition, validation_image, validation_label = item
        else:
            validation_prompt, validation_condition, validation_image = item
            validation_label = None
            
        if val_dataset is not None:
            validation_image = validation_image.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
            validation_condition = validation_condition.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
        else:
            validation_condition = Image.open(validation_condition).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)

        with torch.autocast("cuda"):
            images = pipeline(
                [validation_prompt] * args.num_validation_images,
                [validation_condition] * args.num_validation_images,
                num_inference_steps=35,
                generator=generator
            ).images

        metrics = calculate_pairwise_metrics(images, validation_image, device=accelerator.device)
        ssim_score_value = metrics['ssim']
        ms_ssim_score_value = metrics['ms_ssim']
        lpips_score_value = metrics['lpips_alex']
        psnr_score_value = metrics['psnr']
        fid_gen_accum_nonema.extend(images)
        fid_real_accum_nonema.append(validation_image)
        fid_score_value = -1

        logger.info(f"SSIM score for validation: {ssim_score_value}")
        logger.info(f"MS-SSIM score for validation: {ms_ssim_score_value}")
        logger.info(f"LPIPS score for validation: {lpips_score_value}")
        logger.info(f"PSNR score for validation: {psnr_score_value}")

        clip_scores = []
        for generated_image in images:
            if is_medclip_metric:
                with torch.no_grad():
                    inputs = metric_processor(
                        images=[generated_image, validation_image],
                        return_tensors="pt",
                        padding=True
                    )
                    for key in inputs:
                        if isinstance(inputs[key], torch.Tensor):
                            inputs[key] = inputs[key].to(accelerator.device)
                    
                    outputs = metric_model(**inputs)
                    img_embeds = outputs['img_embeds']
                    
                    img_embeds = F.normalize(img_embeds, dim=1)
                    
                    clip_score = (img_embeds[0] * img_embeds[1]).sum().cpu().item()
                    clip_score = 1.0 * np.clip(clip_score, 0, None)  # w=1
            else:
                clip_score = calculate_single_image_clip_score(generated_image, validation_image, 
                                                              clip_model=metric_model, 
                                                              clip_processor=metric_processor, 
                                                              clip_tokenizer=metric_tokenizer, w=1)
            clip_scores.append(clip_score)

        
        valid_clip_scores = [score for score in clip_scores if score != -1.0]
        avg_clip_score = np.mean(valid_clip_scores) if valid_clip_scores else -1.0
        logger.info(f"CLIP score for validation: {avg_clip_score}")
        
        masked_metrics_union = None
        masked_metrics_intersection = None
        if reward_model is not None and validation_label is not None and args.task_name == 'segmentation':
            try:
                images_tensor = torch.stack([
                    transforms.ToTensor()(img) for img in images
                ]).to(accelerator.device)
                images_normalized = normalize(images_tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                
                with torch.no_grad():
                    if 'segman::' in args.reward_model_name_or_path:
                        reward_model.to(accelerator.device, dtype=torch.float32)

                        predictions = reward_model(images_normalized)
                    elif args.reward_model_name_or_path.endswith('.pth') or args.reward_model_name_or_path.endswith('.pt'):
                        from torchvision.transforms.functional import rgb_to_grayscale
                        transunet_inputs = rgb_to_grayscale(images_normalized)
                        transunet_inputs = torchvision.transforms.functional.resize(transunet_inputs, (224, 224))
                        predictions = reward_model(transunet_inputs.float())
                        predictions = torch.nn.functional.interpolate(
                            predictions, size=(512, 512), mode='bilinear', align_corners=False
                        )
                    else:
                        predictions = reward_model(images_normalized)
                
                if validation_label.ndim == 2:
                    label_tensor = validation_label
                elif validation_label.ndim == 3 and validation_label.shape[0] == 1:
                    label_tensor = validation_label
                else:
                    label_tensor = validation_label
                
                # Resize label to 512x512
                if label_tensor.shape[-2:] != (512, 512):
                    label_tensor_resized = torch.nn.functional.interpolate(
                        label_tensor.unsqueeze(0).unsqueeze(0).float() if label_tensor.ndim == 2 else label_tensor.unsqueeze(0).float(),
                        size=(512, 512), mode='nearest'
                    ).squeeze(0)
                    if label_tensor_resized.ndim == 3 and label_tensor_resized.shape[0] == 1:
                        label_tensor_resized = label_tensor_resized.squeeze(0)
                else:
                    label_tensor_resized = label_tensor
                
                mask_union = create_foreground_mask(label_tensor_resized, predictions, mode='union')

                masked_images_union = apply_mask_to_images(images, mask_union)
                masked_validation_image_union = apply_mask_to_images([validation_image], mask_union)


                masked_metrics_union = calculate_pairwise_metrics(
                    masked_images_union, masked_validation_image_union[0],
                    device=accelerator.device)
                masked_metrics_union['fid'] = -1
                masked_fid_gen_accum_nonema.extend(masked_images_union)
                masked_fid_real_accum_nonema.append(masked_validation_image_union[0])
                logger.info(
                    f"Masked Pairwise - SSIM: {masked_metrics_union['ssim']:.4f}, "
                    f"MS-SSIM: {masked_metrics_union['ms_ssim']:.4f}, "
                    f"LPIPS: {masked_metrics_union['lpips_alex']:.4f}, "
                    f"PSNR: {masked_metrics_union['psnr']:.4f}"
                )

                try:
                    gt_vcdr, pred_vcdrs, cup_valids = _compute_vcdrs_from_predictions(
                        predictions, label_tensor_resized
                    )
                    if gt_vcdr > 0:
                        for r_gen, valid in zip(pred_vcdrs, cup_valids):
                            gen_vcdr_gt_list_nonema.append(gt_vcdr)
                            gen_vcdr_pred_list_nonema.append(r_gen)
                            gen_vcdr_abs_errors_nonema.append(abs(r_gen - gt_vcdr))
                            cup_inside_disc_flags_nonema.append(valid)
                        sample_mae = np.mean([abs(r - gt_vcdr) for r in pred_vcdrs])
                        logger.info(
                            f"Generated vCDR - GT={gt_vcdr:.4f}, "
                            f"preds={[round(r,4) for r in pred_vcdrs]}, "
                            f"sample_MAE={sample_mae:.4f}, "
                            f"cup_valid={sum(cup_valids)}/{len(cup_valids)}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to compute generated vCDR: {e}")

            except Exception as e:
                logger.warning(f"Failed to calculate masked metrics: {e}")
                import traceback
                traceback.print_exc()
                masked_metrics_union = None

        image_logs.append({
            "validation_image": validation_image,
            "validation_condition": validation_condition,
            "validation_prompt": validation_prompt,
            "images": images,
            'ema_images': [],
            'fid_score': fid_score_value,
            'ssim_score': ssim_score_value,
            'ms_ssim_score': ms_ssim_score_value,
            'lpips_score': lpips_score_value,
            'psnr_score': psnr_score_value,
            'clip_scores': clip_scores,
            'avg_clip_score': avg_clip_score,
            'masked_metrics_union': masked_metrics_union,
        })

    if args.use_ema:
        # Store the ControlNet parameters temporarily and load the EMA parameters to perform inference.
        ema_controlnet.copy_to(controlnet.parameters())

        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            safety_checker=None,
            revision=args.revision,
            torch_dtype=weight_dtype,
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        if args.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        logger.info(f"Running validation with {len(validation_prompts)} prompts... ")
        if validation_labels:
            ema_validation_iter = zip(validation_prompts, validation_conditions, validation_images, validation_labels)
        else:
            ema_validation_iter = zip(validation_prompts, validation_conditions, validation_images)
        
        for idx, item in enumerate(ema_validation_iter):
            if validation_labels:
                validation_prompt, validation_condition, validation_image, validation_label = item
            else:
                validation_prompt, validation_condition, validation_image = item
                validation_label = None
                
            if val_dataset is not None:
                validation_condition = validation_condition.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
            else:
                validation_condition = Image.open(validation_condition).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)

            with torch.autocast("cuda"):
                images = pipeline(
                    [validation_prompt] * args.num_validation_images,
                    [validation_condition] * args.num_validation_images,
                    num_inference_steps=35,
                    generator=generator
                ).images

            ema_metrics = calculate_pairwise_metrics(images, validation_image, device=accelerator.device)
            ema_ssim_score_value = ema_metrics['ssim']
            ema_ms_ssim_score_value = ema_metrics['ms_ssim']
            ema_lpips_score_value = ema_metrics['lpips_alex']
            ema_psnr_score_value = ema_metrics['psnr']
            fid_gen_accum_ema.extend(images)
            fid_real_accum_ema.append(validation_image)
            ema_fid_score_value = -1

            ema_clip_scores = []
            for generated_image in images:
                if is_medclip_metric:
                    with torch.no_grad():
                        inputs = metric_processor(
                            images=[generated_image, validation_image],
                            return_tensors="pt",
                            padding=True
                        )
                        for key in inputs:
                            if isinstance(inputs[key], torch.Tensor):
                                inputs[key] = inputs[key].to(accelerator.device)
                        
                        outputs = metric_model(**inputs)
                        img_embeds = outputs['img_embeds']
                        
                        img_embeds = F.normalize(img_embeds, dim=1)
                        
                        ema_clip_score = (img_embeds[0] * img_embeds[1]).sum().cpu().item()
                        ema_clip_score = 1.0 * np.clip(ema_clip_score, 0, None)  # w=1
                else:
                    ema_clip_score = calculate_single_image_clip_score(generated_image, validation_image,
                                                                      clip_model=metric_model, 
                                                                      clip_processor=metric_processor, 
                                                                      clip_tokenizer=metric_tokenizer, w=1)
                ema_clip_scores.append(ema_clip_score)

            
            valid_ema_clip_scores = [score for score in ema_clip_scores if score != -1.0]
            avg_ema_clip_score = np.mean(valid_ema_clip_scores) if valid_ema_clip_scores else -1.0
            logger.info(f"EMA CLIP score for validation: {avg_ema_clip_score}")
            
            ema_masked_metrics_union = None
            ema_masked_metrics_intersection = None
            if reward_model is not None and validation_label is not None and args.task_name == 'segmentation':
                try:
                    ema_images_tensor = torch.stack([
                        transforms.ToTensor()(img) for img in images
                    ]).to(accelerator.device)
                    ema_images_normalized = normalize(ema_images_tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

                    with torch.no_grad():
                        if 'segman::' in args.reward_model_name_or_path:
                            reward_model.to(accelerator.device, dtype=torch.float32)
                            ema_predictions = reward_model(ema_images_normalized)
                        elif args.reward_model_name_or_path.endswith('.pth') or args.reward_model_name_or_path.endswith('.pt'):
                            from torchvision.transforms.functional import rgb_to_grayscale
                            ema_transunet_inputs = rgb_to_grayscale(ema_images_normalized)
                            ema_transunet_inputs = torchvision.transforms.functional.resize(ema_transunet_inputs, (224, 224))
                            ema_predictions = reward_model(ema_transunet_inputs.float())
                            ema_predictions = torch.nn.functional.interpolate(
                                ema_predictions, size=(512, 512), mode='bilinear', align_corners=False
                            )
                        else:
                            ema_predictions = reward_model(ema_images_normalized)
                    
                    if validation_label.ndim == 2:
                        label_tensor = validation_label
                    elif validation_label.ndim == 3 and validation_label.shape[0] == 1:
                        label_tensor = validation_label
                    else:
                        label_tensor = validation_label
                    
                    # Resize label to 512x512
                    if label_tensor.shape[-2:] != (512, 512):
                        label_tensor_resized = torch.nn.functional.interpolate(
                            label_tensor.unsqueeze(0).unsqueeze(0).float() if label_tensor.ndim == 2 else label_tensor.unsqueeze(0).float(),
                            size=(512, 512), mode='nearest'
                        ).squeeze(0)
                        if label_tensor_resized.ndim == 3 and label_tensor_resized.shape[0] == 1:
                            label_tensor_resized = label_tensor_resized.squeeze(0)
                    else:
                        label_tensor_resized = label_tensor
                    
                    ema_mask_union = create_foreground_mask(label_tensor_resized, ema_predictions, mode='union')

                    ema_masked_images_union = apply_mask_to_images(images, ema_mask_union)
                    ema_masked_validation_image_union = apply_mask_to_images([validation_image], ema_mask_union)
                    

                    
                    ema_masked_metrics_union = calculate_pairwise_metrics(
                        ema_masked_images_union, ema_masked_validation_image_union[0],
                        device=accelerator.device)
                    ema_masked_metrics_union['fid'] = -1
                    masked_fid_gen_accum_ema.extend(ema_masked_images_union)
                    masked_fid_real_accum_ema.append(ema_masked_validation_image_union[0])
                    logger.info(
                        f"EMA Masked Pairwise - SSIM: {ema_masked_metrics_union['ssim']:.4f}, "
                        f"MS-SSIM: {ema_masked_metrics_union['ms_ssim']:.4f}, "
                        f"LPIPS: {ema_masked_metrics_union['lpips_alex']:.4f}, "
                        f"PSNR: {ema_masked_metrics_union['psnr']:.4f}"
                    )

                    # --- EMA Generated vCDR ---
                    try:
                        gt_vcdr, pred_vcdrs, cup_valids = _compute_vcdrs_from_predictions(
                            ema_predictions, label_tensor_resized
                        )
                        if gt_vcdr > 0:
                            for r_gen, valid in zip(pred_vcdrs, cup_valids):
                                gen_vcdr_gt_list_ema.append(gt_vcdr)
                                gen_vcdr_pred_list_ema.append(r_gen)
                                gen_vcdr_abs_errors_ema.append(abs(r_gen - gt_vcdr))
                                cup_inside_disc_flags_ema.append(valid)
                            sample_mae = np.mean([abs(r - gt_vcdr) for r in pred_vcdrs])
                            logger.info(
                                f"EMA Generated vCDR - GT={gt_vcdr:.4f}, "
                                f"sample_MAE={sample_mae:.4f}, "
                                f"cup_valid={sum(cup_valids)}/{len(cup_valids)}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to compute EMA generated vCDR: {e}")

                except Exception as e:
                    logger.warning(f"Failed to calculate EMA masked metrics: {e}")
                    import traceback
                    traceback.print_exc()
                    ema_masked_metrics_union = None

            image_logs[idx]['ema_images'] = images
            image_logs[idx]['ema_fid_score'] = ema_fid_score_value
            image_logs[idx]['ema_ssim_score'] = ema_ssim_score_value
            image_logs[idx]['ema_ms_ssim_score'] = ema_ms_ssim_score_value
            image_logs[idx]['ema_lpips_score'] = ema_lpips_score_value
            image_logs[idx]['ema_psnr_score'] = ema_psnr_score_value
            image_logs[idx]['ema_clip_scores'] = ema_clip_scores
            image_logs[idx]['avg_ema_clip_score'] = avg_ema_clip_score
            image_logs[idx]['ema_masked_metrics_union'] = ema_masked_metrics_union

        


    def _aggregate_fid(gens, reals, tag):
        if len(gens) >= 2 and len(reals) >= 2:
            val = calculate_fid_distribution(
                gens, reals, device=accelerator.device, batch_size=16)
            logger.info(f"[FID/{tag}] aggregated over {len(gens)} gen vs {len(reals)} real = {val:.4f}")
            return val
        logger.warning(f"[FID/{tag}] not enough samples (gen={len(gens)}, real={len(reals)}); skip")
        return -1

    avg_fid     = _aggregate_fid(fid_gen_accum_nonema, fid_real_accum_nonema, 'nonema')
    avg_ema_fid = _aggregate_fid(fid_gen_accum_ema,    fid_real_accum_ema,    'ema') \
                  if len(fid_gen_accum_ema) > 0 else -1

    for _log in image_logs:
        _log['fid_score']     = avg_fid
        _log['ema_fid_score'] = avg_ema_fid
    
    ssim_scores = [log.get('ssim_score', -1) for log in image_logs if log.get('ssim_score', -1) != -1]
    ema_ssim_scores = [log.get('ema_ssim_score', -1) for log in image_logs if log.get('ema_ssim_score', -1) != -1]
    
    avg_ssim = np.mean(ssim_scores) if ssim_scores else -1
    avg_ema_ssim = np.mean(ema_ssim_scores) if ema_ssim_scores else -1
    
    ms_ssim_scores = [log.get('ms_ssim_score', -1) for log in image_logs if log.get('ms_ssim_score', -1) != -1]
    ema_ms_ssim_scores = [log.get('ema_ms_ssim_score', -1) for log in image_logs if log.get('ema_ms_ssim_score', -1) != -1]
    
    avg_ms_ssim = np.mean(ms_ssim_scores) if ms_ssim_scores else -1
    avg_ema_ms_ssim = np.mean(ema_ms_ssim_scores) if ema_ms_ssim_scores else -1
    
    lpips_scores = [log.get('lpips_score', -1) for log in image_logs if log.get('lpips_score', -1) != -1]
    ema_lpips_scores = [log.get('ema_lpips_score', -1) for log in image_logs if log.get('ema_lpips_score', -1) != -1]
    
    avg_lpips = np.mean(lpips_scores) if lpips_scores else -1
    avg_ema_lpips = np.mean(ema_lpips_scores) if ema_lpips_scores else -1
    
    psnr_scores = [log.get('psnr_score', -1) for log in image_logs if log.get('psnr_score', -1) != -1]
    ema_psnr_scores = [log.get('ema_psnr_score', -1) for log in image_logs if log.get('ema_psnr_score', -1) != -1]
    
    avg_psnr = np.mean(psnr_scores) if psnr_scores else -1
    avg_ema_psnr = np.mean(ema_psnr_scores) if ema_psnr_scores else -1
    
    clip_scores = [log.get('avg_clip_score', -1) for log in image_logs if log.get('avg_clip_score', -1) != -1]
    ema_clip_scores = [log.get('avg_ema_clip_score', -1) for log in image_logs if log.get('avg_ema_clip_score', -1) != -1]
    
    avg_clip = np.mean(clip_scores) if clip_scores else -1
    avg_ema_clip = np.mean(ema_clip_scores) if ema_clip_scores else -1
    
    masked_fid_union = [log['masked_metrics_union']['fid'] for log in image_logs if log.get('masked_metrics_union') is not None]
    masked_ssim_union = [log['masked_metrics_union']['ssim'] for log in image_logs if log.get('masked_metrics_union') is not None]
    masked_ms_ssim_union = [log['masked_metrics_union']['ms_ssim'] for log in image_logs if log.get('masked_metrics_union') is not None]
    masked_lpips_union = [log['masked_metrics_union']['lpips_alex'] for log in image_logs if log.get('masked_metrics_union') is not None]
    masked_psnr_union = [log['masked_metrics_union']['psnr'] for log in image_logs if log.get('masked_metrics_union') is not None]
    
    ema_masked_fid_union = [log['ema_masked_metrics_union']['fid'] for log in image_logs if log.get('ema_masked_metrics_union') is not None]
    ema_masked_ssim_union = [log['ema_masked_metrics_union']['ssim'] for log in image_logs if log.get('ema_masked_metrics_union') is not None]
    ema_masked_ms_ssim_union = [log['ema_masked_metrics_union']['ms_ssim'] for log in image_logs if log.get('ema_masked_metrics_union') is not None]
    ema_masked_lpips_union = [log['ema_masked_metrics_union']['lpips_alex'] for log in image_logs if log.get('ema_masked_metrics_union') is not None]
    ema_masked_psnr_union = [log['ema_masked_metrics_union']['psnr'] for log in image_logs if log.get('ema_masked_metrics_union') is not None]
    
    _masked_ssim_union     = [v for v in masked_ssim_union     if v != -1]
    _masked_ms_ssim_union  = [v for v in masked_ms_ssim_union  if v != -1]
    _masked_lpips_union    = [v for v in masked_lpips_union    if v != -1]
    _masked_psnr_union     = [v for v in masked_psnr_union     if v != -1]
    avg_masked_ssim_union    = np.mean(_masked_ssim_union)    if _masked_ssim_union    else -1
    avg_masked_ms_ssim_union = np.mean(_masked_ms_ssim_union) if _masked_ms_ssim_union else -1
    avg_masked_lpips_union   = np.mean(_masked_lpips_union)   if _masked_lpips_union   else -1
    avg_masked_psnr_union    = np.mean(_masked_psnr_union)    if _masked_psnr_union    else -1
    avg_masked_fid_union = _aggregate_fid(
        masked_fid_gen_accum_nonema, masked_fid_real_accum_nonema, 'masked/nonema'
    ) if len(masked_fid_gen_accum_nonema) > 0 else -1

    _ema_masked_ssim_union    = [v for v in ema_masked_ssim_union    if v != -1]
    _ema_masked_ms_ssim_union = [v for v in ema_masked_ms_ssim_union if v != -1]
    _ema_masked_lpips_union   = [v for v in ema_masked_lpips_union   if v != -1]
    _ema_masked_psnr_union    = [v for v in ema_masked_psnr_union    if v != -1]
    avg_ema_masked_ssim_union    = np.mean(_ema_masked_ssim_union)    if _ema_masked_ssim_union    else -1
    avg_ema_masked_ms_ssim_union = np.mean(_ema_masked_ms_ssim_union) if _ema_masked_ms_ssim_union else -1
    avg_ema_masked_lpips_union   = np.mean(_ema_masked_lpips_union)   if _ema_masked_lpips_union   else -1
    avg_ema_masked_psnr_union    = np.mean(_ema_masked_psnr_union)    if _ema_masked_psnr_union    else -1
    avg_ema_masked_fid_union = _aggregate_fid(
        masked_fid_gen_accum_ema, masked_fid_real_accum_ema, 'masked/ema'
    ) if len(masked_fid_gen_accum_ema) > 0 else -1
    
    log_dict = {
        "validation_fid": avg_fid,
        "validation_ema_fid": avg_ema_fid,
        "validation_ssim": avg_ssim,
        "validation_ema_ssim": avg_ema_ssim,
        "validation_ms_ssim": avg_ms_ssim,
        "validation_ema_ms_ssim": avg_ema_ms_ssim,
        "validation_lpips": avg_lpips,
        "validation_ema_lpips": avg_ema_lpips,
        "validation_psnr": avg_psnr,
        "validation_ema_psnr": avg_ema_psnr,
        "validation_clip": avg_clip,
        "validation_ema_clip": avg_ema_clip
    }
    
    if avg_masked_fid_union != -1:
        log_dict.update({
            "validation_masked_fid": avg_masked_fid_union,
            "validation_masked_ssim": avg_masked_ssim_union,
            "validation_masked_ms_ssim": avg_masked_ms_ssim_union,
            "validation_masked_lpips": avg_masked_lpips_union,
            "validation_masked_psnr": avg_masked_psnr_union,
            "validation_ema_masked_fid": avg_ema_masked_fid_union,
            "validation_ema_masked_ssim": avg_ema_masked_ssim_union,
            "validation_ema_masked_ms_ssim": avg_ema_masked_ms_ssim_union,
            "validation_ema_masked_lpips": avg_ema_masked_lpips_union,
            "validation_ema_masked_psnr": avg_ema_masked_psnr_union
        })
    
    accelerator.log(log_dict, step=step)
    
    logger.info(f"Average FID score: {avg_fid}, Average EMA FID score: {avg_ema_fid}")
    logger.info(f"Average SSIM score: {avg_ssim}, Average EMA SSIM score: {avg_ema_ssim}")
    logger.info(f"Average MS-SSIM score: {avg_ms_ssim}, Average EMA MS-SSIM score: {avg_ema_ms_ssim}")
    logger.info(f"Average LPIPS score: {avg_lpips}, Average EMA LPIPS score: {avg_ema_lpips}")
    logger.info(f"Average PSNR score: {avg_psnr}, Average EMA PSNR score: {avg_ema_psnr}")
    logger.info(f"Average CLIP score: {avg_clip}, Average EMA CLIP score: {avg_ema_clip}")
    
    if avg_masked_fid_union != -1:
        logger.info(f"Average Masked Metrics - FID: {avg_masked_fid_union:.4f}, SSIM: {avg_masked_ssim_union:.4f}, MS-SSIM: {avg_masked_ms_ssim_union:.4f}, LPIPS: {avg_masked_lpips_union:.4f}, PSNR: {avg_masked_psnr_union:.4f}")
        logger.info(f"Average EMA Masked Metrics - FID: {avg_ema_masked_fid_union:.4f}, SSIM: {avg_ema_masked_ssim_union:.4f}, MS-SSIM: {avg_ema_masked_ms_ssim_union:.4f}, LPIPS: {avg_ema_masked_lpips_union:.4f}, PSNR: {avg_ema_masked_psnr_union:.4f}")

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                ema_images = log["ema_images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                validation_condition = log["validation_condition"]
                fid_score = log.get('fid_score', -1)
                ema_fid_score = log.get('ema_fid_score', -1)
                ssim_score = log.get('ssim_score', -1)
                ema_ssim_score = log.get('ema_ssim_score', -1)
                ms_ssim_score = log.get('ms_ssim_score', -1)
                ema_ms_ssim_score = log.get('ema_ms_ssim_score', -1)
                lpips_score = log.get('lpips_score', -1)
                ema_lpips_score = log.get('ema_lpips_score', -1)
                psnr_score = log.get('psnr_score', -1)
                ema_psnr_score = log.get('ema_psnr_score', -1)
                clip_score = log.get('avg_clip_score', -1)
                ema_clip_score = log.get('avg_ema_clip_score', -1)

                validation_prompt = validation_prompt + ['EMA'] * len(validation_prompt)

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                for image in ema_images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
                
                if fid_score != -1:
                    tracker.writer.add_scalar("fid_score", fid_score, step)
                if ema_fid_score != -1:
                    tracker.writer.add_scalar("ema_fid_score", ema_fid_score, step)
                
                if ssim_score != -1:
                    tracker.writer.add_scalar("ssim_score", ssim_score, step)
                if ema_ssim_score != -1:
                    tracker.writer.add_scalar("ema_ssim_score", ema_ssim_score, step)
                
                if ms_ssim_score != -1:
                    tracker.writer.add_scalar("ms_ssim_score", ms_ssim_score, step)
                if ema_ms_ssim_score != -1:
                    tracker.writer.add_scalar("ema_ms_ssim_score", ema_ms_ssim_score, step)
                
                if lpips_score != -1:
                    tracker.writer.add_scalar("lpips_score", lpips_score, step)
                if ema_lpips_score != -1:
                    tracker.writer.add_scalar("ema_lpips_score", ema_lpips_score, step)
                
                if psnr_score != -1:
                    tracker.writer.add_scalar("psnr_score", psnr_score, step)
                if ema_psnr_score != -1:
                    tracker.writer.add_scalar("ema_psnr_score", ema_psnr_score, step)
                
                if clip_score != -1:
                    tracker.writer.add_scalar("clip_score", clip_score, step)
                if ema_clip_score != -1:
                    tracker.writer.add_scalar("ema_clip_score", ema_clip_score, step)
        elif tracker.name == "wandb":

            formatted_images = []
            for log in image_logs:
                images = log["images"]
                ema_images = log["ema_images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                validation_condition = log["validation_condition"]
                fid_score = log.get('fid_score', -1)
                ema_fid_score = log.get('ema_fid_score', -1)
                ssim_score = log.get('ssim_score', -1)
                ema_ssim_score = log.get('ema_ssim_score', -1)
                ms_ssim_score = log.get('ms_ssim_score', -1)
                ema_ms_ssim_score = log.get('ema_ms_ssim_score', -1)
                lpips_score = log.get('lpips_score', -1)
                ema_lpips_score = log.get('ema_lpips_score', -1)
                psnr_score = log.get('psnr_score', -1)
                ema_psnr_score = log.get('ema_psnr_score', -1)
                clip_score = log.get('avg_clip_score', -1)
                ema_clip_score = log.get('avg_ema_clip_score', -1)

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet input image"))
                formatted_images.append(wandb.Image(validation_condition, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

                for image in ema_images:
                    image = wandb.Image(image, caption='EMA')
                    formatted_images.append(image)

            # wandb_log = {"validation": formatted_images}
            wandb_log = {}
            if avg_fid != -1:
                wandb_log["validation_fid"] = avg_fid
            if avg_ema_fid != -1:
                wandb_log["validation_ema_fid"] = avg_ema_fid
            if avg_ssim != -1:
                wandb_log["validation_ssim"] = avg_ssim
            if avg_ema_ssim != -1:
                wandb_log["validation_ema_ssim"] = avg_ema_ssim
            if avg_ms_ssim != -1:
                wandb_log["validation_ms_ssim"] = avg_ms_ssim
            if avg_ema_ms_ssim != -1:
                wandb_log["validation_ema_ms_ssim"] = avg_ema_ms_ssim
            if avg_lpips != -1:
                wandb_log["validation_lpips"] = avg_lpips
            if avg_ema_lpips != -1:
                wandb_log["validation_ema_lpips"] = avg_ema_lpips
            if avg_psnr != -1:
                wandb_log["validation_psnr"] = avg_psnr
            if avg_ema_psnr != -1:
                wandb_log["validation_ema_psnr"] = avg_ema_psnr
            if avg_clip != -1:
                wandb_log["validation_clip"] = avg_clip
            if avg_ema_clip != -1:
                wandb_log["validation_ema_clip"] = avg_ema_clip
            
            if avg_masked_fid_union != -1:
                wandb_log["validation_masked_fid"] = avg_masked_fid_union
                wandb_log["validation_masked_ssim"] = avg_masked_ssim_union
                wandb_log["validation_masked_ms_ssim"] = avg_masked_ms_ssim_union
                wandb_log["validation_masked_lpips"] = avg_masked_lpips_union
                wandb_log["validation_masked_psnr"] = avg_masked_psnr_union
                wandb_log["validation_ema_masked_fid"] = avg_ema_masked_fid_union
                wandb_log["validation_ema_masked_ssim"] = avg_ema_masked_ssim_union
                wandb_log["validation_ema_masked_ms_ssim"] = avg_ema_masked_ms_ssim_union
                wandb_log["validation_ema_masked_lpips"] = avg_ema_masked_lpips_union
                wandb_log["validation_ema_masked_psnr"] = avg_ema_masked_psnr_union

            if len(gen_vcdr_abs_errors_nonema) > 0:
                vcdr_mae = float(np.mean(gen_vcdr_abs_errors_nonema))
                vcdr_rmse = float(np.sqrt(np.mean(np.array(gen_vcdr_abs_errors_nonema) ** 2)))
                cup_valid_rate = float(np.mean(cup_inside_disc_flags_nonema))
                wandb_log["validation_generated_vcdr_mae"] = vcdr_mae
                wandb_log["validation_generated_vcdr_rmse"] = vcdr_rmse
                wandb_log["validation_cup_inside_disc_rate"] = cup_valid_rate
                if len(gen_vcdr_gt_list_nonema) >= 2:
                    try:
                        from scipy import stats as _st
                        r_val, _ = _st.pearsonr(
                            gen_vcdr_gt_list_nonema, gen_vcdr_pred_list_nonema
                        )
                        wandb_log["validation_generated_vcdr_pearson_r"] = float(r_val)
                    except Exception:
                        pass
                logger.info(
                    f"[GEN vCDR] n={len(gen_vcdr_abs_errors_nonema)}  "
                    f"MAE={vcdr_mae:.4f}  RMSE={vcdr_rmse:.4f}  "
                    f"cup_valid_rate={cup_valid_rate:.3f}"
                )

            if len(gen_vcdr_abs_errors_ema) > 0:
                vcdr_mae_ema = float(np.mean(gen_vcdr_abs_errors_ema))
                vcdr_rmse_ema = float(np.sqrt(np.mean(np.array(gen_vcdr_abs_errors_ema) ** 2)))
                cup_valid_rate_ema = float(np.mean(cup_inside_disc_flags_ema))
                wandb_log["validation_ema_generated_vcdr_mae"] = vcdr_mae_ema
                wandb_log["validation_ema_generated_vcdr_rmse"] = vcdr_rmse_ema
                wandb_log["validation_ema_cup_inside_disc_rate"] = cup_valid_rate_ema
                if len(gen_vcdr_gt_list_ema) >= 2:
                    try:
                        from scipy import stats as _st
                        r_val_ema, _ = _st.pearsonr(
                            gen_vcdr_gt_list_ema, gen_vcdr_pred_list_ema
                        )
                        wandb_log["validation_ema_generated_vcdr_pearson_r"] = float(r_val_ema)
                    except Exception:
                        pass
                logger.info(
                    f"[EMA GEN vCDR] n={len(gen_vcdr_abs_errors_ema)}  "
                    f"MAE={vcdr_mae_ema:.4f}  RMSE={vcdr_rmse_ema:.4f}  "
                    f"cup_valid_rate={cup_valid_rate_ema:.3f}"
                )

            tracker.log(wandb_log)
        elif tracker.name == "local":
            local_save_dir = os.path.join(args.output_dir, "validation_images", f"step_{step}")
            os.makedirs(local_save_dir, exist_ok=True)
            
            for idx, log in enumerate(image_logs):
                images = log["images"]
                ema_images = log["ema_images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                validation_condition = log["validation_condition"]
                
                control_path = os.path.join(local_save_dir, f"sample_{idx}_control.png")
                validation_condition.save(control_path)
                
                input_path = os.path.join(local_save_dir, f"sample_{idx}_input.png")
                validation_image.save(input_path)
                
                for img_idx, image in enumerate(images):
                    img_path = os.path.join(local_save_dir, f"sample_{idx}_generated_{img_idx}.png")
                    image.save(img_path)
                
                for img_idx, image in enumerate(ema_images):
                    img_path = os.path.join(local_save_dir, f"sample_{idx}_ema_generated_{img_idx}.png")
                    image.save(img_path)
                
                prompt_path = os.path.join(local_save_dir, f"sample_{idx}_prompt.txt")
                with open(prompt_path, 'w', encoding='utf-8') as f:
                    f.write(f"{validation_prompt}\n")
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        if reward_model is not None:
            reward_model = None

    return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--reward_model_name_or_path",
        type=str,
        default=None,
        help="Path to reward model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--grad_scale", type=float, default=1, help="Scale divided for grad loss value."
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="v_prediction",
        choices=["epsilon", "v_prediction"],
        help="Set noise scheduler config.prediction_type. Use 'epsilon' or 'v_prediction'.",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--control_guidance_start",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--control_guidance_end",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=8000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=30, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=16,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default='limingcv/reward_controlnet',
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--timestep_sampling_start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--timestep_sampling_end",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--min_timestep_rewarding",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max_timestep_rewarding",
        type=int,
        default=800,
    )
    parser.add_argument(
        "--loss_weight_strategy",
        type=str,
        default="fixed_timestep",
        choices=["fixed_timestep", "cosine_decay", "snr", "snr*cosine", "piecewise_cosine"],
        help=(
            "Strategy for determining loss weights for intermediate image decoding losses. "
            "'fixed_timestep': Use min/max_timestep_rewarding to filter samples (original method). "
            "'cosine_decay': Apply cosine decay weight based on timestep. "
            "'snr': Use Signal-to-Noise Ratio (SNR) as weight (normalized). "
            "'snr*cosine': Combined SNR × Cosine weight (w_cos^gamma * w_snr^beta_snr). "
            "'piecewise_cosine': Piecewise cosine weight (1.0 when x <= t1, cosine decay when x > t1). "
            "Affects reward_loss, total_clip_loss, and avg_cup_disc_loss."
        ),
    )
    parser.add_argument(
        "--cosine_decay_min_weight",
        type=float,
        default=1.0,
        help="Minimum weight value for cosine decay strategy (applied at max timestep).",
    )
    parser.add_argument(
        "--cosine_decay_max_weight",
        type=float,
        default=1000.0,
        help="Maximum weight value for cosine decay strategy (applied at min timestep).",
    )
    parser.add_argument(
        "--snr_cosine_gamma",
        type=float,
        default=1.0,
        help="Exponent for cosine weight in snr*cosine strategy. Default: 1.0.",
    )
    parser.add_argument(
        "--snr_cosine_beta_snr",
        type=float,
        default=0.5,
        help="Exponent for SNR weight in snr*cosine strategy. Default: 0.5.",
    )
    parser.add_argument(
        "--piecewise_cosine_t1",
        type=float,
        default=700.0,
        help="Breakpoint t1 for piecewise_cosine strategy. When timestep <= t1, weight is 1.0; when timestep > t1, weight uses cosine decay. Default: 700.0.",
    )
    parser.add_argument(
        "--reward_loss_weight",
        type=float,
        default=1.0,
        help="Weight for reward loss in the total loss calculation.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default='segmentation',
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--keep_in_memory",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--use_filter",
        action="store_true",
        help="Whether to use filter_file.txt to filter the dataset.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--mask_only_conditioning",
        action="store_true",
        default=False,
        help="If set, use pure mask (black background) as conditioning, matching "
             "the original ControlNet / ControlNet++ semantic-segmentation protocol. "
             "Default: use CLAHE-enhanced fundus as background with color-coded "
             "cup/rim overlay (this codebase's convention).",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default=None,
        help="The column of the dataset containing the original labels. `seg_map` for ADE20K; `panoptic_seg_map` for COCO-Stuff.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=10,
        help=(
            "Max number of samples for validation during training, default to 10"
        ),
    )
    parser.add_argument(
        "--image_condition_dropout",
        type=float,
        default=0,
        help="Probability of image conditions to be replaced with tensors with zero value. Defaults to 0.",
    )
    parser.add_argument(
        "--text_condition_dropout",
        type=float,
        default=0,
        help="Probability of image prompts to be replaced with empty strings. Defaults to 0.05.",
    )
    parser.add_argument(
        "--all_condition_dropout",
        type=float,
        default=0,
        help="Probability of abandon all the conditions.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="reward_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--clip_loss_weight_v1_1",
        type=float,
        default=0.0,
        help="Weight for CLIP loss in the total loss calculation.",
    )
    parser.add_argument(
        "--clip_model_for_loss",
        type=str,
        default="baseline",
        choices=["custom", "baseline", "medclip"],
        help="Which CLIP model to use for calculating CLIP loss: 'custom' (self-trained), 'baseline' (pretrained), or 'medclip' (MedCLIP).",
    )
    parser.add_argument(
        "--clip_model_for_metric",
        type=str,
        default="baseline",
        choices=["custom", "baseline", "medclip"],
        help="Which CLIP model to use for calculating CLIP metrics: 'custom' (self-trained), 'baseline' (pretrained), or 'medclip' (MedCLIP).",
    )
    parser.add_argument(
        "--custom_clip_model_path",
        type=str,
        default="./checkpoints/custom_clip.ckpt",
        help="Path to the custom trained CLIP model checkpoint. If not provided, custom model will not be loaded.",
    )
    parser.add_argument(
        "--cup_disc_loss_weight_v1_1",
        type=float,
        default=1.0,
        help="Weight for cup-disc ratio loss in the total loss calculation.",
    )
    parser.add_argument(
        "--combined_loss_weight",
        type=float,
        default=1.0,
        help="Weight for combined loss (reward_loss + total_clip_loss + avg_cup_disc_loss) in the total loss calculation.",
    )
    parser.add_argument(
        "--cup_disc_loss_max",
        type=float,
        default=1.0,
        help="[DEPRECATED after loss v2] Maximum value for cup-disc ratio loss clipping. "
             "Ignored by the new Huber-based vCDR loss; kept only for CLI backward-compat.",
    )
    parser.add_argument(
        "--vcdr_loss_alpha",
        type=float,
        default=10.0,
        help="Temperature α for soft row pooling in the σ-ratio vCDR surrogate. "
             "Larger α → closer to hard max. α=10 gives dense gradients on smooth "
             "probability maps while keeping the mask-side target close to hard max.",
    )
    parser.add_argument(
        "--use_uncertainty_weight",
        action="store_true",
        help="Whether to use uncertainty weighting in DiceLoss for segmentation task.",
    )
    parser.add_argument(
        "--uncertainty_loss_weight",
        type=float,
        default=0.0,
        help="Weight for uncertainty MSE loss. If > 0, adds MSE loss to encourage reducing prediction uncertainty. Defaults to 0.0.",
    )
    parser.add_argument(
        "--reward_use_cross_entropy",
        action="store_true",
        help="For the FairSeg dataset, use cross-entropy loss instead of DiceUncertaintyLoss for reward loss calculation. Defaults to False (use DiceUncertaintyLoss).",
    )
    parser.add_argument(
        "--enable_random_disc_replacement",
        action="store_true",
        help="Enable random disc replacement in FairSegDataset. When enabled, the disc (but not the cup) will be randomly replaced with another sample's disc, aligned by center position.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.text_condition_dropout < 0 or args.text_condition_dropout > 1:
        raise ValueError("`--text_condition_dropout` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


class FairSegDataset(torch.utils.data.Dataset):
    """FairSeg dataset loader (npz files with `disc_cup_mask` and demographic split in `data_summary.csv`)."""
    def __init__(self, args, split='train', gen_data=False, file_list=None, gen_scale_masks=False, tokenizer=None):
        self.args = args
        self.split = split
        self.gen_data = gen_data
        self.gen_scale_masks = gen_scale_masks
        self.data = []
        self.tokenizer = tokenizer
        
        self.enable_random_disc_replacement = getattr(args, 'enable_random_disc_replacement', False)

        # Initialize CLAHE for image enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Define color palette for mask visualization
        self.palette = {
            0: (0, 0, 0),
            -2: (255, 0, 0),
            -1: (0, 0, 255),
            3: (0, 255, 0),
        }

        # Load summary file
        summary_path = os.path.join(args.dataset_name, 'data_summary.csv')
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Summary file not found at {summary_path}")

        self.summary_data = pd.read_csv(summary_path).set_index('filename')

        if file_list is not None:
            self.data = file_list
        else:
            # Get all npz files from All directory
            all_npz_files = glob(os.path.join(args.dataset_name, 'All', "*.npz"))

        # Apply filter if filter_file.txt exists and use_filter is enabled
        if hasattr(args, 'use_filter') and args.use_filter:
            filter_path = os.path.join(args.dataset_name, 'filter_file.txt')
            if os.path.exists(filter_path):
                with open(filter_path, 'r') as f:
                    filter_list = f.read()
                    filter_list = filter_list.split("\n")

                # Convert filter list to match filename format
                filter_list = [i.replace(".png", ".npz").replace(".jpg", ".npz") for i in filter_list]
                filter_list = set(filter_list)
                if "" in filter_list:
                    filter_list.remove("")

                # Filter npz files based on filename
                all_npz_files = [f for f in all_npz_files if os.path.basename(f) in filter_list]
                logger.info(f"Applied filter, remaining files: {len(all_npz_files)}")
            else:
                raise FileNotFoundError(f"Filter file not found at {filter_path}")

        # Filter files based on split using summary data
        for npz_file in all_npz_files:
            filename = os.path.basename(npz_file)
            if filename in self.summary_data.index:
                use_value = self.summary_data.loc[filename, 'use']
                if (split == 'train' and use_value == 'training') or \
                   (split == 'validation' and use_value == 'validation') or \
                   (split == 'test' and use_value == 'test'):
                    self.data.append(npz_file)

        logger.info(f"Loaded {len(self.data)} {split} samples")

    def __len__(self):
        return len(self.data)

    def random_crop(self, image, start_h, start_w, crop_height, crop_width):
        """Random crop function similar to MyDataset"""
        if image.shape[0] < crop_height or image.shape[1] < crop_width:
            raise ValueError("Image dimensions should be larger than crop dimensions")
        cropped_image = image[start_h:start_h+crop_height, start_w:start_w+crop_width]
        return cropped_image

    def prepare_transunet_input(self, image):
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        
        if image.shape != (224, 224):
            from scipy.ndimage import zoom
            zoom_factors = (224 / image.shape[0], 224 / image.shape[1])
            image = zoom(image, zoom_factors, order=3)
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        
        return image

    def check_contact(self,mask,max_min_distance=4):

        blue_mask = (mask == -2).astype(np.uint8)
        red_mask = np.logical_or(mask == -1, mask == -2).astype(np.uint8)
        red_contours = measure.find_contours(red_mask, 0.99)[0]
        blue_contours = measure.find_contours(blue_mask, 0.99)[0]

        mask1_polygon = Polygon(red_contours)
        mask2_polygon = Polygon(blue_contours)
        distance = mask1_polygon.exterior.distance(mask2_polygon.exterior)
        if mask2_polygon.within(mask1_polygon) and distance > max_min_distance:
            return False
        else:
            return True

    def cal_ellipse(self,mask,origin_mask):
        contours = measure.find_contours(mask, 0.99)[0]
        ellipse = cv2.fitEllipse(contours.reshape(-1,1,2).astype(int))
        # _,(short,long),_ = ellipse
        mask = np.ascontiguousarray(mask)
        out = cv2.ellipse(mask, ellipse, (3), 2)
        rows, cols = np.where(out == 3)
        # vCDR (vertical cup-to-disc ratio): use vertical diameter max_y-min_y
        # to match clinical definition used in glaucoma screening (Jonas 1999).
        max_y = np.max(rows)
        min_y = np.min(rows)
        long = max_y - min_y
        return long,origin_mask


    def cal_proportion(self,mask):

        red_mask = np.logical_or(mask == -1, mask == -2).astype(np.uint8)
        blue_mask = (mask == -2).astype(np.uint8)
        red_long,_ = self.cal_ellipse(red_mask,mask)

        blue_long,_ = self.cal_ellipse(blue_mask,mask)

        relative_size = blue_long/red_long
        return round(relative_size,6),mask



    def get_cup_center(self, mask_image):
        blue_mask = (mask_image == -2).astype(np.uint8)
        
        if np.sum(blue_mask) == 0:
            return None
        
        rows, cols = np.where(blue_mask > 0)
        center_row = np.mean(rows)
        center_col = np.mean(cols)
        
        return (center_row, center_col)

    def replace_cup_keep_disc(self, current_mask, other_mask):
        current_disc_mask = np.logical_or(current_mask == -1, current_mask == -2).astype(np.float64)
        
        other_cup_mask = (other_mask == -2).astype(np.float64)
        
        if np.sum(other_cup_mask) == 0:
            return current_mask.copy()
        
        if np.sum(current_disc_mask) == 0:
            return current_mask.copy()
        
        current_cup_center = self.get_cup_center(current_mask)
        if current_cup_center is None:
            return current_mask.copy()
        
        other_cup_center = self.get_cup_center(other_mask)
        if other_cup_center is None:
            return current_mask.copy()
        
        other_mask_processed = other_mask.copy()
        
        offset_row = int(current_cup_center[0] - other_cup_center[0])
        offset_col = int(current_cup_center[1] - other_cup_center[1])
        
        disc_mask_bool = current_disc_mask > 0
        
        other_cup_rows, other_cup_cols = np.where(other_mask_processed == -2)
        
        new_rows = other_cup_rows + offset_row
        new_cols = other_cup_cols + offset_col
        
        is_out_of_disc = False
        if len(new_rows) > 0:
            valid_count = 0
            for i in range(len(new_rows)):
                r, c = new_rows[i], new_cols[i]
                if 0 <= r < current_mask.shape[0] and 0 <= c < current_mask.shape[1]:
                    if disc_mask_bool[r, c]:
                        valid_count += 1
            
            if valid_count < len(new_rows):
                is_out_of_disc = True
        
        if is_out_of_disc:
            max_iterations = 20
            iteration = 0
            scale_factor = 0.95
            
            while is_out_of_disc and iteration < max_iterations:
                try:
                    other_mask_processed = self.scale_bule_mask(other_mask_processed, scale_factor)
                    
                    blue_mask_check = (other_mask_processed == -2).astype(np.uint8)
                    if np.sum(blue_mask_check) == 0:
                        return current_mask.copy()
                    
                    other_cup_center_new = self.get_cup_center(other_mask_processed)
                    if other_cup_center_new is None:
                        return current_mask.copy()
                    
                    offset_row = int(current_cup_center[0] - other_cup_center_new[0])
                    offset_col = int(current_cup_center[1] - other_cup_center_new[1])
                    
                    other_cup_rows, other_cup_cols = np.where(other_mask_processed == -2)
                    new_rows = other_cup_rows + offset_row
                    new_cols = other_cup_cols + offset_col
                    
                    if len(new_rows) > 0:
                        valid_count = 0
                        for i in range(len(new_rows)):
                            r, c = new_rows[i], new_cols[i]
                            if 0 <= r < current_mask.shape[0] and 0 <= c < current_mask.shape[1]:
                                if disc_mask_bool[r, c]:
                                    valid_count += 1
                        
                        if valid_count == len(new_rows):
                            is_out_of_disc = False
                            break
                    else:
                        is_out_of_disc = False
                        break
                except:
                    break
                
                iteration += 1
        
        replaced_mask = np.zeros_like(current_mask, dtype=np.float64)
        
        replaced_mask[current_disc_mask > 0] = current_mask[current_disc_mask > 0]
        
        replaced_mask[replaced_mask == -2] = -1
        
        other_cup_rows, other_cup_cols = np.where(other_mask_processed == -2)
        
        new_rows = other_cup_rows + offset_row
        new_cols = other_cup_cols + offset_col
        
        valid_new_rows = []
        valid_new_cols = []
        
        for i in range(len(new_rows)):
            r, c = new_rows[i], new_cols[i]
            if 0 <= r < replaced_mask.shape[0] and 0 <= c < replaced_mask.shape[1]:
                if disc_mask_bool[r, c]:
                    valid_new_rows.append(r)
                    valid_new_cols.append(c)
        
        if len(valid_new_rows) == 0:
            return current_mask.copy()
        
        replaced_mask[valid_new_rows, valid_new_cols] = -2
        
        cup_disc_ratio = 1.0
        try:
            cup_disc_ratio, _ = self.cal_proportion(replaced_mask)
        except:
            cup_disc_ratio = 1.0
        
        if cup_disc_ratio > 0.8:
            blue_mask_check = (replaced_mask == -2).astype(np.uint8)
            if np.sum(blue_mask_check) == 0:
                return replaced_mask
            
            max_iterations = 20
            iteration = 0
            scale_factor = 0.95
            
            while cup_disc_ratio > 0.8 and iteration < max_iterations:
                try:
                    replaced_mask = self.scale_bule_mask(replaced_mask, scale_factor)
                    
                    blue_mask_check = (replaced_mask == -2).astype(np.uint8)
                    if np.sum(blue_mask_check) == 0:
                        break
                    
                    cup_disc_ratio, _ = self.cal_proportion(replaced_mask)
                except:
                    break
                
                iteration += 1
        
        return replaced_mask

    def getitem(self, idx):
        """Main data loading function, similar to MyDataset"""
        data_file = self.data[idx]
        raw_data = np.load(data_file, allow_pickle=True)

        # Get filename and summary info
        filename = os.path.basename(data_file)
        info = self.summary_data.loc[filename]

        # Extract image from npz file
        modified_image = raw_data['slo_fundus']
        modified_image = modified_image.astype(np.uint8)
        
        transunet_image = modified_image.copy()
        
        modified_image = np.array([modified_image, modified_image, modified_image]).transpose(1, 2, 0)

        # Apply CLAHE enhancement
        clahe_img = np.zeros_like(modified_image)
        for i in range(3):
            clahe_img[:, :, i] = self.clahe.apply(modified_image[:, :, i])
        mask_image = raw_data['disc_cup_mask']
        label = np.zeros_like(mask_image, dtype=np.int64)
        label[mask_image == -1] = 1
        label[mask_image == -2] = 2

        replaced_mask_image = None
        replaced_label = None
        if self.enable_random_disc_replacement :
            other_idx = random.randint(0, len(self.data) - 1)
            if other_idx != idx:
                try:
                    other_data_file = self.data[other_idx]
                    other_raw_data = np.load(other_data_file, allow_pickle=True)
                    
                    if 'disc_cup_mask' in other_raw_data:
                        other_mask_image = other_raw_data['disc_cup_mask']
                        
                        if other_mask_image.shape == mask_image.shape:
                            replaced_mask_image = self.replace_cup_keep_disc(mask_image, other_mask_image)
                            replaced_label = np.zeros_like(replaced_mask_image, dtype=np.int64)
                            replaced_label[replaced_mask_image == -1] = 1
                            replaced_label[replaced_mask_image == -2] = 2
                except Exception as e:
                    logger.warning(f"Failed to replace cup for sample {idx}: {e}")
        
        processing_mask_image = replaced_mask_image if replaced_mask_image is not None else mask_image
        processing_label = replaced_label if replaced_label is not None else label
        
        # Calculate original proportion
        origin_mask_proportion, processing_mask_image = self.cal_proportion(processing_mask_image)

        # Generate scaled masks if required
        if self.gen_scale_masks:
            scaled_masks, relativs = self.get_scale_mask(processing_mask_image)
            prompts = [', '.join(
                info.iloc[1:4].values.tolist()) + f",Cup-to-Disc Ratio {i}" for i in relativs]
            scaled_masks.append(processing_mask_image)
            relativs.append(origin_mask_proportion)
        else:
            scaled_masks, relativs = [], []
            prompts = []

        # Create main prompt
        prompt = ', '.join(
            info.iloc[1:4].values.tolist())  + f",Cup-to-Disc Ratio {origin_mask_proportion}"
        prompts.append(prompt)

        # Create conditioning image (mask with color coding)
        if getattr(self.args, 'mask_only_conditioning', False):
            clahe_img_mask = np.zeros_like(clahe_img)
        else:
            clahe_img_mask = clahe_img.copy()
        clahe_img_mask[processing_mask_image == -1.0] = self.palette[-1]
        clahe_img_mask[processing_mask_image == -2.0] = self.palette[-2]
        clahe_img_mask[processing_mask_image == 3] = self.palette[3]

        # Apply random crop
        crop_height = crop_width = 512
        start_h = (clahe_img_mask.shape[0] - crop_height + 1) // 2
        start_w = (clahe_img_mask.shape[1] - crop_width + 1) // 2

        source = self.random_crop(clahe_img_mask, start_h, start_w, crop_height, crop_width)
        target = self.random_crop(clahe_img, start_h, start_w, crop_height, crop_width)
        transunet_image = self.random_crop(transunet_image, start_h, start_w, crop_height, crop_width)
        mask_image = self.random_crop(mask_image, start_h, start_w, crop_height, crop_width)
        label = self.random_crop(label, start_h, start_w, crop_height, crop_width)
        processing_label = self.random_crop(processing_label, start_h, start_w, crop_height, crop_width)
        if replaced_mask_image is not None:
            replaced_mask_image = self.random_crop(replaced_mask_image, start_h, start_w, crop_height, crop_width)

        # Process scaled masks
        for i in range(len(scaled_masks)):
            scaled_mask = scaled_masks[i]
            if getattr(self.args, 'mask_only_conditioning', False):
                target_img_copy = np.zeros_like(clahe_img)
            else:
                target_img_copy = clahe_img.copy()
            target_img_copy[scaled_mask == -1.0] = self.palette[-1]
            target_img_copy[scaled_mask == -2.0] = self.palette[-2]
            target_img_copy = target_img_copy.astype(np.uint8)
            target_img_copy = self.random_crop(target_img_copy, start_h, start_w, crop_height, crop_width)
            target_img_copy = cv2.cvtColor(target_img_copy, cv2.COLOR_BGR2RGB)
            target_img_copy = target_img_copy.astype(np.float32) / 255.0
            scaled_masks[i] = target_img_copy

        # Apply data augmentation for training
        # if not self.gen_data:
        #     r = random.randint(1, 100)
        #     # r = 1
        #     if r >= 50:
        #         source = cv2.flip(source, 1)
        #         target = cv2.flip(target, 1)
        #         transunet_image = cv2.flip(transunet_image, 1)
        #         mask_image = cv2.flip(mask_image, 1)
        #         label = cv2.flip(label, 1)
        #         processing_label = cv2.flip(processing_label, 1)
        #         if replaced_mask_image is not None:
        #             replaced_mask_image = cv2.flip(replaced_mask_image, 1)
        #     r = random.randint(1, 100)
        #     # r = 1
        #     if r >= 50:
        #         source = cv2.flip(source, 0)
        #         target = cv2.flip(target, 0)
        #         transunet_image = cv2.flip(transunet_image, 0)
        #         mask_image = cv2.flip(mask_image, 0)
        #         label = cv2.flip(label, 0)
        #         processing_label = cv2.flip(processing_label, 0)
        #         if replaced_mask_image is not None:
        #             replaced_mask_image = cv2.flip(replaced_mask_image, 0)

        # Convert to RGB and normalize
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        clahe_target = target.copy()

        # Normalize source images to [0, 1]
        conditioning_pixel_values = source.astype(np.float32) / 255.0
        # Normalize target images to [-1, 1]
        pixel_values = (target.astype(np.float32) / 127.5) - 1.0

        conditioning_pixel_values = conditioning_pixel_values.transpose(2, 0, 1)
        pixel_values = pixel_values.transpose(2, 0, 1)
        
        transunet_input = self.prepare_transunet_input(transunet_image)


        # Tokenize caption
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                prompt, 
                max_length=self.tokenizer.model_max_length, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            ).input_ids.squeeze(0)
        else:
            input_ids = torch.tensor([0])  # placeholder



        # Extract additional metadata
        name = filename.replace(".npz", "")
        glaucoma_label = 0  # Default value
        race = int(raw_data.get('race', 0))
        gender = int(raw_data.get('gender', 0))
        ethnicity = int(raw_data.get('ethnicity', 0))
        language = int(raw_data.get('language', 0))
        
        return_dict = {
            'pixel_values': pixel_values,
            'conditioning_pixel_values': conditioning_pixel_values,
            'input_ids': input_ids,
            'labels': processing_label,
            'prompt': prompt,
            'name': name,
            'glaucoma_label': glaucoma_label,
            'race': race,
            'gender': gender,
            'ethnicity': ethnicity,
            'language': language,
            'clahe_target': clahe_target,
            'extra_mask': scaled_masks,
            'relativs': relativs,
            'prompts': prompts,
            'origin_mask_proportion':origin_mask_proportion,
            'transunet_input': transunet_input,
            'mask_image': mask_image,
            'label': label
        }
        
        if replaced_mask_image is not None:
            return_dict['replaced_mask_image'] = replaced_mask_image
            return_dict['replaced_label'] = processing_label
        
        return return_dict
    def scale_bule_mask(self,mask_image,scale_factor):
        red_mask = np.logical_or(mask_image == -1, mask_image == -2).astype(np.uint8)
        blue_mask = (mask_image == -2).astype(np.uint8)
        zero_mask = np.logical_or(mask_image == 0, mask_image == None)
        rows, cols = np.where(blue_mask > 0)
        blue_center = (np.mean(rows), np.mean(cols))
        top, bottom = np.min(rows), np.max(rows)
        left, right = np.min(cols), np.max(cols)
        blue_mask_scaled = ndimage.zoom(blue_mask[top:bottom + 1, left:right + 1], scale_factor, order=0)
        blue_mask_scaled = blue_mask_scaled.astype(np.float64)
        blue_mask_scaled[blue_mask_scaled == 1] = -2
        blue_mask_scaled[blue_mask_scaled == 0] = -1
        if scale_factor < 1:
            new_top = top + (bottom - top + 1) * (1 - scale_factor) // 2
            new_bottom = bottom - (bottom - top + 1) * (1 - scale_factor) // 2
            new_left = left + (right - left + 1) * (1 - scale_factor) // 2
            new_right = right - (right - left + 1) * (1 - scale_factor) // 2
        else:
            new_top = top - (bottom - top + 1) * (scale_factor - 1) // 2
            new_bottom = bottom + (bottom - top + 1) * (scale_factor - 1) // 2
            new_left = left - (right - left + 1) * (scale_factor - 1) // 2
            new_right = right + (right - left + 1) * (scale_factor - 1) // 2
        new_center = (new_top + (new_bottom - new_top) / 2, new_left + (new_right - new_left) / 2)
        offset_row = int(blue_center[0] - new_center[0])
        offset_col = int(blue_center[1] - new_center[1])
        new_top = int(new_top)
        new_bottom = int(new_bottom)
        new_left = int(new_left)
        new_right = int(new_right)

        scaled_mask = np.zeros_like(mask_image)
        scaled_mask[red_mask > 0] = -1

        scaled_mask[new_top + offset_row:new_top + offset_row + blue_mask_scaled.shape[0],
        new_left + offset_col:new_left + offset_col + blue_mask_scaled.shape[1]] = blue_mask_scaled
        scaled_mask[zero_mask] = 0
        return scaled_mask
    def solve(self,mask_image,target_relative_size):
        def solve_func(params, mask_image,target_relative_size):
            [scale] = params
            scaled_mask = self.scale_bule_mask(mask_image, scale)
            try:
                relative_size,_ = self.cal_proportion(scaled_mask)
            except:

                relative_size = 10


            distance = abs(target_relative_size - relative_size)

            return distance

        initial_guess = [1]

        result = minimize(solve_func, initial_guess, args=(mask_image,target_relative_size),method='nelder-mead',tol=1e-3,bounds=[(0.1,3)])
        [relative_size] = result.x

        return relative_size

    def get_scale_mask(self, mask_image):

        mask_relative_size, _ = self.cal_proportion(mask_image)
        scale_factors = np.arange(-0.8, 2, 0.05)
        # scale_factors = [1]
        scaled_masks = []
        relativs = []
        for scale_factor in scale_factors:
            try:
                if scale_factor == mask_relative_size:
                    continue
                scale = self.solve(mask_image, target_relative_size=mask_relative_size + scale_factor)

                scaled_mask = self.scale_bule_mask(mask_image, scale)
                relative_size, scaled_mask = self.cal_proportion(scaled_mask)
                try:
                    contact = self.check_contact(scaled_mask)
                except:
                    contact = True

                if relative_size >= 0.10 and relative_size <= 0.90 and contact == False:
                    scaled_masks.append(scaled_mask)
                    relativs.append(relative_size)
                elif contact == True or relative_size > 0.90:
                    break
            except:
                continue

        return scaled_masks, relativs

    def __getitem__(self, idx):
        """Main entry point with error handling, similar to MyDataset"""
        while True:
            try:
                return self.getitem(idx)
            except Exception as e:
                # logger.warning(f"Error loading sample {idx}: {e}")
                idx = random.randint(0, len(self) - 1)


def load_fairseg_dataset(args, split='train', tokenizer=None):
    """Load the FairSeg dataset using the summary file's `use` column for the split."""
    dataset = FairSegDataset(args, split, tokenizer=tokenizer)

    # Return dataset directly for PyTorch DataLoader
    if split == 'train':
        return {
            'train': dataset
        }
    elif split == 'validation':
        return {
            'validation': dataset
        }
    elif split == 'test':
        return {
            'test': dataset
        }


def make_train_dataset(args, tokenizer, accelerator, split='train'):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Check if it's the FairSeg dataset
        if 'fairseg' in args.dataset_name.lower():
            # Load the FairSeg dataset with custom logic
            dataset = load_fairseg_dataset(args, split, tokenizer)
            # For FairSeg, return the dataset directly as it's already a PyTorch dataset
            return dataset, dataset[split]
        else:
            # Original dataset loading logic
            if args.dataset_name.count('/') == 1:
                # Downloading and loading a dataset from the hub.
                dataset = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    cache_dir=args.cache_dir,
                    keep_in_memory=args.keep_in_memory,
                )
            else:
                dataset = load_from_disk(
                    dataset_path=args.dataset_name,
                    keep_in_memory=args.keep_in_memory,
                )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
                keep_in_memory=args.keep_in_memory,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset[split].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    elif args.conditioning_image_column in ['canny', 'lineart', 'hed']:
        conditioning_image_column = image_column
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    resolution = (args.resolution, args.resolution)
    image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            # transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            # transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    label_image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST, antialias=True),
            # transforms.CenterCrop(args.resolution),
        ]
    )

    def preprocess_train(examples):
        # Handle both PIL images and file paths
        if isinstance(examples[image_column][0], str):
            # Load images from file paths
            pil_images = [Image.open(image_path).convert("RGB") for image_path in examples[image_column]]
        else:
            # Handle PIL images directly
            pil_images = [image.convert("RGB") for image in examples[image_column]]

        images = [image_transforms(image) for image in pil_images]

        if args.conditioning_image_column in ['canny', 'lineart', 'hed']:
            conditioning_images = images
        else:
            # Handle both PIL images and file paths for conditioning images
            if isinstance(examples[conditioning_image_column][0], str):
                conditioning_pil_images = [Image.open(image_path).convert("RGB") for image_path in examples[conditioning_image_column]]
            else:
                conditioning_pil_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
            conditioning_images = [conditioning_image_transforms(image) for image in conditioning_pil_images]

        if args.label_column is not None:
            dtype = torch.long
            labels = [torch.tensor(np.asarray(label), dtype=dtype).unsqueeze(0) for label in examples[args.label_column]]
            labels = [label_image_transforms(label) for label in labels]

        # perform groupped random crop for image/conditioning_image/label
        if args.label_column is not None:
            grouped_data = [torch.cat([x, y, z]) for (x, y, z) in zip(images, conditioning_images, labels)]
            grouped_data = group_random_crop(grouped_data, args.resolution)

            images = [x[:3, :, :] for x in grouped_data]
            conditioning_images = [x[3:6, :, :] for x in grouped_data]
            labels = [x[6:, :, :] for x in grouped_data]

            # (1, H, W) => (H, w)
            if args.task_name == "segmentation":
                labels = [label.squeeze(0) for label in labels]

            examples[args.label_column] = labels
        else:
            grouped_data = [torch.cat([x, y]) for (x, y) in zip(images, conditioning_images)]
            grouped_data = group_random_crop(grouped_data, args.resolution)

            images = [x[:3, :, :] for x in grouped_data]
            conditioning_images = [x[3:, :, :] for x in grouped_data]

        # Dropout some of features for classifier-free guidance.
        for i, img_condition in enumerate(conditioning_images):
            rand_num = random.random()
            if rand_num < args.image_condition_dropout:
                conditioning_images[i] = torch.zeros_like(img_condition)
            elif rand_num < args.image_condition_dropout + args.text_condition_dropout:
                examples[caption_column][i] = ""
            elif rand_num < args.image_condition_dropout + args.text_condition_dropout + args.all_condition_dropout:
                conditioning_images[i] = torch.zeros_like(img_condition)
                examples[caption_column][i] = ""

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed)
            # rewrite the shuffled dataset on disk as contiguous chunks of data
            dataset["train"] = dataset["train"].flatten_indices()
            dataset["train"] = dataset["train"].select(range(args.max_train_samples))

        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

        # For FairSeg, also create validation dataset if not exists
        if 'fairseg' in args.dataset_name.lower():
            if 'validation' not in dataset:
                val_dataset = load_fairseg_dataset(args, 'validation', tokenizer)
                dataset.update(val_dataset)

    return dataset, train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([torch.tensor(example["pixel_values"]) for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    origin_mask_proportion = torch.stack([torch.tensor(example["origin_mask_proportion"]) for example in examples])
    origin_mask_proportion = origin_mask_proportion.to(memory_format=torch.contiguous_format).float()


    conditioning_pixel_values = torch.stack([torch.tensor(example["conditioning_pixel_values"]) for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    # Check if this is the FairSeg dataset (has 'labels' key directly)
    if "labels" in examples[0]:
        labels = torch.stack([torch.tensor(example["labels"]) for example in examples])
        labels = labels.to(memory_format=torch.contiguous_format).float()
    elif args.label_column is not None:
        labels = torch.stack([example[args.label_column] for example in examples])
        labels = labels.to(memory_format=torch.contiguous_format).float()
    else:
        labels = conditioning_pixel_values

    if "transunet_input" in examples[0]:
        transunet_inputs = torch.stack([example["transunet_input"] for example in examples])
        transunet_inputs = transunet_inputs.to(memory_format=torch.contiguous_format).float()
    else:
        transunet_inputs = None
    original_texts = [example["prompt"] for example in examples]


    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
        "labels": labels,
        "transunet_inputs": transunet_inputs,
        "original_texts": original_texts,
        "origin_mask_proportion": origin_mask_proportion,
    }


def main(args):

    if os.environ.get("USE_LOCAL_PROXY", "0") == "1":
        proxy_url = os.environ.get("HTTP_PROXY_URL", "http://127.0.0.1:7890")
        os.environ['HTTP_PROXY']  = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        print(f"[proxy] Using local proxy: {proxy_url}")
    else:
        for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(_v, None)


    from datetime import datetime
    import random
    import string
    
    current_date = datetime.now().strftime("%m%d")
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))

    # === ablation tag ===
    _ablation_tag = os.environ.get("ABLATION_TAG", "").strip()
    _ablation_prefix = f"{_ablation_tag}__" if _ablation_tag else ""
    

    

    
    param_suffix_parts = []
    
    if args.clip_loss_weight_v1_1 != 0:
        param_suffix_parts.append("clip_loss1.1")
    
    if args.reward_loss_weight != 0:
        param_suffix_parts.append("reward_loss")
    
    if args.use_uncertainty_weight:
        param_suffix_parts.append("use_uncertainty_weight")
    
    if args.uncertainty_loss_weight != 0:
        param_suffix_parts.append("uncertainty_loss")
    
    if args.cup_disc_loss_weight_v1_1 != 0:
        param_suffix_parts.append("cup_disc_loss1.1")
    
    if args.combined_loss_weight != 1.0:
        param_suffix_parts.append(f"combined_loss{args.combined_loss_weight}")

    if (args.reward_loss_weight == 0
            and args.uncertainty_loss_weight == 0
            and args.cup_disc_loss_weight_v1_1 == 0
            and args.clip_loss_weight_v1_1 == 0):
        param_suffix_parts.append("pretrain_only")

    param_suffix_parts.append(f"{args.loss_weight_strategy}")
    
    if args.reward_model_name_or_path:
        if args.reward_model_name_or_path.lower().startswith("segman"):
            param_suffix_parts.append("segman")
        elif "TU" in args.reward_model_name_or_path:
            param_suffix_parts.append("TU")
    
    param_suffix = "_" + "_".join(param_suffix_parts) if param_suffix_parts else ""
    
    output_path = Path(args.output_dir)
    if len(output_path.parts) > 1:
        new_parts = list(output_path.parts[:-1]) + [f"{_ablation_prefix}{current_date}_{args.prediction_type}_{random_suffix}{param_suffix}"]
        args.output_dir = str(Path(*new_parts))
    else:
        args.output_dir = f"{_ablation_prefix}{current_date}_{args.prediction_type}_{random_suffix}{param_suffix}"
    
    print(f"Generated output directory: {args.output_dir}  {args.output_dir}")
    
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    _report_list = (
        [args.report_to] if isinstance(args.report_to, str) else list(args.report_to)
    )
    if any(x == "wandb" for x in _report_list) or args.report_to == "all":
        if not is_wandb_available():
            raise RuntimeError(
                "report_to contains 'wandb' but the wandb package is not installed in the current Python env.\n"
                "Run: pip install wandb && wandb login\n"
                "To skip wandb, pass --report_to=tensorboard or --report_to=none."
            )
        try:
            import wandb as _wb
            assert hasattr(_wb, "init") and hasattr(_wb, "__version__"), (
                f"wandb is installed but the imported module is not the real package (__file__={getattr(_wb, '__file__', None)}).\n"
                "Typical cause: a same-named file/dir in cwd shadows the package. Start from a different cwd or remove the shadowing path."
            )
        except Exception as _e:
            raise RuntimeError(f"wandb sanity check failed: {_e}")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    global InceptionV3_model, loss_fn_alex
    if InceptionV3_model is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        InceptionV3_model = InceptionV3([block_idx]).to(accelerator.device)
        logger.info(f"InceptionV3 model initialized on device: {accelerator.device}")
    if loss_fn_alex is None:
        loss_fn_alex = lpips.LPIPS(net='alex').to(accelerator.device)
        logger.info(f"LPIPS model initialized on device: {accelerator.device}")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    #   stabilityai/stable-diffusion-2-1       -> v_prediction (768)
    #   stabilityai/stable-diffusion-2-1-base  -> epsilon      (512)
    if noise_scheduler.config.prediction_type != args.prediction_type:
        raise ValueError(
            f"Prediction-type mismatch: pretrained '{args.pretrained_model_name_or_path}' "
            f"ships scheduler.prediction_type='{noise_scheduler.config.prediction_type}', "
            f"but --prediction_type={args.prediction_type}. UNet is frozen in this script, "
            f"so forcing the scheduler alone corrupts the training target. "
            f"Use 'stabilityai/stable-diffusion-2-1' for v_prediction (768px) or "
            f"'stabilityai/stable-diffusion-2-1-base' for epsilon (512px)."
        )
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    reward_model = get_reward_model(args.task_name, args.reward_model_name_or_path, device=accelerator.device)

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1

            while len(weights) > 0:
                weights.pop()
                model = models[i]

                sub_dir = "controlnet"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    reward_model.requires_grad_(False)
    controlnet.train()

    # Create EMA for the ControlNet.
    if args.use_ema:

        if args.controlnet_model_name_or_path:
            ema_controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
        else:
            ema_controlnet = ControlNetModel.from_unet(unet)
        ema_controlnet = EMAModel(ema_controlnet.parameters(), model_cls=ControlNetModel, model_config=ema_controlnet.config)

    else:
        ema_controlnet = None

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    if args.use_ema:
        ema_controlnet.to(accelerator.device)

    # Optimizer creation
    # optimized_parameters = list(controlnet.parameters()) + list(reward_model.parameters()) + list(unet.parameters())
    optimizer = optimizer_class(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset, train_dataset = make_train_dataset(args, tokenizer, accelerator)

    if args.validation_prompt is None and args.validation_image is None:
        if 'validation' in dataset.keys():
            val_dataset = dataset['validation']
        else:
            # For non-FairSeg datasets, create validation split
            if 'fairseg' not in args.dataset_name.lower():
                dataset = train_dataset.train_test_split(test_size=0.00005)
                train_dataset, val_dataset = dataset['train'], dataset['test']
            else:
                val_dataset = load_fairseg_dataset(args, "validation", tokenizer)
    else:
        val_dataset = None

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # unet, reward_model = accelerator.prepare(unet, reward_model)

    # Prepare others after preparing the model
    controlnet,ema_controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet,ema_controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    reward_model.to(accelerator.device, dtype=weight_dtype)
    reward_model.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        class LocalTracker:
            def __init__(self, name="local"):
                self.name = name
            
            def log(self, data, step=None, **kwargs):
                pass

            def finish(self):
                pass

        
        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs={"wandb": {"name": args.output_dir.split('/')[-1],"mode": "online"}})
        
        accelerator.trackers.append(LocalTracker())

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.resume_from_checkpoint))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        loss_per_epoch = 0.
        pretrain_loss_per_epoch = 0.
        reward_loss_per_epoch = 0.
        dice_loss_per_epoch = 0.
        uncertainty_mse_per_epoch = 0.
        clip_loss_per_epoch = 0.
        cup_disc_loss_per_epoch = 0.
        cup_disc_difference_per_epoch = 0.

        train_loss, train_pretrain_loss, train_reward_loss = 0., 0., 0.

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]  # text condition
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)  # image condition
                origin_mask_proportion = batch["origin_mask_proportion"].to(dtype=weight_dtype)

                # This step is necessary. It took us a long time to find out this issue
                # The input of the canny/hed/lineart model does not require normalization of the image
                if args.conditioning_image_column == "canny":
                    low_threshold = 0.1 # low_threshold = random.uniform(0, 1)
                    high_threshold = 0.2 # high_threshold = random.uniform(low_threshold, 1)
                    with torch.no_grad():
                        # mean & std used in image transformations
                        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                        std = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                        # magnitude, edge
                        denormalized_condition_image = controlnet_image * std + mean
                        labels, controlnet_image = reward_model(denormalized_condition_image, low_threshold, high_threshold)
                        controlnet_image = controlnet_image.expand(-1, 3, -1, -1)  # (B, 3, H, W)
                elif args.conditioning_image_column in ['lineart', 'hed']:
                    with torch.no_grad():
                        # mean & std used in image transformations
                        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                        std = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                        denormalized_condition_image = controlnet_image * std + mean
                        labels = reward_model(denormalized_condition_image.to(weight_dtype))
                        controlnet_image = labels.expand(-1, 3, -1, -1)  # (B, 3, H, W)
                        controlnet_image = 1 - controlnet_image if args.task_name == 'lineart' else controlnet_image

                """
                Training ControlNet
                """
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(args.timestep_sampling_start, args.timestep_sampling_end, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                pretrain_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")





                """
                Rewarding ControlNet
                """
                # Predict the single-step denoised latents
                pred_original_sample = [
                    noise_scheduler.step(noise, t, noisy_latent).pred_original_sample.to(weight_dtype) \
                        for (noise, t, noisy_latent) in zip(model_pred, timesteps, noisy_latents)
                ]
                pred_original_sample = torch.stack(pred_original_sample)


                # Map the denoised latents into RGB images
                pred_original_sample = 1 / vae.config.scaling_factor * pred_original_sample
                image = vae.decode(pred_original_sample.to(weight_dtype)).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                


                # image normalization, depends on different reward models
                # This step is necessary. It took us a long time to find out this issue
                if args.task_name == 'depth':
                    image = torchvision.transforms.functional.resize(image, (384, 384))
                    image = normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                elif args.task_name in ['canny', 'lineart', 'hed']:
                    pass
                else:
                    image = normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

                # reward model inference
                if args.task_name == 'canny':
                    outputs = reward_model(image.to(accelerator.device), low_threshold, high_threshold)
                elif args.task_name == 'segmentation' and 'segman::' in args.reward_model_name_or_path:
                    reward_model.to(accelerator.device, dtype=torch.float32)
                    outputs = reward_model(image.to(accelerator.device, dtype=torch.float32))
                elif args.task_name == 'segmentation' and (args.reward_model_name_or_path.endswith('.pth') or args.reward_model_name_or_path.endswith('.pt')):
                    # TransUNet → generated_image → VAE.decode → UNet / ControlNet。

                    transunet_inputs = image.to(accelerator.device, dtype=torch.float32)

                    from torchvision.transforms.functional import rgb_to_grayscale
                    transunet_inputs = rgb_to_grayscale(transunet_inputs)

                    transunet_inputs = torchvision.transforms.functional.resize(transunet_inputs, (224, 224))

                    reward_model.to(accelerator.device, dtype=torch.float32)
                    out = reward_model(transunet_inputs)
                    outputs = torch.nn.functional.interpolate(
                        out,
                        size=(512, 512),
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    outputs = reward_model(image.to(accelerator.device))

                # normalize the predicted depth to (0, 1]
                if type(outputs) == transformers.modeling_outputs.DepthEstimatorOutput:

                    # map predicted depth into [0, 1]
                    outputs = outputs.predicted_depth
                    outputs = torchvision.transforms.functional.resize(outputs, (args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR)
                    max_values = outputs.view(args.train_batch_size, -1).amax(dim=1, keepdim=True).view(args.train_batch_size, 1, 1)
                    outputs = outputs / max_values

                    # map label into [0, 1]
                    labels = batch["labels"].mean(dim=1)  # (N, 3, H, W) -> (N, H, W)
                    max_values = labels.view(labels.size(0), -1).max(dim=1)[0]
                    labels = labels / max_values.view(-1, 1, 1)

                # kornia.filters.canny return a tuple with (magnitude, edge)
                elif args.task_name == 'canny':
                    outputs = outputs[0]   # (B, 1, H, W)
                elif args.task_name in ['lineart', 'hed']:
                    pass
                else:
                    labels = batch["labels"]

                # Avoid nan loss when using FP16 (happen in softmax)
                # FP32 and BF16 both work well
                if image.dtype == torch.float16:
                    if isinstance(outputs, torch.Tensor):
                        outputs = outputs.to(torch.float32)
                        labels = labels.to(torch.float32)
                    elif isinstance(outputs, list):
                        outputs = [x.to(torch.float32) for x in outputs]
                        labels = [x.to(torch.float32) for x in labels]


                # For depth and segmentation, we resize the label to the size of model output
                if args.task_name == 'segmentation':
                    labels = label_transform(labels, args.task_name, args.dataset_name, output_size=outputs.shape[-2:])
                elif args.task_name in ['depth', 'canny', 'lineart', 'hed']:
                    labels = label_transform(labels, args.task_name, args.dataset_name)
                else:
                    raise NotImplementedError(f"Not support task: {args.task_name}.")

                labels = [x.to(accelerator.device) for x in labels] if isinstance(labels, list) else labels.to(accelerator.device)

                loss_weights, timestep_mask = compute_loss_weights(timesteps, args, accelerator.device, noise_scheduler)

                # ============================================================
                #
                #
                # ============================================================
                _need_reward_forward = (args.reward_loss_weight != 0) or (args.uncertainty_loss_weight != 0)

                if _need_reward_forward:
                    reward_loss_result = get_reward_loss(
                        outputs,
                        labels,
                        args.dataset_name,
                        args.task_name,
                        use_uncertainty_weight=args.use_uncertainty_weight,
                        uncertainty_loss_weight=args.uncertainty_loss_weight,
                        reduction='none',
                        use_cross_entropy=getattr(args, 'reward_use_cross_entropy', False)
                    )

                    use_cross_entropy = getattr(args, 'reward_use_cross_entropy', False) and "fairseg" in args.dataset_name.lower()
                    if isinstance(reward_loss_result, tuple) and len(reward_loss_result) == 3:
                        _, dice_loss_component, uncertainty_mse_loss = reward_loss_result
                    else:
                        dice_loss_component = reward_loss_result
                        uncertainty_mse_loss = torch.tensor(0.0, device=accelerator.device, dtype=image.dtype)

                    dice_loss_component = dice_loss_component.reshape_as(loss_weights)
                    dice_loss_component = (loss_weights * dice_loss_component).sum() / (loss_weights.sum() + 1e-10)
                    dice_loss_component = args.reward_loss_weight * dice_loss_component

                    if isinstance(reward_loss_result, tuple):
                        uncertainty_mse_loss = uncertainty_mse_loss.reshape_as(loss_weights)
                        uncertainty_mse_loss = (loss_weights * uncertainty_mse_loss).sum() / (loss_weights.sum() + 1e-10)
                        uncertainty_mse_loss = args.uncertainty_loss_weight * uncertainty_mse_loss

                    reward_loss = dice_loss_component + uncertainty_mse_loss
                else:
                    reward_loss = torch.tensor(0.0, device=accelerator.device, dtype=image.dtype)
                    dice_loss_component = torch.tensor(0.0, device=accelerator.device, dtype=image.dtype)
                    uncertainty_mse_loss = torch.tensor(0.0, device=accelerator.device, dtype=image.dtype)

                # ============================================================
                # [REMOVED FOR PAPER] SAL (Semantic Alignment Loss, CLIP-based)
                # ============================================================
                total_clip_loss = torch.tensor(0.0, device=accelerator.device, dtype=image.dtype)
                # clip_loss_weight = getattr(args, 'clip_loss_weight_v1_1', 1)
                # if clip_loss_weight != 0 and timestep_mask.any():
                #     clip_loss_per_sample = calculate_clip_loss(image, batch["pixel_values"], batch["original_texts"], accelerator.device, clip_timestep_mask=timestep_mask, args=args)
                #     clip_loss_per_sample = clip_loss_per_sample.reshape(-1, 1)  # (B, 1)
                #     clip_loss_weighted = (loss_weights * clip_loss_per_sample).sum() / (loss_weights.sum() + 1e-10)
                #     total_clip_loss = clip_loss_weighted * clip_loss_weight
                # else:
                #     total_clip_loss = torch.tensor(0.0, device=accelerator.device, dtype=image.dtype)

                # ============================================================
                # vCDR consistency loss.
                # ============================================================
                cup_disc_loss_weight = getattr(args, 'cup_disc_loss_weight_v1_1', 1.0)
                if args.task_name == 'segmentation' and cup_disc_loss_weight != 0:
                    vcdr_alpha = getattr(args, 'vcdr_loss_alpha', 10.0)

                    cup_disc_differences, r_sigma_pred, r_sigma_target = \
                        calculate_cup_disc_ratio_difference_from_outputs(
                            outputs,
                            batch["labels"],
                            accelerator.device,
                            timestep_mask=timestep_mask,
                            return_all=True,
                            alpha=vcdr_alpha,
                        )
                    cup_disc_loss = cup_disc_differences.pow(2)
                    cup_disc_loss = cup_disc_loss.reshape_as(loss_weights)

                    m_float = timestep_mask.to(
                        device=accelerator.device, dtype=loss_weights.dtype
                    ).reshape_as(loss_weights)
                    effective_weight = loss_weights * m_float
                    denom = effective_weight.sum() + 1e-10
                    cup_disc_loss_weighted = (effective_weight * cup_disc_loss).sum() / denom
                    avg_cup_disc_loss = cup_disc_loss_weighted * cup_disc_loss_weight

                    abs_diff = cup_disc_differences.abs().reshape_as(loss_weights)
                    avg_cup_disc_difference = (
                        (m_float * abs_diff).sum() / (m_float.sum() + 1e-10)
                    )
                else:
                    avg_cup_disc_loss = torch.tensor(0.0, device=accelerator.device, dtype=image.dtype)
                    avg_cup_disc_difference = torch.tensor(0.0, device=accelerator.device, dtype=image.dtype)

                combined_loss = reward_loss + avg_cup_disc_loss
                loss = pretrain_loss + combined_loss * args.combined_loss_weight

                """
                Losses
                """
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_pretrain_loss = accelerator.gather(pretrain_loss.repeat(args.train_batch_size)).mean()
                avg_reward_loss = accelerator.gather(reward_loss.repeat(args.train_batch_size)).mean()
                avg_clip_loss = accelerator.gather(total_clip_loss.repeat(args.train_batch_size)).mean()
                avg_cup_disc_loss = accelerator.gather(avg_cup_disc_loss.repeat(args.train_batch_size)).mean()
                avg_cup_disc_difference = accelerator.gather(avg_cup_disc_difference.repeat(args.train_batch_size)).mean()
                avg_dice_loss = accelerator.gather(dice_loss_component.repeat(args.train_batch_size)).mean()
                avg_uncertainty_mse = accelerator.gather(uncertainty_mse_loss.repeat(args.train_batch_size)).mean()

                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                train_pretrain_loss += avg_pretrain_loss.item() / args.gradient_accumulation_steps
                train_reward_loss += avg_reward_loss.item() / args.gradient_accumulation_steps
                train_clip_loss = avg_clip_loss.item() / args.gradient_accumulation_steps
                train_cup_disc_loss = avg_cup_disc_loss.item() / args.gradient_accumulation_steps
                train_cup_disc_difference = avg_cup_disc_difference.item() / args.gradient_accumulation_steps
                train_dice_loss = avg_dice_loss.item() / args.gradient_accumulation_steps
                train_uncertainty_mse = avg_uncertainty_mse.item() / args.gradient_accumulation_steps

                # Back propagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_controlnet.step(controlnet.parameters())
                progress_bar.update(1)
                global_step += 1

                # loss when perform gradient backward
                accelerator.log({
                        "loss":      train_loss,
                        "diff_loss": train_pretrain_loss,
                        "ce_loss":   train_reward_loss,
                        "vcdr_loss": train_cup_disc_loss,
                        "vcdr_diff": train_cup_disc_difference,
                        "lr":        lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step
                )
                loss_per_epoch += train_loss
                pretrain_loss_per_epoch += train_pretrain_loss
                reward_loss_per_epoch += train_reward_loss
                dice_loss_per_epoch += train_dice_loss
                uncertainty_mse_per_epoch += train_uncertainty_mse
                clip_loss_per_epoch += train_clip_loss
                cup_disc_loss_per_epoch += train_cup_disc_loss
                cup_disc_difference_per_epoch += train_cup_disc_difference

                train_loss, train_pretrain_loss, train_reward_loss = 0., 0., 0.

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # directly save the state_dict
                        if accelerator.distributed_type != accelerate.DistributedType.FSDP:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)

                            if args.use_ema:
                                ema_controlnet.save_pretrained(f'{save_path}/controlnet_ema')

                            logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        if accelerator.is_main_process:
                            start_time = time.time()
                            image_logs = log_validation(
                                vae,
                                text_encoder,
                                tokenizer,
                                unet,
                                controlnet,
                                ema_controlnet,
                                args,
                                accelerator,
                                weight_dtype,
                                global_step,
                                val_dataset
                            )

                            end_time = time.time()
                            logger.info(f"Validation time: {end_time - start_time} seconds")

            # only show in the progress bar
            logs = {
                "loss_step": loss.detach().item(),
                "pretrain_loss_step": pretrain_loss.detach().item(),
                "reward_loss_step": reward_loss.detach().item(),
                "dice_loss_step": dice_loss_component.detach().item(),
                "uncertainty_mse_step": uncertainty_mse_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "train_cup_disc_loss": train_cup_disc_loss,
                "train_cup_disc_difference": train_cup_disc_difference
            }
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)

            # FSDP save model need to call all the ranks
            if global_step % args.checkpointing_steps == 0:
                if accelerator.distributed_type == accelerate.DistributedType.FSDP:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved accelerator state to {save_path}")

                    # Gather all of the state in the rank 0 device
                    accelerator.wait_for_everyone()
                    with FSDP.state_dict_type(controlnet, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                        state_dict = accelerator.get_state_dict(controlnet)

                    # Saving FSDP state
                    if accelerator.is_main_process:
                        torch.save(state_dict, os.path.join(save_path, 'controlnet_state_dict.pt'))
                        logger.info(f"Saved ControlNet state to {save_path}")

            if global_step >= args.max_train_steps:
                break

        logs = {
            "loss_epoch":      loss_per_epoch / len(train_dataloader),
            "diff_loss_epoch": pretrain_loss_per_epoch / len(train_dataloader),
            "ce_loss_epoch":   reward_loss_per_epoch / len(train_dataloader),
            "vcdr_loss_epoch": cup_disc_loss_per_epoch / len(train_dataloader),
            "vcdr_diff_epoch": cup_disc_difference_per_epoch / len(train_dataloader),
        }
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()

    # If we use FSDP, saving the state_dict
    if accelerator.distributed_type == accelerate.DistributedType.FSDP:
        with FSDP.state_dict_type(controlnet, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            state_dict = accelerator.get_state_dict(controlnet)
            ema_state_dict = accelerator.get_state_dict(ema_controlnet) if args.use_ema else None

        if accelerator.is_main_process:
            torch.save(state_dict, os.path.join(args.output_dir, 'controlnet_state_dict.pt'))
            if args.use_ema:
                torch.save(ema_state_dict, os.path.join(args.output_dir, 'controlnet_state_dict_ema.pt'))
            logger.info(f"Saved ControlNet state to {args.output_dir}")
    else:
        controlnet = accelerator.unwrap_model(controlnet)

        controlnet.save_pretrained(args.output_dir)
        if args.use_ema:
            ema_controlnet.save_pretrained(args.output_dir + '_ema')

    if accelerator.is_main_process:
        if args.push_to_hub:
            for _ in range(100):
                try:
                    save_model_card(
                        repo_id,
                        image_logs=image_logs,
                        base_model=args.pretrained_model_name_or_path,
                        repo_folder=args.output_dir,
                    )
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        path_in_repo=args.output_dir.replace('work_dirs/', ''),
                        commit_message=f"End of training {args.output_dir.split('/')[-1]}",
                        ignore_patterns=["step_*", "epoch_*"],
                        token=args.hub_token
                    )
                    break
                except:
                    continue

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)