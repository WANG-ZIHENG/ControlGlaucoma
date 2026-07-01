#!/usr/bin/env python
# coding=utf-8

import os
import argparse
import logging
import uuid
import csv
import shutil
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms.functional import normalize
from torchvision import transforms
import torch.nn.functional as F
from monai.losses import DiceLoss

from accelerate import PartialState
PartialState()

from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)

from finetune_controlnet import (
    FairSegDataset,
    load_fairseg_dataset,
    import_model_class_from_model_name_or_path,
)

from utils import get_reward_model

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def extract_model_info_from_path(controlnet_path):
    import re
    path = os.path.normpath(controlnet_path)
    
    pattern = r'.*[/\\]([^/\\]+)[/\\](checkpoint-\d+)[/\\].*'
    match = re.search(pattern, path)
    
    if match:
        parent_dir = match.group(1)
        checkpoint_name = match.group(2)
        
        v_pred_match = re.search(r'v_prediction_([a-zA-Z0-9]+)', parent_dir)
        if v_pred_match:
            model_name_part = f"v_prediction_{v_pred_match.group(1)}"
            return f"{model_name_part}_{checkpoint_name}"
        
        prefix_match = re.match(r'^\d+_(.+)', parent_dir)
        if prefix_match:
            model_name_part = prefix_match.group(1)
            return f"{model_name_part}_{checkpoint_name}"
        
        return f"{parent_dir}_{checkpoint_name}"
    
    path_parts = [p for p in path.split(os.sep) if p]
    exclude_pattern = re.compile(r'^(controlnet_ema|controlnet)$', re.IGNORECASE)
    
    for part in reversed(path_parts):
        if part and not exclude_pattern.match(part):
            return part
    
    return "model"


def parse_args():
    parser = argparse.ArgumentParser(description="ControlGlaucoma image generation script.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='stabilityai/stable-diffusion-2-1',
        help="Path or HuggingFace id of the Stable Diffusion backbone.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default="./checkpoints/controlnet_vpred",
        help="Path to the trained ControlNet weights.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional model revision.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="./data/fairseg",
        help="Dataset root (Harvard-FairSeg by default).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split.",
    )
    parser.add_argument(
        "--use_filter",
        action="store_true",
        default=False,
        help="Restrict the sample list to filenames listed in <dataset_name>/filter_file.txt.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_images",
        help="Directory for the generated images.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=35,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images generated per prompt.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100000,
        help="Upper bound on the number of samples to process.",
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode.",
    )
    parser.add_argument(
        "--enable_xformers",
        action="store_true",
        help="Enable xformers memory-efficient attention.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device (cuda/cpu).",
    )
    parser.add_argument(
        "--enable_random_disc_replacement",
        action="store_true",
        default=False,
        help="Experimental: randomly swap the disc region across samples during generation.",
    )
    parser.add_argument(
        "--gen_scale_masks",
        action="store_true",
        default=False,
        help="Experimental: also generate images for scaled variants of each mask.",
    )
    parser.add_argument(
        "--reward_model",
        type=str,
        default="segman::../SegMAN/segmentation/local_configs/segman/base/segman_b_ade.py::../checkpoints/segman_b/segman_b.pth",
        help="Frozen segmentation model used to compute Dice and mIoU.",
    )
    parser.add_argument(
        "--calculate_metrics",
        action="store_true",
        default=True,
        help="Compute Dice and mIoU per generated image.",
    )
    parser.add_argument(
        "--no-calculate_metrics",
        dest="calculate_metrics",
        action="store_false",
        help="Skip the Dice/mIoU computation.",
    )
    args = parser.parse_args()
    return args


def load_pipeline(args, device):
    logger.info("Loading model ...")

    if args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
    else:
        raise ValueError("pretrained_model_name_or_path is required")
    
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision
    )
    
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)
    
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
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    if args.enable_xformers:
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory-efficient attention enabled")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")

    logger.info("Model loaded.")
    return pipeline


def calculate_dice_miou(pred_mask, target_mask, num_classes=3):
    if target_mask.shape != pred_mask.shape:
        target_mask = torch.nn.functional.interpolate(
            target_mask.unsqueeze(0).unsqueeze(0).float(),
            size=pred_mask.shape,
            mode='nearest'
        ).squeeze().long()
    
    num_classes = int(max(int(pred_mask.max().item()), int(target_mask.max().item())) + 1)
    num_classes = max(num_classes, 3)
    
    pred_oh = F.one_hot(pred_mask, num_classes=num_classes).permute(2, 0, 1).unsqueeze(0).float()
    target_oh = F.one_hot(target_mask, num_classes=num_classes).permute(2, 0, 1).unsqueeze(0).float()
    
    dice_loss_fn = DiceLoss(include_background=False, reduction='mean', softmax=False, sigmoid=False)
    dice_value = float(1.0 - dice_loss_fn(pred_oh, target_oh).item())
    
    ious = []
    for cls in range(1, num_classes):
        pred_cls = (pred_mask == cls)
        target_cls = (target_mask == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union > 0:
            iou = intersection / union
            ious.append(iou.item())
    
    if len(ious) > 0:
        miou_value = float(np.mean(ious))
    else:
        miou_value = 0.0
    
    return dice_value, miou_value


def generate_images(args, pipeline, dataset, device, reward_model=None):
    logger.info(f"Start generating, dataset size: {len(dataset)}")

    sample_masks_cache = {}

    # Output directory selection
    output_dir_replacement = None
    output_dir_scale = None
    
    if args.enable_random_disc_replacement:
        output_dir_replacement = os.path.join(args.output_dir, "replacement")
        os.makedirs(output_dir_replacement, exist_ok=True)
        logger.info(f"Replacement-mode images will be saved to: {output_dir_replacement}")
    
    if args.gen_scale_masks:
        output_dir_scale = os.path.join(args.output_dir, "scale")
        os.makedirs(output_dir_scale, exist_ok=True)
        logger.info(f"Scale-mode images will be saved to: {output_dir_scale}")
    
    if not args.enable_random_disc_replacement and not args.gen_scale_masks:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Images will be saved to: {args.output_dir}")
    
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None
    
    num_samples = len(dataset)
    if args.max_samples is not None:
        num_samples = min(num_samples, args.max_samples)
    
    all_results = []
    
    for idx in tqdm(range(num_samples), desc="Generating"):
        try:
            sample = dataset[idx]
            
            prompt = sample['prompt']
            sample_name = sample.get('name', f'sample_{idx}')
            
            conditioning_pixel_values = sample['conditioning_pixel_values']  # Shape: [3, H, W], range [0, 1]

            current_disc_cup_mask = None
            if 'labels' in sample:
                current_disc_cup_mask = extract_disc_cup_mask_from_labels(sample['labels'])
                if current_disc_cup_mask is not None:
                    sample_masks_cache[sample_name] = current_disc_cup_mask.copy()
            
            if isinstance(conditioning_pixel_values, torch.Tensor):
                conditioning_pixel_values_original = conditioning_pixel_values.cpu().numpy().copy()
                conditioning_pixel_values = conditioning_pixel_values.cpu().numpy()
            elif isinstance(conditioning_pixel_values, np.ndarray):
                conditioning_pixel_values_original = conditioning_pixel_values.copy()
            else:
                raise ValueError(f"Unsupported conditioning_pixel_values type: {type(conditioning_pixel_values)}")
            
            if conditioning_pixel_values.max() <= 1.0:
                conditioning_pixel_values = (conditioning_pixel_values * 255.0).astype(np.uint8)
            else:
                conditioning_pixel_values = conditioning_pixel_values.astype(np.uint8)
            
            if conditioning_pixel_values.shape[0] == 3:
                conditioning_pixel_values = np.transpose(conditioning_pixel_values, (1, 2, 0))
            
            conditioning_image = Image.fromarray(conditioning_pixel_values)
            
            conditioning_image = conditioning_image.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
            
            with torch.autocast("cuda" if device.type == "cuda" else "cpu"):
                images = pipeline(
                    [prompt] * args.num_images_per_prompt,
                    [conditioning_image] * args.num_images_per_prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).images
            
            if args.enable_random_disc_replacement and output_dir_replacement is not None:
                current_output_dir = output_dir_replacement
            else:
                current_output_dir = args.output_dir
            
            first_uuid_4digit = None
            for img_idx, image in enumerate(images):
                uuid_4digit = str(uuid.uuid4())[:4]
                if img_idx == 0:
                    first_uuid_4digit = uuid_4digit
                if args.num_images_per_prompt == 1:
                    save_path = os.path.join(current_output_dir, f"{sample_name}_{uuid_4digit}_generate.png")
                else:
                    save_path = os.path.join(current_output_dir, f"{sample_name}_{uuid_4digit}_{img_idx}_generate.png")
                image.save(save_path)
                
                if args.calculate_metrics and reward_model is not None and 'labels' in sample:
                    try:
                        transform = transforms.Compose([
                            transforms.Resize((512, 512)),
                            transforms.ToTensor(),
                        ])
                        img_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, H, W), range [0, 1]
                        
                        img_norm = normalize(img_tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        
                        with torch.no_grad():
                            reward_model.eval()
                            seg_outputs = reward_model(img_norm.float())  # (1, num_classes, H, W)
                        
                        pred_mask = torch.argmax(seg_outputs, dim=1)[0]  # (H, W)
                        
                        target_np = sample['labels']  # (H, W), values in {0,1,2}
                        target_mask = torch.tensor(target_np, device=device, dtype=torch.long)
                        
                        dice_value, miou_value = calculate_dice_miou(pred_mask, target_mask, num_classes=3)
                        
                        result_item = {
                            'sample_name': sample_name,
                            'image_path': save_path,
                            'uuid': uuid_4digit,
                            'image_idx': img_idx,
                            'dice': dice_value,
                            'miou': miou_value,
                            'mode': 'normal' if not args.enable_random_disc_replacement else 'replacement',
                            'cup_disc_ratio': sample.get('origin_mask_proportion'),
                            'disc_cup_mask': current_disc_cup_mask.copy() if current_disc_cup_mask is not None else None
                        }
                        all_results.append(result_item)
                        
                    except Exception as e:
                        logger.warning(f"Metric computation failed for sample {sample_name} image {img_idx}: {e}")
                        import traceback
                        traceback.print_exc()
            

            conditioning_save_path = os.path.join(current_output_dir, f"{sample_name}.png")
            conditioning_image.save(conditioning_save_path)
            


            if 'pixel_values' in sample:
                pixel_values = sample['pixel_values']  # Shape: [3, H, W], range [-1, 1]
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.cpu().numpy()
                elif isinstance(pixel_values, np.ndarray):
                    pass
                else:
                    logger.warning(f"Unsupported pixel_values type: {type(pixel_values)}")
                    continue

                # Denormalize from [-1, 1] to [0, 255]
                pixel_values = ((pixel_values + 1.0) * 127.5).astype(np.uint8)

                if pixel_values.shape[0] == 3:
                    pixel_values = np.transpose(pixel_values, (1, 2, 0))

                original_image = Image.fromarray(pixel_values)
                original_image = original_image.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
                original_save_path = os.path.join(current_output_dir, f"{sample_name}_original.png")
                original_image.save(original_save_path)

            prompt_save_path = os.path.join(current_output_dir, f"{sample_name}_prompt.txt")
            with open(prompt_save_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            if args.gen_scale_masks and output_dir_scale is not None and 'extra_mask' in sample and len(sample.get('extra_mask', [])) > 0:
                scaled_masks = sample['extra_mask']  # List of numpy arrays, shape [H, W, 3], range [0, 1]
                relativs = sample.get('relativs', [])  # List of Cup-to-Disc Ratio values
                prompts = sample.get('prompts', [])  # List of prompts for each scaled mask
                
                logger.info(f"Generating {len(scaled_masks)} scaled-mask images for sample {sample_name}")
                
                original_prompt_save_path = os.path.join(output_dir_scale, f"{sample_name}_original_prompt.txt")
                with open(original_prompt_save_path, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                
                if 'pixel_values' in sample:
                    pixel_values = sample['pixel_values']  # Shape: [3, H, W], range [-1, 1]
                    if isinstance(pixel_values, torch.Tensor):
                        pixel_values = pixel_values.cpu().numpy()
                    elif isinstance(pixel_values, np.ndarray):
                        pass
                    else:
                        logger.warning(f"Unsupported pixel_values type: {type(pixel_values)}")
                        pixel_values = None
                    
                    if pixel_values is not None:
                        # Denormalize from [-1, 1] to [0, 255]
                        pixel_values = ((pixel_values + 1.0) * 127.5).astype(np.uint8)
                        
                        if pixel_values.shape[0] == 3:
                            pixel_values = np.transpose(pixel_values, (1, 2, 0))
                        
                        original_image = Image.fromarray(pixel_values)
                        original_image = original_image.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
                        original_save_path_scale = os.path.join(output_dir_scale, f"{sample_name}_original.png")
                        original_image.save(original_save_path_scale)
                
                for mask_idx, (scaled_mask, ratio, scale_prompt) in enumerate(zip(scaled_masks, relativs, prompts)):
                    try:
                        if isinstance(scaled_mask, torch.Tensor):
                            scaled_mask_np = scaled_mask.cpu().numpy()
                        elif isinstance(scaled_mask, np.ndarray):
                            scaled_mask_np = scaled_mask.copy()
                        else:
                            logger.warning(f"Unsupported scaled_mask type: {type(scaled_mask)}")
                            continue
                        
                        if scaled_mask_np.max() <= 1.0:
                            scaled_mask_np = (scaled_mask_np * 255.0).astype(np.uint8)
                        else:
                            scaled_mask_np = scaled_mask_np.astype(np.uint8)
                        
                        if len(scaled_mask_np.shape) == 3:
                            if scaled_mask_np.shape[0] == 3 and scaled_mask_np.shape[0] < scaled_mask_np.shape[1]:
                                # [C, H, W] -> [H, W, C]
                                scaled_mask_np = np.transpose(scaled_mask_np, (1, 2, 0))
                        
                        scaled_mask_np = np.clip(scaled_mask_np, 0, 255).astype(np.uint8)
                        
                        scaled_conditioning_image = Image.fromarray(scaled_mask_np)
                        scaled_conditioning_image = scaled_conditioning_image.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
                        
                        scaled_conditioning_save_path = os.path.join(
                            output_dir_scale, 
                            f"{sample_name}_scaled_ratio_{ratio:.3f}.png"
                        )
                        scaled_conditioning_image.save(scaled_conditioning_save_path)
                        
                        with torch.autocast("cuda" if device.type == "cuda" else "cpu"):
                            scaled_images = pipeline(
                                [scale_prompt] * args.num_images_per_prompt,
                                [scaled_conditioning_image] * args.num_images_per_prompt,
                                num_inference_steps=args.num_inference_steps,
                                guidance_scale=args.guidance_scale,
                                generator=generator,
                            ).images
                        
                        if first_uuid_4digit is None:
                            first_uuid_4digit = str(uuid.uuid4())[:4]
                        
                        for img_idx, scaled_image in enumerate(scaled_images):
                            if args.num_images_per_prompt == 1:
                                scaled_save_path = os.path.join(
                                    output_dir_scale, 
                                    f"{sample_name}_{first_uuid_4digit}_extra_{ratio:.2f}_generate.png"
                                )
                            else:
                                scaled_save_path = os.path.join(
                                    output_dir_scale, 
                                    f"{sample_name}_{first_uuid_4digit}_extra_{ratio:.2f}_{img_idx}_generate.png"
                                )
                            scaled_image.save(scaled_save_path)
                            
                            if args.calculate_metrics and reward_model is not None and 'labels' in sample:
                                try:
                                    transform = transforms.Compose([
                                        transforms.Resize((512, 512)),
                                        transforms.ToTensor(),
                                    ])
                                    img_tensor = transform(scaled_image).unsqueeze(0).to(device)  # (1, 3, H, W), range [0, 1]
                                    
                                    img_norm = normalize(img_tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    
                                    with torch.no_grad():
                                        reward_model.eval()
                                        seg_outputs = reward_model(img_norm.float())  # (1, num_classes, H, W)
                                    
                                    pred_mask = torch.argmax(seg_outputs, dim=1)[0]  # (H, W)
                                    
                                    target_np = sample['labels']  # (H, W), values in {0,1,2}
                                    target_mask = torch.tensor(target_np, device=device, dtype=torch.long)
                                    
                                    dice_value, miou_value = calculate_dice_miou(pred_mask, target_mask, num_classes=3)
                                    
                                    result_item = {
                                        'sample_name': sample_name,
                                        'image_path': scaled_save_path,
                                        'uuid': first_uuid_4digit,
                                        'image_idx': img_idx,
                                        'dice': dice_value,
                                        'miou': miou_value,
                                        'mode': 'scale',
                                        'scale_ratio': ratio,
                                        'cup_disc_ratio': ratio,
                                        'disc_cup_mask': current_disc_cup_mask.copy() if current_disc_cup_mask is not None else None
                                    }
                                    all_results.append(result_item)
                                    
                                except Exception as e:
                                    logger.warning(f"Metric computation failed for sample {sample_name} scaled image (ratio={ratio:.2f}, idx={img_idx}): {e}")
                                    import traceback
                                    traceback.print_exc()
                        
                        # scaled_prompt_save_path = os.path.join(
                        #     output_dir_scale,
                        #     f"{sample_name}_scaled_prompt.txt"
                        # )
                        # with open(scaled_prompt_save_path, 'w', encoding='utf-8') as f:
                        #     f.write(scale_prompt)
                    
                    except Exception as e:
                        logger.error(f"Error processing scaled mask {mask_idx} for sample {sample_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
        
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if args.enable_random_disc_replacement and output_dir_replacement is not None:
        logger.info(f"Replacement-mode generation done. Saved to: {output_dir_replacement}")
    if args.gen_scale_masks and output_dir_scale is not None:
        logger.info(f"Scale-mode generation done. Saved to: {output_dir_scale}")
    if not args.enable_random_disc_replacement and not args.gen_scale_masks:
        logger.info(f"Generation done. Saved to: {args.output_dir}")
    
    if args.calculate_metrics and len(all_results) > 0:
        csv_output_dir = args.output_dir
        if args.enable_random_disc_replacement and output_dir_replacement is not None:
            csv_output_dir = output_dir_replacement
        elif args.gen_scale_masks and output_dir_scale is not None:
            csv_output_dir = output_dir_scale
        
        csv_file = os.path.join(csv_output_dir, "generation_metrics.csv")
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['sample_name', 'image_path', 'uuid', 'image_idx', 'mode', 'dice', 'miou']
            if any('scale_ratio' in r.keys() for r in all_results):
                fieldnames.insert(-2, 'scale_ratio')
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in all_results:
                row = {k: v for k, v in result.items() if k in fieldnames}
                writer.writerow(row)
        
        logger.info(f"Metrics written to: {csv_file}")
        logger.info(f"   {len(all_results)} rows")
        
        if len(all_results) > 0:
            dices = [r['dice'] for r in all_results]
            mious = [r['miou'] for r in all_results]
            logger.info(f"   Dice: mean={np.mean(dices):.4f}, std={np.std(dices):.4f}, min={np.min(dices):.4f}, max={np.max(dices):.4f}")
            logger.info(f"   mIoU: mean={np.mean(mious):.4f}, std={np.std(mious):.4f}, min={np.min(mious):.4f}, max={np.max(mious):.4f}")


def main():
    args = parse_args()

    model_info = extract_model_info_from_path(args.controlnet_model_name_or_path)
    base_output_dir = args.output_dir
    args.output_dir = os.path.join(base_output_dir, model_info)
    logger.info(f"Model tag extracted from path: {model_info}")
    logger.info(f"Output dir updated to: {args.output_dir}")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("Loading dataset ...")
    class DatasetArgs:
        def __init__(self):
            self.dataset_name = args.dataset_name
            self.use_filter = args.use_filter
            self.enable_random_disc_replacement = args.enable_random_disc_replacement
    
    dataset_args = DatasetArgs()
    
    from finetune_controlnet import FairSegDataset

    dataset = FairSegDataset(
        dataset_args, 
        split=args.split, 
        gen_data=True,
        gen_scale_masks=args.gen_scale_masks,
        tokenizer=None
    )
    logger.info(f"Dataset loaded, size: {len(dataset)}")
    if args.gen_scale_masks:
        logger.info("gen_scale_masks enabled: will generate one image per scaled mask")
    
    pipeline = load_pipeline(args, device)
    
    reward_model = None
    if args.calculate_metrics:
        logger.info("Loading segmentation model ...")
        reward_model = get_reward_model('segmentation', args.reward_model, device=device).eval()
        logger.info("Segmentation model loaded.")
    
    generate_images(args, pipeline, dataset, device, reward_model=reward_model)
    
    logger.info("All tasks done.")


if __name__ == "__main__":
    main()

