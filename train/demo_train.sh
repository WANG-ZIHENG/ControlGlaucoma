#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/data/fairseg}
CKPT_ROOT=${CKPT_ROOT:-../checkpoints}
OUTPUT_DIR=${OUTPUT_DIR:-./runs/controlglaucoma_demo}

BASE_CONTROLNET=${BASE_CONTROLNET:-thibaud/controlnet-sd21-ade20k-diffusers}
SEGMAN_CKPT="${CKPT_ROOT}/segman_b/segman_b.pth"
SEGMAN_CFG="../SegMAN/segmentation/local_configs/segman/base/segman_b_ade.py"

python finetune_controlnet.py \
    --seed=42 \
    --prediction_type=v_prediction \
    --loss_weight_strategy=fixed_timestep \
    --max_timestep_rewarding=800 \
    --reward_loss_weight=1.0 \
    --cup_disc_loss_weight_v1_1=1.0 \
    --vcdr_loss_alpha=10.0 \
    --reward_use_cross_entropy \
    --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1 \
    --controlnet_model_name_or_path="${BASE_CONTROLNET}" \
    --reward_model_name_or_path="segman::${SEGMAN_CFG}::${SEGMAN_CKPT}" \
    --output_dir="${OUTPUT_DIR}" \
    --task_name=segmentation \
    --dataset_name="${DATA_ROOT}" \
    --use_filter \
    --resolution=512 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-5 \
    --mixed_precision=fp16 \
    --gradient_checkpointing \
    --dataloader_num_workers=8 \
    --max_train_steps=8000 \
    --lr_scheduler=constant_with_warmup \
    --lr_warmup_steps=30 \
    --grad_scale=0.5 \
    --use_ema \
    --validation_steps=500 \
    --checkpointing_steps=2000 \
    --image_column=pixel_values \
    --conditioning_image_column=conditioning_pixel_values \
    --caption_column=prompt
