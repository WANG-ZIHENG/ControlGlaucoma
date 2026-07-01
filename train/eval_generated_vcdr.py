#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    calculate_cup_disc_ratio_from_mask,
    check_cup_inside_disc,
)


# --------------------------------------------------------
# --------------------------------------------------------
def build_pipeline(checkpoint_dir, sd_model="stabilityai/stable-diffusion-2-1",
                   prediction_type="v_prediction", device="cuda"):
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        DDIMScheduler,
    )

    weight_dtype = torch.float16
    print(f"[build_pipeline] loading ControlNet from {checkpoint_dir}")
    controlnet = ControlNetModel.from_pretrained(
        checkpoint_dir, torch_dtype=weight_dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model, controlnet=controlnet,
        torch_dtype=weight_dtype, safety_checker=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, prediction_type=prediction_type
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


# --------------------------------------------------------
# --------------------------------------------------------
def build_segman(cfg_path, pth_path, device="cuda"):
    from mmseg.apis import init_segmentor
    print(f"[build_segman] loading SegMAN from {pth_path}")
    model = init_segmentor(cfg_path, pth_path, device=device)
    model.eval()
    return model


def segment_image(seg_model, rgb_image_np, device="cuda"):
    import mmcv
    from mmseg.apis import inference_segmentor
    # inference_segmentor expects BGR image
    bgr = rgb_image_np[:, :, ::-1].copy()
    result = inference_segmentor(seg_model, bgr)
    pred = result[0]  # (H, W), int labels 0/1/2
    mask = np.zeros_like(pred, dtype=np.int32)
    mask[pred == 1] = -1
    mask[pred == 2] = -2
    return mask


# --------------------------------------------------------
# --------------------------------------------------------
def load_gt_mask(npz_path):
    raw = np.load(npz_path, allow_pickle=True)
    mask = raw["disc_cup_mask"].astype(np.int32)
    if "conditioning_pixel_values" in raw.files:
        cond = raw["conditioning_pixel_values"]
    else:
        cond = np.zeros((*mask.shape, 3), dtype=np.uint8)
        cond[mask == -1] = (0, 0, 255)   # disc rim = blue
        cond[mask == -2] = (255, 0, 0)   # cup = red
    prompt = str(raw["prompt"]) if "prompt" in raw.files else "a color fundus photograph"
    return mask, cond, prompt


# --------------------------------------------------------
# --------------------------------------------------------
@torch.no_grad()
def evaluate(
    checkpoint_dir, data_dir, val_list_file,
    sd_model, prediction_type,
    reward_cfg, reward_pth,
    output_json, num_inference_steps=30, resolution=512,
    device="cuda",
):
    from PIL import Image

    with open(val_list_file) as f:
        val_stems = {ln.strip() for ln in f if ln.strip()}
    print(f"[eval] {len(val_stems)} validation stems")

    all_npz = sorted(glob(os.path.join(data_dir, "*.npz")))
    val_npz = [p for p in all_npz if Path(p).stem in val_stems]
    print(f"[eval] matched {len(val_npz)} npz files")

    pipe = build_pipeline(checkpoint_dir, sd_model, prediction_type, device)
    seg = build_segman(reward_cfg, reward_pth, device)

    records = []
    t0 = time.time()
    for i, npz_path in enumerate(val_npz):
        stem = Path(npz_path).stem
        gt_mask, cond_np, prompt = load_gt_mask(npz_path)

        r_gt = calculate_cup_disc_ratio_from_mask(gt_mask)
        if r_gt <= 0:
            continue

        cond_pil = Image.fromarray(cond_np) if cond_np.dtype == np.uint8 else \
                   Image.fromarray((cond_np * 255).astype(np.uint8))
        gen_image = pipe(
            prompt,
            image=cond_pil,
            num_inference_steps=num_inference_steps,
            height=resolution, width=resolution,
            generator=torch.Generator(device=device).manual_seed(42),
        ).images[0]
        gen_np = np.array(gen_image)

        pred_mask = segment_image(seg, gen_np, device)

        # Cup-inside-disc validity
        cup_valid = check_cup_inside_disc(pred_mask)

        # Generated vCDR
        if cup_valid:
            r_gen = calculate_cup_disc_ratio_from_mask(pred_mask)
        else:
            r_gen = 0.0

        records.append({
            "stem": stem,
            "r_gt": float(r_gt),
            "r_gen": float(r_gen),
            "cup_valid": bool(cup_valid),
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(val_npz) - i - 1) / rate
            print(f"  [{i+1}/{len(val_npz)}] rate={rate:.2f} imgs/s, ETA={eta:.0f}s")

    r_gt_arr = np.array([r["r_gt"] for r in records])
    r_gen_arr = np.array([r["r_gen"] for r in records])
    n = len(r_gt_arr)
    cup_valid_rate = float(np.mean([r["cup_valid"] for r in records]))

    mae = float(np.mean(np.abs(r_gen_arr - r_gt_arr)))
    rmse = float(np.sqrt(np.mean((r_gen_arr - r_gt_arr) ** 2)))
    bias = float(np.mean(r_gen_arr - r_gt_arr))

    pearson_r, pearson_p = stats.pearsonr(r_gt_arr, r_gen_arr)
    spearman_rho, spearman_p = stats.spearmanr(r_gt_arr, r_gen_arr)

    # ICC(3,1)
    stacked = np.stack([r_gt_arr, r_gen_arr], axis=0)
    mean_per_sample = stacked.mean(axis=0)
    mean_per_rater = stacked.mean(axis=1)
    grand_mean = stacked.mean()
    ss_total = ((stacked - grand_mean) ** 2).sum()
    ss_between_samples = 2 * ((mean_per_sample - grand_mean) ** 2).sum()
    ss_between_raters = n * ((mean_per_rater - grand_mean) ** 2).sum()
    ss_residual = ss_total - ss_between_samples - ss_between_raters
    ms_between_samples = ss_between_samples / max(n - 1, 1)
    ms_residual = ss_residual / max(n - 1, 1)
    icc = (ms_between_samples - ms_residual) / (ms_between_samples + ms_residual) if (ms_between_samples + ms_residual) > 0 else float("nan")

    # Bland-Altman
    diff = r_gen_arr - r_gt_arr
    ba_mean = float(diff.mean())
    ba_sd = float(diff.std(ddof=1))
    loa_up = ba_mean + 1.96 * ba_sd
    loa_lo = ba_mean - 1.96 * ba_sd

    summary = {
        "checkpoint": checkpoint_dir,
        "n_samples": n,
        "cup_valid_rate": cup_valid_rate,
        "vCDR_MAE": round(mae, 4),
        "vCDR_RMSE": round(rmse, 4),
        "vCDR_bias": round(bias, 4),
        "Pearson_r": round(float(pearson_r), 4),
        "Spearman_rho": round(float(spearman_rho), 4),
        "ICC_3_1": round(float(icc), 4),
        "Bland_Altman_mean": round(ba_mean, 4),
        "Bland_Altman_SD": round(ba_sd, 4),
        "LoA_upper_95": round(float(loa_up), 4),
        "LoA_lower_95": round(float(loa_lo), 4),
    }

    out = {"summary": summary, "per_sample": records}
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] saved → {output_json}")

    print("\n===== Summary =====")
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")
    print()
    return summary


# --------------------------------------------------------
# --------------------------------------------------------
def batch_evaluate(args):
    run_dirs = sorted(glob(os.path.join(args.root_dir, args.pattern)))
    print(f"[batch] found {len(run_dirs)} candidate run dirs")

    all_summaries = []
    for run in run_dirs:
        ckpts = sorted(glob(os.path.join(run, "checkpoint-*")))
        if not ckpts:
            ckpts = [os.path.join(run, "controlnet")]
        if not ckpts:
            print(f"[batch] SKIP {run}: no checkpoint found")
            continue

        ckpt = ckpts[-1]
        if os.path.isdir(os.path.join(ckpt, "controlnet")):
            ckpt = os.path.join(ckpt, "controlnet")

        run_name = os.path.basename(run)
        out_json = os.path.join(args.output_dir, f"eval_{run_name}.json")
        if os.path.isfile(out_json) and not args.overwrite:
            print(f"[batch] SKIP {run_name} (already evaluated)")
            with open(out_json) as f:
                all_summaries.append(json.load(f)["summary"])
            continue

        print(f"\n[batch] evaluating {run_name} → {ckpt}")
        try:
            summary = evaluate(
                checkpoint_dir=ckpt,
                data_dir=args.data_dir,
                val_list_file=args.val_list,
                sd_model=args.sd_model,
                prediction_type=args.prediction_type,
                reward_cfg=args.reward_cfg,
                reward_pth=args.reward_pth,
                output_json=out_json,
                num_inference_steps=args.num_inference_steps,
            )
            summary["run_name"] = run_name
            all_summaries.append(summary)
        except Exception as e:
            print(f"[batch] FAIL {run_name}: {e}")
            import traceback; traceback.print_exc()

    if all_summaries:
        csv_path = os.path.join(args.output_dir, "all_runs_summary.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            w.writeheader()
            for s in all_summaries:
                w.writerow(s)
        print(f"\n[batch] summary CSV → {csv_path}")


# --------------------------------------------------------
# main
# --------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", action="store_true")
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--root_dir", type=str, default=None)
    p.add_argument("--pattern", type=str, default="*seed*")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--val_list", type=str, required=True)
    p.add_argument("--sd_model", type=str, default="stabilityai/stable-diffusion-2-1")
    p.add_argument("--prediction_type", type=str, default="v_prediction",
                   choices=["v_prediction", "epsilon"])
    p.add_argument("--reward_cfg", type=str,
        default="../SegMAN/segmentation/local_configs/segman/base/segman_b_ade.py")
    p.add_argument("--reward_pth", type=str,
        default="../checkpoints/segman_b/segman_b.pth")
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./eval_results")
    args = p.parse_args()

    if args.batch:
        assert args.root_dir, "--root_dir required in batch mode"
        os.makedirs(args.output_dir, exist_ok=True)
        batch_evaluate(args)
    else:
        assert args.checkpoint_dir and args.output_json, \
            "In single mode, provide --checkpoint_dir and --output_json"
        evaluate(
            checkpoint_dir=args.checkpoint_dir,
            data_dir=args.data_dir,
            val_list_file=args.val_list,
            sd_model=args.sd_model,
            prediction_type=args.prediction_type,
            reward_cfg=args.reward_cfg,
            reward_pth=args.reward_pth,
            output_json=args.output_json,
            num_inference_steps=args.num_inference_steps,
        )


if __name__ == "__main__":
    main()
