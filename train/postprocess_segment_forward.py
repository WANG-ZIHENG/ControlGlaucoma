from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from utils import get_reward_model, calculate_cup_disc_ratio_from_mask


VARIANT_PREFIXES_DEFAULT = [
    "A1_vPred_preTrained",
    "A2_vPred_FineTuned",
    "B1_Eps_preTrained",
    "B2_Eps_FineTuned",
]


def load_segman(cfg_path, ckpt_path, device):
    print(f"[load] SegMAN cfg={cfg_path}")
    print(f"[load] SegMAN ckpt={ckpt_path}")
    reward_model_path = f"segman::{cfg_path}::{ckpt_path}"
    rm = get_reward_model("segmentation", reward_model_path, device=str(device))
    rm.to(device).eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    return rm


@torch.no_grad()
def segman_predict_png(reward_model, png_path, device):
    """Read a PNG, apply ImageNet norm, run SegMAN, return argmax class-id (H, W)."""
    img = Image.open(png_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0      # (H, W, 3) in [0,1]
    t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (t - mean) / std
    logits = reward_model(x)
    return logits.argmax(dim=1)[0].cpu().numpy()        # (H, W) int


def classid_to_rgb(pm):
    """Class-id mask {0=bg, 1=rim, 2=cup} → RGB png (red=cup, blue=rim)."""
    rgb = np.zeros((pm.shape[0], pm.shape[1], 3), dtype=np.uint8)
    rgb[pm == 1] = (0, 0, 255)      # rim = blue
    rgb[pm == 2] = (255, 0, 0)      # cup = red
    return rgb


def dice_binary(pred_mask, gt_mask, cls, eps=1e-6):
    p = (pred_mask == cls); g = (gt_mask == cls)
    inter = int((p & g).sum())
    denom = int(p.sum()) + int(g.sum())
    if denom == 0:
        return 1.0
    return (2.0 * inter + eps) / (denom + eps)


def vcdr_from_classid(pm):
    """Fu-ellipse vCDR on class-id mask {0,1,2}; returns float."""
    signed = np.zeros_like(pm, dtype=np.float32)
    signed[pm == 1] = -1.0
    signed[pm == 2] = -2.0
    try:
        return float(calculate_cup_disc_ratio_from_mask(signed))
    except Exception:
        return 0.0


def load_gt_mask(sample_dir: Path):
    """Try to locate GT class-id mask from saved input conditioning or npz.
    Returns (gt_mask_2d, source_desc) or (None, None) if not found."""
    cond = sample_dir / "00_input_conditioning.png"
    if cond.exists():
        arr = np.array(Image.open(cond).convert("RGB"))
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        gt = np.zeros(arr.shape[:2], dtype=np.int64)
        gt[(r > 128) & (g < 96) & (b < 96)] = 2   # red → cup
        gt[(r < 96)  & (g < 96) & (b > 128)] = 1  # blue → rim
        if (gt == 2).sum() > 0 or (gt == 1).sum() > 0:
            return gt, str(cond)
    return None, None


def process_variant(sample_dir: Path, prefix: str, reward_model, device,
                    gt_mask=None, update_csv=True):
    fwd_dir = sample_dir / f"{prefix}__forward"
    out_dir = sample_dir / f"{prefix}__forward_segmask"
    csv_path = sample_dir / f"{prefix}__metrics.csv"

    if not fwd_dir.is_dir():
        print(f"  [skip] {prefix}: no forward dir at {fwd_dir}")
        return False

    pngs = sorted(fwd_dir.glob("t_*.png"))
    if not pngs:
        print(f"  [skip] {prefix}: no PNGs under {fwd_dir}")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [{prefix}] segmenting {len(pngs)} forward PNGs  →  {out_dir.name}/")

    rows = []
    for i, png in enumerate(pngs):
        try:
            t_int = int(png.stem.split("_")[1])
        except (ValueError, IndexError):
            continue
        pm = segman_predict_png(reward_model, png, device)
        Image.fromarray(classid_to_rgb(pm)).save(out_dir / png.name, optimize=False)

        row = {"t": t_int}
        if gt_mask is not None:
            row["dice_cup_noisy"] = dice_binary(pm, gt_mask, 2)
            row["dice_rim_noisy"] = dice_binary(pm, gt_mask, 1)
            row["vcdr_noisy"]     = vcdr_from_classid(pm)
        rows.append(row)

        if (i + 1) % 100 == 0 or i == len(pngs) - 1:
            print(f"    [{i+1}/{len(pngs)}] t={t_int}")

    if update_csv and csv_path.exists() and gt_mask is not None:
        df_existing = pd.read_csv(csv_path)
        df_new = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
        # merge on t (left join to preserve all existing rows)
        merged = df_existing.merge(df_new, on="t", how="left", suffixes=("", "_dup"))
        # drop any accidental dup cols
        for c in list(merged.columns):
            if c.endswith("_dup"):
                merged.drop(columns=[c], inplace=True)
        # compute vcdr_err_noisy if vcdr_gt present
        if "vcdr_noisy" in merged.columns and "vcdr_gt" in merged.columns:
            merged["vcdr_err_noisy"] = (merged["vcdr_noisy"] - merged["vcdr_gt"]).abs()
        # Backup original CSV only if no forward columns were present before
        backup = csv_path.with_suffix(".csv.fwdbak")
        if not backup.exists():
            csv_path.rename(backup)
            print(f"  [{prefix}] original csv backed up → {backup.name}")
        merged.to_csv(csv_path, index=False)
        print(f"  [{prefix}] csv updated with dice_cup_noisy / dice_rim_noisy / vcdr_noisy / vcdr_err_noisy")
    elif not csv_path.exists():
        # No CSV to merge into; write a standalone one for the forward-side metrics
        df_new = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
        df_new.to_csv(sample_dir / f"{prefix}__forward_metrics.csv", index=False)
        print(f"  [{prefix}] wrote standalone forward_metrics.csv (no existing metrics csv to merge)")

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_dir", required=True,
                    help="e.g. <SAMPLE_DIR>/data_00893")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="variant prefixes to process (default: auto-scan all *__forward/ dirs)")
    ap.add_argument("--segman_config",
                    default=str(HERE / ".." / "SegMAN" / "segmentation" / "local_configs" / "segman" / "base" / "segman_b_ade.py"))
    ap.add_argument("--segman_ckpt",
                    default=str(HERE / ".." / "checkpoints" / "segman_b" / "segman_b.pth"))
    ap.add_argument("--skip_csv_update", action="store_true",
                    help="Only save segmask PNGs; do not add a noisy column to metrics.csv.")
    args = ap.parse_args()

    sample_dir = Path(args.sample_dir).resolve()
    if not sample_dir.is_dir():
        raise SystemExit(f"sample_dir not a directory: {sample_dir}")

    # ---- Determine variants to process ----
    if args.variants:
        variants = args.variants
    else:
        variants = sorted({
            d.name.rsplit("__forward", 1)[0]
            for d in sample_dir.iterdir()
            if d.is_dir() and d.name.endswith("__forward")
        })
    if not variants:
        raise SystemExit(f"no *__forward/ directories found under {sample_dir}")
    print(f"[plan] will process variants: {variants}")

    # ---- Load GT mask from condition image ----
    gt_mask, gt_src = load_gt_mask(sample_dir)
    if gt_mask is None:
        print(f"[warn] GT mask not found (no valid 00_input_conditioning.png) — "
              f"will skip dice / vcdr metrics, only save PNGs.")
    else:
        n_cup = int((gt_mask == 2).sum()); n_rim = int((gt_mask == 1).sum())
        print(f"[gt] loaded from {gt_src}   cup={n_cup}  rim={n_rim}")

    # ---- Load SegMAN ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    reward_model = load_segman(args.segman_config, args.segman_ckpt, device)

    # ---- Process ----
    n_ok = 0
    for prefix in variants:
        ok = process_variant(sample_dir, prefix, reward_model, device,
                             gt_mask=gt_mask, update_csv=not args.skip_csv_update)
        n_ok += int(ok)

    print(f"\n[DONE] processed {n_ok}/{len(variants)} variants in {sample_dir}")


if __name__ == "__main__":
    main()
