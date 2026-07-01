
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from utils import calculate_cup_disc_ratio_from_mask  # Fu-ellipse vCDR


def segmask_png_to_classid(png_path):
    """Decode the RGB-coded segmask PNG back to {0=bg, 1=rim, 2=cup}.

    Encoding (matches eval_single_step_visualize.py):
        bg  → (0,   0,   0)
        rim → (0,   0,   255)
        cup → (255, 0,   0)
    """
    arr = np.array(Image.open(png_path).convert("RGB"))
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    cls = np.zeros(arr.shape[:2], dtype=np.int64)
    # rim = blue
    cls[(r == 0) & (g == 0) & (b == 255)] = 1
    # cup = red
    cls[(r == 255) & (g == 0) & (b == 0)] = 2
    return cls


def png_or_npz_to_classid_mask(path):
    """Load a GT mask source and return (H, W) class-id mask {0, 1, 2}.

    Accepts:
        - PNG with grayscale {0, 128, 255}
        - PNG with grayscale {0, 1, 2}
        - PNG with RGB palette (red=cup, blue=rim)  — used for our condition image
        - .npz with field 'disc_cup_mask' in {0, -1, -2}
    """
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=True)
        ms = data['disc_cup_mask']
        cls = np.zeros(ms.shape, dtype=np.int64)
        cls[ms == -1] = 1
        cls[ms == -2] = 2
        return cls

    arr = np.array(Image.open(p))
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        # RGB palette
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        cls = np.zeros(arr.shape[:2], dtype=np.int64)
        cls[(r > 128) & (g < 96) & (b < 96)] = 2   # cup = red
        cls[(r < 96)  & (g < 96) & (b > 128)] = 1  # rim = blue
        return cls
    # 2D grayscale
    m = arr.astype(int) if arr.ndim == 2 else arr[..., 0].astype(int)
    uniq = set(np.unique(m).tolist())
    if uniq.issubset({0, 1, 2}):
        return m.astype(np.int64)
    if uniq.issubset({0, 128, 255}):
        cls = np.zeros_like(m, dtype=np.int64)
        cls[m == 128] = 1
        cls[m == 255] = 2
        return cls
    raise ValueError(f"Unrecognized GT mask encoding at {p}: unique={sorted(uniq)[:10]}")


def vcdr_from_classid(cls_mask):
    """Wrap calculate_cup_disc_ratio_from_mask which expects {0, -1, -2}."""
    if cls_mask.ndim == 3:
        cls_mask = cls_mask[0]
    signed = np.zeros(cls_mask.shape, dtype=np.float32)
    signed[cls_mask == 1] = -1.0
    signed[cls_mask == 2] = -2.0
    try:
        return float(calculate_cup_disc_ratio_from_mask(signed))
    except Exception:
        return 0.0


def save_vcdr_curve(df, vcdr_gt, title, out_path):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(df.t, df.vcdr_pred, color="#9467bd", label="vCDR(recon)")
    ax1.axhline(vcdr_gt, color="k", ls=":", alpha=0.6, lw=1.2,
                label=f"GT vCDR = {vcdr_gt:.4f}")
    ax1.set_xlabel("timestep t"); ax1.set_ylabel("vCDR")
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


VARIANT_PREFIXES = [
    "A1_vPred_preTrained",
    "A2_vPred_FineTuned",
    "B1_Eps_preTrained",
    "B2_Eps_FineTuned",
]


def process_variant(sample_dir, prefix, vcdr_gt):
    seg_dir = sample_dir / f"{prefix}__segmask"
    csv_path = sample_dir / f"{prefix}__metrics.csv"
    if not seg_dir.is_dir():
        print(f"  [skip] {prefix}: no segmask dir at {seg_dir}")
        return False
    if not csv_path.exists():
        print(f"  [skip] {prefix}: no metrics csv at {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    if "vcdr_pred" in df.columns:
        print(f"  [skip] {prefix}: vcdr_pred already in csv (already processed)")
        # still regenerate plot in case user wants it refreshed
        save_vcdr_curve(df, vcdr_gt,
                        title=prefix, out_path=sample_dir / f"{prefix}__vcdr_curve.png")
        return True

    # Build a t→vcdr map by reading every segmask PNG
    pngs = sorted(seg_dir.glob("t_*.png"))
    print(f"  [{prefix}] decoding {len(pngs)} segmask PNGs...")
    t_to_vcdr = {}
    for png in pngs:
        try:
            t_int = int(png.stem.split("_")[1])
        except (ValueError, IndexError):
            continue
        cls = segmask_png_to_classid(png)
        t_to_vcdr[t_int] = vcdr_from_classid(cls)

    # Add columns to CSV (rows whose t lacks a PNG → NaN)
    df["vcdr_pred"] = df["t"].map(t_to_vcdr)
    df["vcdr_gt"]   = vcdr_gt
    df["vcdr_err"]  = (df["vcdr_pred"] - df["vcdr_gt"]).abs()

    # Backup original
    backup = csv_path.with_suffix(".csv.bak")
    if not backup.exists():
        csv_path.rename(backup)
        df.to_csv(csv_path, index=False)
        print(f"  [{prefix}] wrote {csv_path}  (orig backed up to {backup.name})")
    else:
        df.to_csv(csv_path, index=False)
        print(f"  [{prefix}] wrote {csv_path}  (backup already existed)")

    # Plot
    df_plot = df.dropna(subset=["vcdr_pred"]).reset_index(drop=True)
    if len(df_plot) > 0:
        save_vcdr_curve(df_plot, vcdr_gt,
                        title=prefix, out_path=sample_dir / f"{prefix}__vcdr_curve.png")
        print(f"  [{prefix}] saved vcdr_curve.png")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_dir", required=True,
                    help="Sample output dir (contains <prefix>__segmask/ + <prefix>__metrics.csv).")
    ap.add_argument("--gt_source", default=None,
                    help="GT mask source: a PNG (grayscale or RGB palette) or .npz with 'disc_cup_mask'. "
                         "If omitted, falls back to <sample_dir>/00_input_conditioning.png "
                         "(decode red/blue palette).")
    args = ap.parse_args()

    sample_dir = Path(args.sample_dir).resolve()
    if not sample_dir.is_dir():
        raise SystemExit(f"sample_dir not a directory: {sample_dir}")

    # Resolve GT source
    if args.gt_source is None:
        guess = sample_dir / "00_input_conditioning.png"
        if not guess.exists():
            raise SystemExit(
                f"No --gt_source given and fallback {guess} not found. "
                f"Pass --gt_source <path-to-mask-or-npz>."
            )
        args.gt_source = str(guess)
        print(f"[gt_source] auto-fallback to {args.gt_source}")

    print(f"[load] GT mask from {args.gt_source}")
    gt_cls = png_or_npz_to_classid_mask(args.gt_source)
    n_cup = int((gt_cls == 2).sum())
    n_rim = int((gt_cls == 1).sum())
    print(f"[gt mask] cup pixels = {n_cup}, rim pixels = {n_rim}")
    if n_cup == 0 and n_rim == 0:
        raise SystemExit("GT mask decoded to all-bg. Check --gt_source.")

    vcdr_gt = vcdr_from_classid(gt_cls)
    print(f"[gt vCDR] = {vcdr_gt:.4f}")

    print(f"\n[process] sample_dir = {sample_dir}")
    n_ok = 0
    for prefix in VARIANT_PREFIXES:
        ok = process_variant(sample_dir, prefix, vcdr_gt)
        n_ok += int(ok)

    print(f"\n[DONE] processed {n_ok}/{len(VARIANT_PREFIXES)} variants in {sample_dir}")


if __name__ == "__main__":
    main()
