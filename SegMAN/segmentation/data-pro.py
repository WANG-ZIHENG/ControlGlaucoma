import os
import numpy as np
import imageio
from tqdm import tqdm

def convert_masks(
    input_dir='data/segman/test/masks',
    output_dir='data/segman/test/masks_fixed'
):
    """
    Convert segmentation masks with pixel values [0,128,255] into [0,1,2],
    keeping file names the same. Progress bar will be shown during conversion.
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 .png 文件
    mask_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]

    print(f"✅ Found {len(mask_files)} mask files.")
    print(f"🔧 Converting from [0,128,255] to [0,1,2] ...")

    # 逐张处理并显示进度条
    for fname in tqdm(mask_files, desc='Processing', ncols=80):
        src_path = os.path.join(input_dir, fname)
        dst_path = os.path.join(output_dir, fname)

        try:
            # 读取图像
            mask = imageio.imread(src_path)
            if mask.ndim == 3:  # 若是RGB则转灰度
                mask = mask[..., 0]

            # 映射值
            mask_converted = np.zeros_like(mask, dtype=np.uint8)
            mask_converted[mask == 128] = 1
            mask_converted[mask == 255] = 2

            # 保存新图像
            imageio.imwrite(dst_path, mask_converted)

        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

    print(f"\n✅ All done! Converted masks are saved to:\n   {output_dir}")

    # 验证转换结果
    check_values(output_dir)

def check_values(folder):
    """
    Check unique values in converted masks to confirm correctness.
    """
    print("\n🔍 Checking unique pixel values in converted masks ...")
    sample_files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
    bad_files = []
    for fname in tqdm(sample_files[:100], desc='Verifying', ncols=80):
        arr = imageio.imread(os.path.join(folder, fname))
        unique_vals = np.unique(arr)
        if not set(unique_vals.tolist()).issubset({0,1,2}):
            bad_files.append((fname, unique_vals))
    if bad_files:
        print(f"⚠️  Found {len(bad_files)} problematic masks:")
        for f, vals in bad_files[:5]:
            print(f"   {f}: values = {vals}")
    else:
        print("✅ Verification passed: all masks contain only {0,1,2}.")

if __name__ == "__main__":
    convert_masks()
