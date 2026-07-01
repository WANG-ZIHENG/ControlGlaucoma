#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的模型推理脚本
用法: python simple_inference.py --config CONFIG_PATH --checkpoint CHECKPOINT_PATH --image IMAGE_PATH
"""

import argparse
import os
import cv2
import numpy as np
import torch
import mmcv
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmseg.apis import inference_segmentor, show_result_pyplot


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='简单的语义分割模型推理')
    parser.add_argument('--config', default='local_configs/segman/base/segman_b_ade.py', help='配置文件路径')
    parser.add_argument('--checkpoint', default='../../checkpoints/segman_b/segman_b.pth', help='模型权重文件路径')
    parser.add_argument('--image',default='data/segman/val/images/sample.png', help='输入图像路径')
    parser.add_argument('--output', default='output.png', help='输出结果路径')
    parser.add_argument('--device', default='cuda:0', help='使用的设备 (cuda:0 或 cpu)')
    parser.add_argument('--opacity', type=float, default=0.5, 
                        help='分割结果的透明度 (0-1之间)')
    args = parser.parse_args()
    return args


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"权重文件不存在: {args.checkpoint}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"图像文件不存在: {args.image}")
    
    print(f"正在加载配置文件: {args.config}")
    # 加载配置文件
    cfg = mmcv.Config.fromfile(args.config)
    
    # 设置为测试模式
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    
    print(f"正在构建模型...")
    # 构建模型
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    print(f"正在加载权重: {args.checkpoint}")
    # 加载权重
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # 获取类别信息
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    
    # 重要：将配置附加到模型对象上（inference_segmentor需要这个）
    model.cfg = cfg
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"正在对图像进行推理: {args.image}")
    # 进行推理
    result = inference_segmentor(model, args.image)
    
    # 可视化结果
    print(f"正在保存结果到: {args.output}")
    img = mmcv.imread(args.image)
    
    # 使用模型的show_result方法可视化
    model.show_result(
        img, 
        result, 
        out_file=args.output,
        opacity=args.opacity
    )
    
    print(f"推理完成! 结果已保存到: {args.output}")
    
    # 打印一些统计信息
    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        print(f"\n模型类别数: {len(model.CLASSES)}")
        print(f"类别列表: {model.CLASSES}")
    
    # 打印分割结果中的唯一类别
    seg_map = result[0]
    unique_labels = np.unique(seg_map)
    print(f"\n当前图像中检测到的类别ID: {unique_labels.tolist()}")


if __name__ == '__main__':
    main()

