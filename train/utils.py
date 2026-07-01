from __future__ import annotations
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as F
import sys
import os
import cv2
from skimage import measure

from PIL import Image
from typing import Optional
from functools import partial
from torch import Tensor
from torchvision import transforms
from torch.nn.modules.loss import _Loss
import torch

from canny_tools import Canny  # canny edge detection
from mmengine.hub import get_model  # segmentation
from transformers import DPTForDepthEstimation  # depth estimation

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TransUNet_networks'))
from model_interface import TransUNetInterface
from collections.abc import Callable, Sequence
import math
import mmcv
from mmcv.runner import load_checkpoint
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SegMAN', 'segmentation'))
from mmseg.models import build_segmentor
from mmseg.apis import inference_segmentor


from torchvision.transforms import RandomCrop
from collections.abc import Callable, Sequence
from monai.utils import DiceCEReduction, LossReduction, Weight, deprecated_arg, look_up_option, pytorch_after
import warnings
from monai.networks import one_hot



class SegMANWrapper(nn.Module):
    def __init__(self, config_path, checkpoint_path, device='cuda'):
        super().__init__()
        
        cfg = mmcv.Config.fromfile(config_path)
        
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        
        self.model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        
        checkpoint = load_checkpoint(self.model, checkpoint_path, map_location='cpu')
        
        if 'CLASSES' in checkpoint.get('meta', {}):
            self.model.CLASSES = checkpoint['meta']['CLASSES']
        if 'PALETTE' in checkpoint.get('meta', {}):
            self.model.PALETTE = checkpoint['meta']['PALETTE']
        
        self.model.cfg = cfg
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        print("SegMAN model loaded.")
        if hasattr(self.model, 'CLASSES') and self.model.CLASSES is not None:
            print(f"num classes: {len(self.model.CLASSES)}")
            print(f"classes: {self.model.CLASSES}")
    
    def forward(self, x):
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_denorm = x * imagenet_std + imagenet_mean

        x_denorm = x_denorm.clamp(0, 1)

        x_255 = x_denorm * 255.0

        # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
        segman_mean = torch.tensor([123.675, 116.28, 103.53], device=x.device).view(1, 3, 1, 1)
        segman_std = torch.tensor([58.395, 57.12, 57.375], device=x.device).view(1, 3, 1, 1)
        x_normalized = (x_255 - segman_mean) / segman_std

        result = self.model.encode_decode(x_normalized, img_metas=None)
        return result
    
    def to(self, *args, **kwargs):
        if len(args) > 0:
            device_arg = args[0]
            if isinstance(device_arg, (torch.device, str)):
                self.device = device_arg
        elif 'device' in kwargs:
            self.device = kwargs['device']
        
        self.model = self.model.to(*args, **kwargs)
        return self


class DiceUncertaintyLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The `smooth_nr` and `smooth_dr` parameters are values added to the intersection and union components of
    the inter-over-union calculation to smooth results respectively, these values should be small.

    The original paper: Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric
    Medical Image Segmentation, 3DV, 2016.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
        use_uncertainty_weight: bool = True,
        uncertainty_loss_weight: float = 0.0,
        exclude_background_uncertainty: bool = True,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.
            use_uncertainty_weight: if True, apply uncertainty-based weighting to the loss computation.
                Defaults to True.
            uncertainty_loss_weight: weight for uncertainty MSE loss. If > 0, adds MSE loss between 
                average uncertainty and 0 to encourage the model to reduce prediction uncertainty.
                Defaults to 0.0 (disabled).
            exclude_background_uncertainty: if True, only compute uncertainty MSE on foreground pixels 
                (where target is not 0) to exclude background influence. Defaults to True.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        self.use_uncertainty_weight = use_uncertainty_weight
        self.uncertainty_loss_weight = float(uncertainty_loss_weight)
        self.exclude_background_uncertainty = exclude_background_uncertainty
        weight = torch.as_tensor(weight) if weight is not None else None
        self.register_buffer("class_weight", weight)
        self.class_weight: None | torch.Tensor

    def pixel_entropy_from_logits(self,logits, eps=1e-12):
        probs = torch.nn.functional.softmax(logits, dim=1)  # [B, K, H, W]
        ent = -(probs * (probs.clamp(min=eps).log())).sum(dim=1, keepdim=True)  # [B,1,H,W]
        return ent

    def normalize_entropy(self,ent, K):
        return (ent / (torch.log(torch.tensor(K, dtype=ent.dtype, device=ent.device)))).clamp(0, 1)

    def uncertainty_weight(self,ent_norm, kappa=3.0):
        return 1.0 / (1.0 + kappa * ent_norm)  # [B,1,H,W]


    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Returns:
            tuple: (total_loss, dice_loss, uncertainty_mse_loss)
                - total_loss: dice_loss + uncertainty_loss_weight * uncertainty_mse_loss
                - dice_loss: the dice loss component
                - uncertainty_mse_loss: the uncertainty MSE loss component (0 if uncertainty_loss_weight=0)

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> from monai.losses.dice import *  # NOQA
            >>> import torch
            >>> from monai.losses.dice import DiceLoss
            >>> B, C, H, W = 7, 5, 3, 2
            >>> logits = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = DiceLoss(reduction='none')
            >>> total_loss, dice_loss, uncertainty_mse_loss = self(input, target)
            >>> assert np.broadcast_shapes(total_loss.shape, input.shape) == input.shape
        """
        if self.sigmoid:
            input = torch.sigmoid(logits)

        n_pred_ch = logits.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(logits, 1)

        B, K, H, W = logits.shape
        
        original_target = target.clone() if self.exclude_background_uncertainty and self.uncertainty_loss_weight > 0 else None
        original_pred_classes = torch.argmax(input, dim=1, keepdim=True) if self.exclude_background_uncertainty and self.uncertainty_loss_weight > 0 else None  # [B, 1, H, W]
        
        ent_norm = None
        if self.use_uncertainty_weight or self.uncertainty_loss_weight > 0:
            #  Compute pixel-wise entropy
            ent = self.pixel_entropy_from_logits(logits, eps=1e-12)  # [B,1,H,W]

            #  Normalize entropy to [0,1]
            ent_norm = self.normalize_entropy(ent, K)  # [B,1,H,W]
        
        if self.use_uncertainty_weight:
            #  Compute uncertainty weights
            weights = self.uncertainty_weight(ent_norm, kappa=3)  # [B,1,H,W]

            # Expand weights to match probs shape: [B,1,H,W] -> [B,K,H,W]
            weights = weights.expand(-1, K, -1, -1)  # [B,K,H,W]
        else:
            weights = torch.ones(B, K, H, W, dtype=logits.dtype, device=logits.device)


        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        # Weighted intersection and union across all pixels
        input = input * weights  # [B,K,H,W]
        target = target * weights  # [B,K,H,W]


        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            ground_o = torch.sum(target**2, dim=reduce_axis)
            pred_o = torch.sum(input**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        dice_loss: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        num_of_classes = target.shape[1]
        if self.class_weight is not None and num_of_classes != 1:
            # make sure the lengths of weights are equal to the number of classes
            if self.class_weight.ndim == 0:
                self.class_weight = torch.as_tensor([self.class_weight] * num_of_classes)
            else:
                if self.class_weight.shape[0] != num_of_classes:
                    raise ValueError(
                        """the length of the `weight` sequence should be the same as the number of classes.
                        If `include_background=False`, the weight should not include
                        the background category class 0."""
                    )
            if self.class_weight.min() < 0:
                raise ValueError("the value/values of the `weight` should be no less than 0.")
            # apply class_weight to loss
            dice_loss = dice_loss * self.class_weight.to(dice_loss)

        if self.reduction == LossReduction.MEAN.value:
            dice_loss = torch.mean(dice_loss)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            dice_loss = torch.sum(dice_loss)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # For 'none' reduction, we want per-sample loss (not per-class-per-sample)
            # Average over the class dimension to get [B] shape
            if dice_loss.ndim > 1:
                dice_loss = dice_loss.mean(dim=tuple(range(1, dice_loss.ndim)))  # [B, C, ...] -> [B]
            # Now dice_loss is [B] shape
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        uncertainty_mse = torch.tensor(0.0, dtype=dice_loss.dtype, device=dice_loss.device)
        if self.uncertainty_loss_weight > 0:
            # ent_norm shape: [B, 1, H, W]
            
            if self.exclude_background_uncertainty:
                if original_target.ndim == 3:
                    # [B, H, W] -> [B, 1, H, W]
                    target_foreground_mask = (original_target != 0).unsqueeze(1).float()
                elif original_target.ndim == 4 and original_target.shape[1] == 1:
                    # [B, 1, H, W]
                    target_foreground_mask = (original_target != 0).float()
                else:
                    # [B, C, H, W] where C > 1, sum over channel dimension
                    target_foreground_mask = (original_target.sum(dim=1, keepdim=True) != 0).float()
                
                pred_foreground_mask = (original_pred_classes != 0).float()  # [B, 1, H, W]
                
                foreground_mask = torch.max(target_foreground_mask, pred_foreground_mask)  # [B, 1, H, W]
                
                # masked_uncertainty shape: [B, 1, H, W]
                masked_uncertainty = ent_norm * foreground_mask  # [B, 1, H, W]
                
                num_foreground_pixels_per_sample = foreground_mask.view(B, -1).sum(dim=1)  # [B]
                
                uncertainty_sum_per_sample = masked_uncertainty.view(B, -1).sum(dim=1)  # [B]
                
                avg_uncertainty_per_sample = torch.where(
                    num_foreground_pixels_per_sample > 0,
                    uncertainty_sum_per_sample / num_foreground_pixels_per_sample,
                    torch.zeros_like(uncertainty_sum_per_sample)
                )  # [B]
            else:
                # ent_norm shape: [B, 1, H, W] -> [B]
                avg_uncertainty_per_sample = ent_norm.view(B, -1).mean(dim=1)  # [B]
            
            uncertainty_mse_per_sample = (avg_uncertainty_per_sample - 0) ** 2  # [B]
            
            if self.reduction == LossReduction.MEAN.value:
                uncertainty_mse = uncertainty_mse_per_sample.mean()
            elif self.reduction == LossReduction.SUM.value:
                uncertainty_mse = uncertainty_mse_per_sample.sum()
            elif self.reduction == LossReduction.NONE.value:
                uncertainty_mse = uncertainty_mse_per_sample  # [B]
            else:
                uncertainty_mse = uncertainty_mse_per_sample.mean()  # fallback
        
        total_loss = dice_loss + self.uncertainty_loss_weight * uncertainty_mse

        return total_loss, dice_loss,  uncertainty_mse




def get_reward_model(task='segmentation', model_path='mmseg::upernet/upernet_r50_4xb4-160k_ade20k-512x512.py', device='cuda'):
    """Return reward model for different tasks.

    Args:
        task (str, optional): Task name. Defaults to 'segmentation'.
        model_path (str, optional): Model name or pre-trained path.
            For SegMAN models, use format: 'segman::config_path::checkpoint_path'
            Example: 'segman::local_configs/segman/base/segman_b_ade.py::checkpoints/segman_b/segman_b.pth'
        device (str, optional): Device to load the model on. Defaults to 'cuda'.

    """
    if task == 'segmentation':
        #segman::local_configs/segman/base/segman_b_ade.py::checkpoints/segman_b/segman_b.pth
        if model_path.startswith('segman::'):
            parts = model_path.split('::')
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid SegMAN model path. Expected format: 'segman::config_path::checkpoint_path'\n"
                    f"Received: {model_path}"
                )
            _, config_path, checkpoint_path = parts
            print("Loading SegMAN model:")
            print(f"  config: {config_path}")
            print(f"  checkpoint: {checkpoint_path}")
            return SegMANWrapper(config_path, checkpoint_path, device=str(device))
        
        elif model_path.endswith('.pth') or model_path.endswith('.pt'):
            return TransUNetInterface(
                model_path=model_path,
                vit_name='R50-ViT-B_16',
                num_classes=3,
                img_size=224,
                n_skip=3,
                vit_patches_size=16
            ).model
        else:
            return get_model(model_path, pretrained=True)
    elif task == 'canny':
        return Canny()
    elif task == 'depth':
        return DPTForDepthEstimation.from_pretrained(model_path)
    elif task == 'lineart':
        model = LineDrawingModel()
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_path, map_location=torch.device('cpu')))
        return model
    elif task == 'hed':
        return HEDdetector(model_path)
    else:
        raise not NotImplementedError("Only support segmentation, canny and depth for now.")
# Prefer MONAI DiceLoss when available; fallback to custom soft dice
_dice_criterion_cache = {}

def get_reward_loss(predictions, labels, dataset_name, task='segmentation', use_uncertainty_weight=True, uncertainty_loss_weight=0.0, reduction="none", use_cross_entropy=False, **args):
    """Return reward loss for different tasks.

    Args:
        task (str, optional): Task name.
        use_uncertainty_weight (bool, optional): Whether to use uncertainty weighting in DiceLoss. Defaults to True.
        uncertainty_loss_weight (float, optional): Weight for uncertainty MSE loss. Defaults to 0.0.
        use_cross_entropy (bool, optional): For the FairSeg dataset, use cross-entropy loss instead of DiceUncertaintyLoss. Defaults to False.

    Returns:
        For segmentation with FairSeg dataset:
            - If use_cross_entropy=True: loss tensor (cross-entropy)
            - If use_cross_entropy=False: tuple of (total_loss, dice_loss, uncertainty_mse_loss)
        For other tasks: loss tensor
    """
    if task == 'segmentation':
        if dataset_name == 'limingcv/Captioned_ADE20K':
            return nn.functional.cross_entropy(predictions, labels, ignore_index=255, **args)
        elif "fairseg" in dataset_name.lower():
            if use_cross_entropy:
                if reduction == "none":
                    ce_loss = nn.functional.cross_entropy(predictions, labels, ignore_index=255, reduction='none', **args)
                    return ce_loss.view(ce_loss.shape[0], -1).mean(dim=1)
                else:
                    return nn.functional.cross_entropy(predictions, labels, ignore_index=255, reduction=reduction, **args)
            else:
                cache_key = (use_uncertainty_weight, uncertainty_loss_weight, reduction)
                if cache_key not in _dice_criterion_cache:
                    _dice_criterion_cache[cache_key] = DiceUncertaintyLoss(
                        include_background=False, 
                        to_onehot_y=True, 
                        softmax=True, 
                        use_uncertainty_weight=use_uncertainty_weight,
                        uncertainty_loss_weight=uncertainty_loss_weight,
                        reduction=reduction

                    )
                dice_loss_fn = _dice_criterion_cache[cache_key]
                target = labels.unsqueeze(1) if labels.ndim == 3 else labels
                return dice_loss_fn(predictions, target)


    
    elif task == 'canny':
        loss = nn.functional.mse_loss(predictions, labels, **args).mean(2)
        return loss.mean((-1,-2))
    elif task in ['depth', 'lineart', 'hed']:
        loss = nn.functional.mse_loss(predictions, labels, **args)
        return loss
    else:
        raise not NotImplementedError("Only support segmentation, canny and depth for now.")


def image_grid(imgs, rows, cols):
    """Image grid for visualization."""
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def map_color_to_index(image, dataset='limingcv/Captioned_ADE20K'):
    """Map colored segmentation image (RGB) into original label format (L).

    Args:
        image (torch.tensor): image tensor with shape (N, 3, H, W).
        dataset (str, optional): Dataset name. Defaults to 'ADE20K'.

    Returns:
        torch.tensor: mask tensor with shape (N, H, W).
    """
    if dataset == 'limingcv/Captioned_ADE20K':
        palette = np.load('ade20k_palette.npy')
    elif dataset == 'limingcv/Captioned_COCOStuff':
        palette = np.load('coco_stuff_palette.npy')
    else:
        raise NotImplementedError("Only support ADE20K and COCO-Stuff dataset for now.")

    image = image * 255
    palette_tensor = torch.tensor(palette, dtype=image.dtype, device=image.device)
    reshaped_image = image.permute(0, 2, 3, 1).reshape(-1, 3)

    # Calculate the difference of colors and find the index of the minimum distance
    indices = torch.argmin(torch.norm(reshaped_image[:, None, :] - palette_tensor, dim=-1), dim=-1)

    # Transform indices back to original shape
    return indices.view(image.shape[0], image.shape[2], image.shape[3])


def seg_label_transform(
        labels,
        dataset_name='limingcv/Captioned_ADE20K',
        output_size=(64, 64),
        interpolation=transforms.InterpolationMode.NEAREST,
        max_size=None,
        antialias=True):
    """Adapt RGB seg_map into loss computation. \
    (1) Map the RGB seg_map into the original label format (Single Channel). \
    (2) Resize the seg_map into the same size as the output feature map. \
    (3) Remove background class if needed (usually for ADE20K).

    Args:
        labels (torch.tensor): Segmentation map. (N, 3, H, W) for ADE20K and (N, H, W) for COCO-Stuff.
        dataset_name (string): Dataset name. Default to 'ADE20K'.
        output_size (tuple): Resized image size, should be aligned with the output of segmentation models.
        interpolation (optional): _description_. Defaults to transforms.InterpolationMode.NEAREST.
        max_size (optional): Defaults to None.
        antialias (optional): Defaults to True.

    Returns:
        torch.tensor: formatted labels for loss computation.
    """

    if dataset_name == 'limingcv/Captioned_ADE20K':
        labels = map_color_to_index(labels, dataset_name)
        labels = F.resize(labels, output_size, interpolation, max_size, antialias)

        # 0 means the background class in ADE20K
        # In a unified format, we use 255 to represent the background class for both ADE20K and COCO-Stuff
        labels = labels - 1
        labels[labels == -1] = 255
    elif dataset_name == 'limingcv/Captioned_COCOStuff':
        labels = F.resize(labels, output_size, interpolation, max_size, antialias)

    return labels.long()

def depth_label_transform(
        labels,
        dataset_name,
        output_size=None,
        interpolation=transforms.InterpolationMode.BILINEAR,
        max_size=None,
        antialias=True
    ):

    if output_size is not None:
        labels = F.resize(labels, output_size, interpolation, max_size, antialias)
    return labels


def edge_label_transform(labels, dataset_name):
    return labels


def label_transform(labels, task, dataset_name, **args):
    if task == 'segmentation':
        return seg_label_transform(labels, dataset_name, **args)
    elif task == 'depth':
        return depth_label_transform(labels, dataset_name, **args)
    elif task in ['canny', 'lineart', 'hed']:
        return edge_label_transform(labels, dataset_name, **args)
    else:
        raise NotImplementedError("Only support segmentation and edge detection for now.")


def group_random_crop(images, resolution):
    """
    Args:
        images (list of PIL Image or Tensor): List of images to be cropped.

    Returns:
        List of PIL Image or Tensor: List of cropped image.
    """

    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    for idx, image in enumerate(images):
        i, j, h, w = RandomCrop.get_params(image, output_size=resolution)
        images[idx] = F.crop(image, i, j, h, w)

    return images


norm_layer = nn.InstanceNorm2d
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class LineDrawingModel(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, n_residual_blocks=3, sigmoid=True):
        super(LineDrawingModel, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out



class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        for i in range(1, layer_number):
            self.convs.append(torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.projection = torch.nn.Conv2d(in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class HEDdetector(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        state_dict = torch.hub.load_state_dict_from_url(model_path, map_location=torch.device('cpu'))

        self.netNetwork = ControlNetHED_Apache2()
        self.netNetwork.load_state_dict(state_dict)

    def __call__(self, input_image):
        H, W = input_image.shape[2], input_image.shape[3]

        edges = self.netNetwork((input_image * 255).clip(0, 255))
        edges = [torch.nn.functional.interpolate(edge, size=(H, W), mode='bilinear') for edge in edges]
        edges = torch.stack(edges, dim=1)
        edge = 1 / (1 + torch.exp(-torch.mean(edges, dim=1)))
        edge = (edge * 255.0).clip(0, 255).to(torch.uint8)

        return edge / 255.0



def check_cup_inside_disc(mask_image):
    disc_mask = (mask_image == -1).astype(np.uint8)
    cup_mask = (mask_image == -2).astype(np.uint8)
    
    if cup_mask.sum() == 0 or disc_mask.sum() == 0:
        return False
    
    from scipy import ndimage
    
    disc_dilated = ndimage.binary_dilation(disc_mask, iterations=1)
    
    cup_inside_disc = np.all(cup_mask <= disc_dilated+cup_mask)
    
    return cup_inside_disc


def calculate_cup_disc_ratio_from_mask(mask_image):
    if not check_cup_inside_disc(mask_image):
        return 0.0

    class1_mask = (mask_image == -1).astype(np.uint8)
    class2_mask = (mask_image == -2).astype(np.uint8)

    red_long = calculate_ellipse_long_axis(class1_mask)

    blue_long = calculate_ellipse_long_axis(class2_mask)

    if red_long > 0:
        relative_size = blue_long / red_long
        return relative_size
    else:
        return 0.0



def calculate_ellipse_long_axis(mask):

    try:
        contours = measure.find_contours(mask, 0.99)
        if len(contours) == 0 or len(contours[0]) < 5:
            print(f"[Warning] calculate_ellipse_long_axis: No valid contours found or too few points (len={len(contours[0]) if contours else 0})")
            return 0.0
        contours = contours[0]
    except Exception as e:
        print(f"[Warning] calculate_ellipse_long_axis: Failed to find contours: {e}")
        return 0.0

    try:
        ellipse = cv2.fitEllipse(contours.reshape(-1, 1, 2).astype(int))
    except Exception as e:
        print(f"[Warning] calculate_ellipse_long_axis: Failed to fit ellipse: {e}")
        return 0.0

    # _,(short,long),_ = ellipse
    mask = np.ascontiguousarray(mask)
    out = cv2.ellipse(mask, ellipse, (3), 2)
    rows, cols = np.where(out == 3)
    
    if len(cols) == 0 or len(rows) == 0:
        print(f"[Warning] calculate_ellipse_long_axis: No ellipse pixels found after drawing")
        return 0.0

    # vCDR: vertical diameter (max_y - min_y) to match clinical definition.
    max_y = np.max(rows)
    min_y = np.min(rows)
    long = max_y - min_y
    return long



# ============================================================
# Differentiable vCDR surrogate loss — re-exported from ``vcdr_loss.py``.
# The actual implementation lives in a standalone module with zero heavy
# dependencies (just torch), so it can be unit-tested on a minimal env.
# Backward-compat aliases are kept:
#   * calculate_cup_disc_ratio_difference_from_outputs(...)
#   * _compute_r_sigma_surrogate(...)
#   * _extract_cup_disc_masks_from_labels(...)
# ============================================================
from vcdr_loss import (
    compute_r_sigma_surrogate as _compute_r_sigma_surrogate,
    extract_cup_disc_masks_from_labels as _extract_cup_disc_masks_from_labels,
    vcdr_signed_difference as calculate_cup_disc_ratio_difference_from_outputs,
)

def cosine_decay(x, start=700, end=1000):
    x = np.asarray(x, dtype=float)

    # piecewise: 1 (<=start), cosine (start~end), 0 (>=end)
    y = np.where(
        x <= start, 1.0,
        np.where(
            x >= end, 0.0,
            0.5 * (1.0 + np.cos(np.pi * (x - start) / (end - start)))
        )
    )

    return y