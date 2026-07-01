"""
Differentiable vCDR consistency loss.

The vertical-spread operator Φ_y is implemented as a softmax-weighted row
pooling followed by a weighted spread over the normalized vertical coordinate
ỹ ∈ [0, 1]. The same Φ_y is applied to the prediction-side cup/disc probability
maps and to the target-side cup/disc binary masks, yielding the surrogate ratio

    r = Φ_y(cup) / (Φ_y(disc) + η).

This module returns the SIGNED difference r̂ − r*; the caller squares it.
"""
from __future__ import annotations
import torch


__all__ = [
    "compute_r_sigma_surrogate",
    "extract_cup_disc_masks_from_labels",
    "vcdr_signed_difference",
]


def compute_r_sigma_surrogate(cup_map, disc_map, alpha=10.0, eps=1e-6, normalize_y=True):
    """Compute Φ_y(cup) / (Φ_y(disc) + eps) for a batch of (cup, disc) maps.

    Args:
        cup_map  : (B, H, W) in [0, 1].
        disc_map : (B, H, W) in [0, 1], disc = cup ∪ rim.
        alpha    : temperature of the row-wise softmax pooling.
        eps      : η for numerical stability.
        normalize_y : map row index to ỹ ∈ [0, 1].

    Returns:
        r        : (B,)  Φ_y(cup) / (Φ_y(disc) + eps).
        phi_cup  : (B,)  Φ_y(cup).
        phi_disc : (B,)  Φ_y(disc).
    """
    _, H, W = cup_map.shape

    attn_cup  = torch.softmax(alpha * cup_map,  dim=-1)
    attn_disc = torch.softmax(alpha * disc_map, dim=-1)
    p_y_cup   = (cup_map  * attn_cup ).sum(dim=-1)
    p_y_disc  = (disc_map * attn_disc).sum(dim=-1)

    if normalize_y:
        y = torch.arange(H, device=cup_map.device, dtype=cup_map.dtype) / max(H - 1, 1)
    else:
        y = torch.arange(H, device=cup_map.device, dtype=cup_map.dtype)

    def _weighted_std(p_y):
        total = p_y.sum(dim=-1) + eps
        mu    = (p_y * y).sum(dim=-1) / total
        y_c   = y.unsqueeze(0) - mu.unsqueeze(-1)
        var   = (p_y * y_c.pow(2)).sum(dim=-1) / total
        return torch.sqrt(var.clamp(min=0.0) + eps)

    phi_cup  = _weighted_std(p_y_cup)
    phi_disc = _weighted_std(p_y_disc)
    r        = phi_cup / (phi_disc + eps)
    return r, phi_cup, phi_disc


def extract_cup_disc_masks_from_labels(labels_tensor):
    """Convert a labels tensor into (M_cup, M_disc) binary maps.

    Accepts (B, H, W), (B, 1, H, W) with class ids {0=bg, 1=rim, 2=cup}, or
    (B, 3, H, W) one-hot.

    Returns:
        cup_mask  : M_cup  = I[Y == 2]
        disc_mask : M_disc = I[Y > 0] = M_cup ∪ M_rim
    """
    if labels_tensor.dim() == 4:
        if labels_tensor.shape[1] == 3:
            cup_mask = labels_tensor[:, 2].float()
            rim_mask = labels_tensor[:, 1].float()
        elif labels_tensor.shape[1] == 1:
            labels_2d = labels_tensor.squeeze(1).float()
            cup_mask = (labels_2d >= 1.5).float()
            rim_mask = ((labels_2d >= 0.5) & (labels_2d < 1.5)).float()
        else:
            raise ValueError(f"unexpected 4D labels shape {tuple(labels_tensor.shape)}")
    elif labels_tensor.dim() == 3:
        labels_2d = labels_tensor.float()
        cup_mask = (labels_2d >= 1.5).float()
        rim_mask = ((labels_2d >= 0.5) & (labels_2d < 1.5)).float()
    else:
        raise ValueError(f"unexpected labels shape {tuple(labels_tensor.shape)}")
    disc_mask = (cup_mask + rim_mask).clamp(max=1.0)
    return cup_mask, disc_mask


def vcdr_signed_difference(
    segmentation_outputs, target, device,
    timestep_mask=None, return_all=False,
    alpha=10.0, eps=1e-6, normalize_y=True,
):
    """Signed difference  r_pred − r_target  used by L_vCDR.

    Args:
        segmentation_outputs: (B, C, H, W) with C ≥ 3  (bg / rim / cup).
        target: label tensor (B,H,W) / (B,1,H,W) / (B,3,H,W).
        timestep_mask: (B, 1) gating samples outside the rewarding band.
    """
    if not isinstance(segmentation_outputs, torch.Tensor):
        raise TypeError(
            f"segmentation_outputs must be Tensor; got {type(segmentation_outputs)}"
        )
    if segmentation_outputs.dim() != 4:
        raise ValueError(
            f"segmentation_outputs must be 4D; got {tuple(segmentation_outputs.shape)}"
        )
    if segmentation_outputs.shape[1] < 3:
        raise ValueError(
            f"Expect ≥3 seg classes (bg/rim/cup); got C={segmentation_outputs.shape[1]}."
        )

    B = segmentation_outputs.shape[0]
    if timestep_mask is None:
        timestep_mask = torch.ones(B, 1, device=device, dtype=torch.bool)

    if not timestep_mask.any():
        zero_with_graph = 0.0 * segmentation_outputs.float().flatten(1).sum(dim=1)
        if return_all:
            z = zero_with_graph.detach().clone()
            return zero_with_graph, z, z
        return zero_with_graph

    probs    = torch.softmax(segmentation_outputs.float(), dim=1)
    cup_prob  = probs[:, 2]
    disc_prob = probs[:, 1] + probs[:, 2]
    r_pred, _, _ = compute_r_sigma_surrogate(
        cup_prob, disc_prob, alpha=alpha, eps=eps, normalize_y=normalize_y,
    )

    with torch.no_grad():
        target_on_dev = target.to(device=device)
        cup_mask, disc_mask = extract_cup_disc_masks_from_labels(target_on_dev)
        cup_mask  = cup_mask.to(dtype=cup_prob.dtype)
        disc_mask = disc_mask.to(dtype=cup_prob.dtype)
        r_target, _, _ = compute_r_sigma_surrogate(
            cup_mask, disc_mask, alpha=alpha, eps=eps, normalize_y=normalize_y,
        )

    differences = r_pred - r_target
    mask_flat = timestep_mask.to(device=device).view(-1).to(dtype=differences.dtype)
    differences = differences * mask_flat

    if return_all:
        return differences, r_pred, r_target
    return differences
