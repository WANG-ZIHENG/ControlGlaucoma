# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import mmcv
import numpy as np
import torch


def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)

    if label_map is not None:
        label_copy = label.clone()
        for old_id, new_id in label_map.items():
            label[label_copy == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]:
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <IoU> ndarray: Per category IoU, shape (num_classes, ).
    """
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <Dice> ndarray: Per category dice, shape (num_classes, ).
    """

    dice_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return dice_result


def mean_fscore(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False,
                beta=1):
    """Calculate Mean F-Score (mFscore)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.


     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Fscore> ndarray: Per category recall, shape (num_classes, ).
            <Precision> ndarray: Per category precision, shape (num_classes, ).
            <Recall> ndarray: Per category f-score, shape (num_classes, ).
    """
    fscore_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mFscore'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
        beta=beta)
    return fscore_result


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics."""
    if not isinstance(gt_seg_maps, (list, tuple)):
        gt_seg_maps = list(gt_seg_maps)
    if not isinstance(results, (list, tuple)):
        results = list(results)
    total_area_intersect, total_area_union, total_area_pred_label, total_area_label = total_intersect_and_union(
        results, gt_seg_maps, num_classes, ignore_index, label_map, reduce_zero_label)
    
    ret_metrics = total_area_to_metrics(
        total_area_intersect, total_area_union, total_area_pred_label,
        total_area_label, metrics, nan_to_num, beta)

    # 🔍 计算 Precision / Recall / PID 时不覆盖已有指标
    if any(m in metrics for m in ['mPrecision', 'mRecall', 'PID']):
        from mmseg.core.evaluation.metrics import total_precision_recall_pid
        add_metrics = total_precision_recall_pid(
            results=results,
            gt_seg_maps=gt_seg_maps,
            num_classes=num_classes,      # ✅ 必须加上
            ignore_index=ignore_index
        )
        for k, v in add_metrics.items():
            if k not in ret_metrics:   # ✅ 防止覆盖 mIoU
                ret_metrics[k] = v

    # ✅ 保底：防止缺少关键指标导致 KeyError
    for key in ['mIoU', 'mDice', 'mAcc']:
        if key not in ret_metrics:
            ret_metrics[key] = np.nan

    return ret_metrics


def pre_eval_to_metrics(pre_eval_results,
                        metrics=['mIoU'],
                        nan_to_num=None,
                        beta=1):
    """Convert pre-eval results to metrics.

    Args:
        pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    total_area_intersect = sum(pre_eval_results[0])
    total_area_union = sum(pre_eval_results[1])
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


# def total_area_to_metrics(total_area_intersect,
#                           total_area_union,
#                           total_area_pred_label,
#                           total_area_label,
#                           metrics=['mIoU'],
#                           nan_to_num=None,
#                           beta=1):
#     """Calculate evaluation metrics
#     Args:
#         total_area_intersect (ndarray): The intersection of prediction and
#             ground truth histogram on all classes.
#         total_area_union (ndarray): The union of prediction and ground truth
#             histogram on all classes.
#         total_area_pred_label (ndarray): The prediction histogram on all
#             classes.
#         total_area_label (ndarray): The ground truth histogram on all classes.
#         metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
#         nan_to_num (int, optional): If specified, NaN values will be replaced
#             by the numbers defined by the user. Default: None.
#      Returns:
#         float: Overall accuracy on all images.
#         ndarray: Per category accuracy, shape (num_classes, ).
#         ndarray: Per category evaluation metrics, shape (num_classes, ).
#     """
#     if isinstance(metrics, str):
#         metrics = [metrics]
#     # allowed_metrics = ['mIoU', 'mDice', 'mFscore']
#     allowed_metrics = ['mIoU', 'mDice', 'mAcc', 'mPrecision', 'mRecall', 'PID']
#     if not set(metrics).issubset(set(allowed_metrics)):
#         raise KeyError('metrics {} is not supported'.format(metrics))

#     all_acc = total_area_intersect.sum() / total_area_label.sum()
#     ret_metrics = OrderedDict({'aAcc': all_acc})
#     for metric in metrics:
#         if metric == 'mIoU':
#             iou = total_area_intersect / total_area_union
#             acc = total_area_intersect / total_area_label
#             ret_metrics['IoU'] = iou
#             ret_metrics['Acc'] = acc
#         elif metric == 'mDice':
#             dice = 2 * total_area_intersect / (
#                 total_area_pred_label + total_area_label)
#             acc = total_area_intersect / total_area_label
#             ret_metrics['Dice'] = dice
#             ret_metrics['Acc'] = acc
#         elif metric == 'mFscore':
#             precision = total_area_intersect / total_area_pred_label
#             recall = total_area_intersect / total_area_label
#             f_value = torch.tensor(
#                 [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
#             ret_metrics['Fscore'] = f_value
#             ret_metrics['Precision'] = precision
#             ret_metrics['Recall'] = recall

#     ret_metrics = {
#         metric: value.numpy()
#         for metric, value in ret_metrics.items()
#     }
#     if nan_to_num is not None:
#         ret_metrics = OrderedDict({
#             metric: np.nan_to_num(metric_value, nan=nan_to_num)
#             for metric, metric_value in ret_metrics.items()
#         })
#     try:
#         # 如果上层 eval_metrics() 已经传入了 results 和 gt_seg_maps
#         import numpy as np
#         from mmseg.core.evaluation.metrics import total_precision_recall_pid
#         if 'results' in locals() and 'gt_seg_maps' in locals():
#             custom_metrics = total_precision_recall_pid(
#                 results=locals()['results'],
#                 gt_seg_maps=locals()['gt_seg_maps'])
#             ret_metrics.update(custom_metrics)
#     except Exception as e:
#         print(f"[Warning] Custom metrics skipped: {e}")
        
#     return ret_metrics

import numpy as np
from collections import OrderedDict

def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1):
    """Safe and complete implementation of total_area_to_metrics."""
    if isinstance(metrics, str):
        metrics = [metrics]

    allowed_metrics = ['mIoU', 'mDice', 'mAcc', 'mPrecision', 'mRecall', 'PID']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError(f'metrics {metrics} is not supported')

    eps = 1e-10  # 防止除零
    all_acc = total_area_intersect.sum() / (total_area_label.sum() + eps)

    iou = total_area_intersect / (total_area_union + eps)
    acc = total_area_intersect / (total_area_label + eps)
    dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label + eps)

    # ✅ 防止 nan 扩散
    iou = np.nan_to_num(iou, nan=0.0)
    acc = np.nan_to_num(acc, nan=0.0)
    dice = np.nan_to_num(dice, nan=0.0)

    # ✅ 汇总指标
    ret_metrics = OrderedDict()
    ret_metrics['aAcc'] = all_acc 
    ret_metrics['IoU'] = iou 
    ret_metrics['Acc'] = acc
    ret_metrics['Dice'] = dice
    ret_metrics['mIoU'] = np.mean(iou)
    ret_metrics['mAcc'] = np.mean(acc)
    ret_metrics['mDice'] = np.mean(dice)

    # ✅ nan_to_num 支持
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            k: np.nan_to_num(v, nan=nan_to_num)
            for k, v in ret_metrics.items()
        })

    return ret_metrics
# ===========================================================
# ✅ Custom Evaluation Metrics Extension (Precision / Recall / PID)
# ===========================================================
import numpy as np

import numpy as np

def precision_recall_pid_per_class(pred_label, gt_label, num_classes, ignore_index):
    """Compute per-class Precision, Recall, and PID using robust mmseg-style stats.
    Returns:
        precisions: (num_classes,)
        recalls:    (num_classes,)
        pids:       (num_classes,)
    """
    eps = 1e-10

    # --- to numpy & squeeze ---
    pred = np.asarray(pred_label)
    gt   = np.asarray(gt_label)

    # --- handle logits/prob maps: take argmax along the class axis ---
    if pred.ndim == 3:
        # try to find the class axis by matching num_classes
        axes = [i for i, s in enumerate(pred.shape) if s == num_classes]
        if len(axes) == 1:
            pred = np.argmax(pred, axis=axes[0])
        else:
            # fallback heuristics: prefer channel-first (C,H,W) then channel-last (H,W,C)
            if pred.shape[0] < 10 and pred.shape[0] == num_classes:
                pred = np.argmax(pred, axis=0)
            elif pred.shape[-1] < 10 and pred.shape[-1] == num_classes:
                pred = np.argmax(pred, axis=-1)
            else:
                # 无法判断类别轴，保底：对最后一维做 argmax
                pred = np.argmax(pred, axis=-1)

    # --- shape guard ---
    if pred.shape != gt.shape:
        # 尽量不崩：裁到共同最小区域；如果仍无效则返回全0
        h = min(pred.shape[0], gt.shape[0])
        w = min(pred.shape[1], gt.shape[1]) if pred.ndim == 2 and gt.ndim == 2 else None
        if w is None:
            return np.zeros(num_classes, dtype=np.float64), np.zeros(num_classes, dtype=np.float64), np.zeros(num_classes, dtype=np.float64)
        pred = pred[:h, :w]
        gt   = gt[:h, :w]

    # --- build valid mask (ignore + label bounds) ---
    valid = (gt != ignore_index)
    valid &= (gt >= 0)
    valid &= (gt < num_classes)

    if not np.any(valid):
        # 没有有效像素：返回全0（保持返回形状）
        return (np.zeros(num_classes, dtype=np.float64),
                np.zeros(num_classes, dtype=np.float64),
                np.zeros(num_classes, dtype=np.float64))

    pred_v = pred[valid].astype(np.int64, copy=False)
    gt_v   = gt[valid].astype(np.int64, copy=False)

    # --- clip/guard unexpected preds out of range ---
    pred_v = np.where((pred_v >= 0) & (pred_v < num_classes), pred_v, -1)
    keep   = pred_v != -1
    if not np.any(keep):
        return (np.zeros(num_classes, dtype=np.float64),
                np.zeros(num_classes, dtype=np.float64),
                np.zeros(num_classes, dtype=np.float64))
    pred_v = pred_v[keep]
    gt_v   = gt_v[keep]

    # --- confusion matrix (rows=gt, cols=pred) ---
    cm = np.bincount(gt_v * num_classes + pred_v,
                     minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    # TP, FP, FN per class
    TP = np.diag(cm).astype(np.float64)
    pred_per_class = cm.sum(axis=0).astype(np.float64)   # predicted as class c (column sum)
    gt_per_class   = cm.sum(axis=1).astype(np.float64)   # ground truth class c (row sum)
    FP = pred_per_class - TP
    FN = gt_per_class - TP

    # precision, recall, pid
    precisions = np.where((TP + FP) > 0, TP / (TP + FP + eps), 0.0)
    recalls    = np.where((TP + FN) > 0, TP / (TP + FN + eps), 0.0)
    pids       = np.where((2*TP + FP + FN) > 0, (2 * TP) / (2 * TP + FP + FN + eps), 0.0)

    # nan-safe (理论上不会nan，这里双保险)
    precisions = np.nan_to_num(precisions, nan=0.0)
    recalls    = np.nan_to_num(recalls,    nan=0.0)
    pids       = np.nan_to_num(pids,       nan=0.0)

    return precisions, recalls, pids



def total_precision_recall_pid(results, gt_seg_maps, num_classes=3, ignore_index=255):
    """Aggregate mean precision, recall, PID over dataset safely."""
    all_precisions, all_recalls, all_pids = [], [], []
    for pred, gt in zip(results, gt_seg_maps):
        p, r, d = precision_recall_pid_per_class(pred, gt, num_classes, ignore_index)
        all_precisions.append(p)
        all_recalls.append(r)
        all_pids.append(d)

    all_precisions = np.nan_to_num(np.stack(all_precisions), nan=0.0)
    all_recalls = np.nan_to_num(np.stack(all_recalls), nan=0.0)
    all_pids = np.nan_to_num(np.stack(all_pids), nan=0.0)

    mPrecision = np.mean(all_precisions)
    mRecall = np.mean(all_recalls)
    PID = np.mean(all_pids)

    return dict(mPrecision=mPrecision, mRecall=mRecall, PID=PID)




# ---- Hook old eval_metrics() to support new metrics ----
from mmseg.core.evaluation.metrics import total_area_to_metrics

def eval_metrics_with_custom(results,
                             gt_seg_maps,
                             num_classes,
                             ignore_index,
                             metrics=['mIoU'],
                             nan_to_num=None,
                             label_map=dict(),
                             reduce_zero_label=False,
                             beta=1):
    """Wrapper for extended metrics support."""
    from mmseg.core.evaluation.metrics import total_intersect_and_union
    ret = eval_metrics(results, gt_seg_maps, num_classes, ignore_index,
                       metrics, nan_to_num, label_map, reduce_zero_label, beta)
    if any(m in metrics for m in ['mPrecision', 'mRecall', 'PID']):
        add = total_precision_recall_pid(results, gt_seg_maps, ignore_index)
        ret.update(add)
    return ret



