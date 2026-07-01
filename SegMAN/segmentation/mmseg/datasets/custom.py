# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose, LoadAnnotations


@DATASETS.register_module()
class CustomDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/en/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
        gt_seg_map_loader_cfg (dict, optional): build LoadAnnotations to
            load gt for evaluation, load from disk by default. Default: None.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 file_client_args=dict(backend='disk')):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.gt_seg_map_loader = LoadAnnotations(
        ) if gt_seg_map_loader_cfg is None else LoadAnnotations(
            **gt_seg_map_loader_cfg)

        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient.infer_client(self.file_client_args)

        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        else:
            for img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        raise NotImplementedError

    def get_gt_seg_map_by_idx(self, index):
        """Get one ground truth segmentation map for evaluation."""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        return results['gt_semantic_seg']

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            yield results['gt_semantic_seg']

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)
            pre_eval_results.append(
                intersect_and_union(
                    pred,
                    seg_map,
                    len(self.CLASSES),
                    self.ignore_index,
                    # as the labels has been converted when dataset initialized
                    # in `get_palette_for_custom_classes ` this `label_map`
                    # should be `dict()`, see
                    # https://github.com/open-mmlab/mmsegmentation/issues/1415
                    # for more ditails
                    label_map=dict(),
                    reduce_zero_label=self.reduce_zero_label))

        return pre_eval_results

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(class_names).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = class_names.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                # Get random state before set seed, and restore
                # random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        # allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        allowed_metrics = ['mIoU', 'mDice', 'mAcc', 'mPrecision', 'mRecall', 'PID']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=dict(),
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0
        if 'mPrecision' in ret_metrics_summary:
            eval_results['mPrecision'] = ret_metrics_summary['mPrecision'] / 100.0
            print_log(f"mPrecision: {ret_metrics_summary['mPrecision']:.2f}", logger)
        if 'mRecall' in ret_metrics_summary:
            eval_results['mRecall'] = ret_metrics_summary['mRecall'] / 100.0
            print_log(f"mRecall: {ret_metrics_summary['mRecall']:.2f}", logger)
        if 'PID' in ret_metrics_summary:
            eval_results['PID'] = ret_metrics_summary['PID'] / 100.0
            print_log(f"PID: {ret_metrics_summary['PID']:.2f}", logger)
        
        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results
    
    # def evaluate(self,
    #              results,
    #              metric='mIoU',
    #              logger=None,
    #              gt_seg_maps=None,
    #              **kwargs):
    #     """Evaluate the dataset.
    
    #     Args:
    #         results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
    #              results or predict segmentation map for computing evaluation
    #              metric.
    #         metric (str | list[str]): Metrics to be evaluated.
    #         logger (logging.Logger | None | str): Logger used for printing.
    #         gt_seg_maps (generator[ndarray]): Custom gt seg maps as input.
    
    #     Returns:
    #         dict[str, float]: Evaluation results.
    #     """
    #     if isinstance(metric, str):
    #         metric = [metric]
    #     # ✅ 支持自定义指标
    #     allowed_metrics = ['mIoU', 'mDice', 'mAcc', 'mPrecision', 'mRecall', 'PID']
    #     if not set(metric).issubset(set(allowed_metrics)):
    #         raise KeyError(f'metric {metric} is not supported')
    
    #     eval_results = {}
    
    #     # ----------- Step 1: 核心评估计算 -----------
    #     if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(results, str):
    #         if gt_seg_maps is None:
    #             gt_seg_maps = self.get_gt_seg_maps()
    #         num_classes = len(self.CLASSES)
    #         ret_metrics = eval_metrics(
    #             results,
    #             gt_seg_maps,
    #             num_classes,
    #             self.ignore_index,
    #             metric,
    #             label_map=dict(),
    #             reduce_zero_label=self.reduce_zero_label)
    #     else:
    #         ret_metrics = pre_eval_to_metrics(results, metric)
    
    #     class_names = self.CLASSES if self.CLASSES is not None else tuple(range(num_classes))
    
    #     # ----------- Step 2: 计算 summary & per-class -----------
    #     ret_metrics_summary = OrderedDict()
    #     for key, val in ret_metrics.items():
    #         # 这里有的指标是标量（float），有的是数组（per-class）
    #         if isinstance(val, (float, np.floating)):
    #             ret_metrics_summary[key] = np.round(val * 100, 2)
    #         else:
    #             ret_metrics_summary[key] = np.round(np.nanmean(val) * 100, 2)
    
    #     # 保留 per-class 的指标
    #     ret_metrics_class = OrderedDict()
    #     for key, val in ret_metrics.items():
    #         if isinstance(val, np.ndarray):
    #             ret_metrics_class[key] = np.round(val * 100, 2)
    #     ret_metrics_class.update({'Class': class_names})
    #     ret_metrics_class.move_to_end('Class', last=False)
    
    #     # ----------- Step 3: 打印 per-class 表格 -----------
    #     from prettytable import PrettyTable
    #     class_table_data = PrettyTable()
    #     for key, val in ret_metrics_class.items():
    #         # ✅ 跳过单值指标（Precision/Recall/PID）
    #         if isinstance(val, (float, np.floating, np.float32, np.float64)):
    #             continue
    #         # ✅ 确保 val 是可迭代的
    #         if not hasattr(val, '__len__'):
    #             val = [val]
    #         class_table_data.add_column(key, val)
    
    #     # ----------- Step 4: 打印 summary 表格 -----------
    #     summary_table_data = PrettyTable()
    #     for key in ['aAcc', 'mIoU', 'mAcc', 'mDice', 'mPrecision', 'mRecall', 'PID']:
    #         if key in ret_metrics_summary:
    #             summary_table_data.add_column(key, [ret_metrics_summary[key]])
    
    #     print_log('per class results:', logger)
    #     print_log('\n' + class_table_data.get_string(), logger=logger)
    #     print_log('Summary:', logger)
    #     print_log('\n' + summary_table_data.get_string(), logger=logger)
    
    #     # ----------- Step 5: 保存结果到 eval_results -----------
    #     for key, val in ret_metrics_summary.items():
    #         eval_results[key] = val / 100.0
    #     for key, val in ret_metrics_class.items():
    #         # ✅ 跳过标量类型的指标（如 Precision/Recall/PID）
    #         if isinstance(val, (float, np.floating)):
    #             continue
    #         # ✅ 确保 val 是可迭代的
    #         if not hasattr(val, '__len__'):
    #             val = [val]
    #         class_table_data.add_column(key, val)
    
    #     return eval_results

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.
    
        Returns:
            dict[str, float]: 各指标（0~1）结果，如 {'aAcc':0.9962, 'mIoU':0.8566, ...}
        """
        import numpy as np
        from collections import OrderedDict
        from prettytable import PrettyTable
        from mmseg.core import eval_metrics, pre_eval_to_metrics
        from mmcv.utils import print_log
    
        # --- 规范 metric 入参 ---
        if isinstance(metric, str):
            metric = [metric]
        base_allowed = {'mIoU', 'mDice', 'mFscore'}
        base_metrics = [m for m in metric if m in base_allowed] or ['mIoU']
    
        # --- 计算基础指标（支持 pre_eval 与直接结果） ---
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                base_metrics,
                label_map=dict(),
                reduce_zero_label=self.reduce_zero_label)
        else:
            ret = pre_eval_to_metrics(results, base_metrics)
    
        # --- 类名 ---
        if self.CLASSES is None:
            num_classes = ret['IoU'].shape[0] if 'IoU' in ret else 0
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
    
        # --- 可选：由 pre_eval 聚合 Precision / Recall / F1 ---
        mPrecision = mRecall = mF1 = None
        try:
            if len(results) > 0 and isinstance(results[0], (list, tuple)) and len(results[0]) == 4:
                sum_inter = 0
                sum_pred  = 0
                sum_label = 0
                for (ai, _, ap, al) in results:
                    ai = np.asarray(ai)
                    ap = np.asarray(ap)
                    al = np.asarray(al)
                    sum_inter += ai
                    sum_pred  += ap
                    sum_label += al
                eps = 1e-10
                prec_cls = sum_inter / (sum_pred  + eps)
                rec_cls  = sum_inter / (sum_label + eps)
                mPrecision = float(np.nanmean(prec_cls))
                mRecall    = float(np.nanmean(rec_cls))
                mF1        = float(np.nanmean(2 * prec_cls * rec_cls / (prec_cls + rec_cls + eps)))
        except Exception as e:
            print_log(f'[Warn] precision/recall compute failed: {e}', logger)
    
        # --- 汇总（百分比展示，但写回前会/100） ---
        summary = OrderedDict()
        if 'aAcc' in ret:  summary['aAcc'] = float(np.round(float(ret['aAcc']) * 100.0, 2))
        if 'IoU' in ret:   summary['mIoU'] = float(np.round(np.nanmean(ret['IoU']) * 100.0, 2))
        if 'Acc' in ret:   summary['mAcc'] = float(np.round(np.nanmean(ret['Acc']) * 100.0, 2))
        if 'Dice' in ret:  summary['mDice'] = float(np.round(np.nanmean(ret['Dice']) * 100.0, 2))
        if 'Fscore' in ret:summary['mFscore'] = float(np.round(np.nanmean(ret['Fscore']) * 100.0, 2))
        if mPrecision is not None: summary['mPrecision'] = float(np.round(mPrecision * 100.0, 2))
        if mRecall    is not None: summary['mRecall']    = float(np.round(mRecall    * 100.0, 2))
        if mF1        is not None: summary['mF1']        = float(np.round(mF1        * 100.0, 2))
    
        # --- per-class（百分比展示） ---
        per_class = OrderedDict()
        for k in ['IoU', 'Acc', 'Dice', 'Fscore']:
            if k in ret:
                per_class[k] = np.round(np.asarray(ret[k], dtype=np.float64) * 100.0, 2)
        per_class.update({'Class': class_names})
        per_class.move_to_end('Class', last=False)
    
        # --- 打印表格 ---
        class_table = PrettyTable()
        for k, v in per_class.items():
            class_table.add_column(k, v if hasattr(v, '__len__') else [v])
    
        summary_table = PrettyTable()
        for k in ['aAcc', 'mIoU', 'mAcc', 'mDice', 'mFscore', 'mPrecision', 'mRecall', 'mF1']:
            if k in summary:
                summary_table.add_column(k, [summary[k]])
    
        print_log('per class results:', logger)
        print_log('\n' + class_table.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table.get_string(), logger=logger)
    
        # --- 返回为 0~1 的 float（避免 tensor 导致 JSON dump 报错） ---
        eval_results = {}
        for k in ['aAcc','mIoU','mAcc','mDice','mFscore','mPrecision','mRecall','mF1']:
            if k in summary:
                eval_results[k] = summary[k] / 100.0
    
        return eval_results


