# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import tempfile
import time
import warnings
import zipfile
from collections import defaultdict
from functools import partial

import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np, eval_angle_error
from .builder import ROTATED_DATASETS
from mmdet.datasets.api_wrappers import COCO

from mmrotate.core import poly2obb_np
from .builder import ROTATED_DATASETS
from .dota import DOTADataset

from collections import OrderedDict
from mmcv import print_log


@ROTATED_DATASETS.register_module()
class MagentineDataset(DOTADataset):
    """ICDAR text dataset for rotated object detection (Support ICDAR2015 and
    ICDAR2017)."""
    
    CLASSES = ('well', 'results')
    #PALETTE = [(165, 42, 42), (189, 183, 107)]

    # palette red and blue
    PALETTE = [(165, 42, 42), (0, 0, 255)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 select_first_k=-1,
                 **kwargs):
        self.version = version
        self.select_first_k = select_first_k

        super().__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.
        Returns:
            list[dict]: Annotation info.
        """
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        count = 0
        for i in self.img_ids:
            data_info = {}
            info = self.coco.load_imgs([i])[0]
            data_info['filename'] = info['file_name']
            data_info['ann'] = {}
            img_id = info['id']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            gt_bboxes = []
            gt_labels = []
            gt_polygons = []
            for ann in ann_info:
                if ann.get('ignore', False):
                    continue
                x1, y1, w, h = ann['bbox']
                if ann['area'] <= 0 or w < 1 or h < 1:
                    continue
                if ann['category_id'] not in self.cat_ids:
                    continue
                try:
                    poly = np.array(ann['segmentation'][0], dtype=np.float32)
                    x, y, w, h, a = poly2obb_np(poly,self.version)
                except:  # noqa: E722
                    continue
                gt_bboxes.append([x, y, w, h, a])
                gt_labels.append(ann['category_id'])
                gt_polygons.append(ann['segmentation'][0])

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)
                data_info['ann']['bboxes_ignore'] = np.zeros(
                    (0, 5), dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array(
                    [], dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.zeros(
                    (0, 8), dtype=np.float32)
                
                data_info['ann']['labels'] = data_info['ann']['labels'] - 1

                data_infos.append(data_info)
            count = count + 1
            if count > self.select_first_k and self.select_first_k > 0:
                break
        return data_infos
    
    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []

        for i, data_info in enumerate(self.data_infos):
            if (not self.filter_empty_gt
                    or data_info['ann']['labels'].size > 0):
                valid_inds.append(i)
        return valid_inds
    
    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=[0.5],#[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                 scale_ranges=None,
                 nproc=12):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            use_07_metric (bool): Whether to use the voc07 metric.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_rbbox_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    #use_07_metric=use_07_metric,
                    dataset=self.CLASSES,
                    logger=logger,
                    nproc=nproc)
                mean_aps.append(mean_ap)
    
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif metric == 'recall':
            raise NotImplementedError

        return eval_results
