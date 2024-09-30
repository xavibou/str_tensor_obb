# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmdet.datasets.api_wrappers import COCO

from mmrotate.core import poly2obb_np
from .builder import ROTATED_DATASETS
from .dota import DOTADataset


@ROTATED_DATASETS.register_module()
class MagentineDataset(DOTADataset):
    """ICDAR text dataset for rotated object detection (Support ICDAR2015 and
    ICDAR2017)."""
    
    CLASSES = ('well', 'results')
    PALETTE = [(165, 42, 42), (189, 183, 107)]

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
                
                # assign class to the unique class 'text'
                data_info['ann']['labels'] = np.zeros_like(data_info['ann']['labels'])

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