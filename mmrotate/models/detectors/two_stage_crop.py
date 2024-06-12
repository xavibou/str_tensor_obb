# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
from mmdet.utils import get_device
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from mmrotate.models.detectors.two_stage_img_split_bridge_tools import *

def resize_bboxes_len6(bboxes_out,scale):
    """Resize bounding boxes with scales."""

    for i in range(len(bboxes_out)):
        box_out=bboxes_out[i]
        
        w_scale = scale
        h_scale = scale
        box_out[:, 0] *= w_scale
        box_out[:, 1] *= h_scale
        box_out[:, 2:4] *= np.sqrt(w_scale * h_scale)

    return bboxes_out

def FullImageCrop(self, imgs, bboxes, labels, patch_shape,
                  gaps,
                  jump_empty_patch=False,
                  mode='train'):
    """
    Args:
        imgs (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        bboxes (list[Tensor]): Each item are the truth boxes for each
            image in [tl_x, tl_y, br_x, br_y] format.
        labels (list[Tensor]): Class indices corresponding to each box
    Returns:
        dict[str, Tensor]: A dictionary of loss components.
    """
    out_imgs = []
    out_bboxes = []
    out_labels = []
    out_metas = []
    device = get_device()
    img_rate_thr = 0.6  # 图片与wins窗口的交并比阈值
    iof_thr = 0.1  # 裁剪后的标签占原标签的比值阈值
    padding_value = [0.0081917211329, -0.004901960784, 0.0055655449953]  # 归一化后的padding值

    if mode == 'train':
        # for i in range(imgs.shape[0]):
        for img, bbox, label in zip(imgs, [bboxes], [labels]):
            p_imgs = []
            p_bboxes = []
            p_labels = []
            p_metas = []
            img = img.cpu()
            # patch
            info = dict()
            info['labels'] = np.array(torch.tensor(label, device='cpu', requires_grad=False))
            info['ann'] = {'bboxes': {}}
            info['width'] = img.shape[2]
            info['height'] = img.shape[1]

            tmp_boxes = torch.tensor(bbox, device='cpu', requires_grad=False)
            info['ann']['bboxes'] = np.array(obb2poly(tmp_boxes, self.version))  # 这里将OBB转换为8点表示形式
            bbbox = info['ann']['bboxes']
            sizes = [patch_shape[0]]
            # gaps=[0]
            windows = get_sliding_window(info, sizes, gaps, img_rate_thr)
            window_anns = get_window_obj(info, windows, iof_thr)
            patchs, patch_infos = crop_and_save_img(info, windows, window_anns,
                                                    img,
                                                    no_padding=True,
                                                    # no_padding=False,
                                                    padding_value=padding_value)

            # 对每张大图分解成的子图集合中的每张子图遍历
            for i, patch_info in enumerate(patch_infos):
                if jump_empty_patch:
                    # 如果该patch中不含有效标签,将其跳过不输出,可在训练时使用

                    if patch_info['labels'] == [-1]:
                        # print('Patch does not contain box.\n')
                        continue
                obj = patch_info['ann']
                if min(obj['bboxes'].shape) == 0:  # 张量为空
                    tmp_boxes = poly2obb(torch.tensor(obj['bboxes']), 'oc')  # oc转化可以处理空张量
                else:
                    tmp_boxes = poly2obb(torch.tensor(obj['bboxes']), self.version)  # 转化回5参数
                p_bboxes.append(tmp_boxes.to(device))
                # p_trunc.append(torch.tensor(obj['trunc'],device=device))  # 是否截断,box全部在win内部时为false
                ## 若box超出win范围则trunc为true
                p_labels.append(torch.tensor(patch_info['labels'], device=device))
                p_metas.append({'x_start': torch.tensor(patch_info['x_start'], device=device),
                                'y_start': torch.tensor(patch_info['y_start'], device=device),
                                'shape': patch_shape, 'trunc': torch.tensor(obj['trunc'], device=device),'img_shape': patch_shape, 'scale_factor': 1})

                patch = patchs[i]
                p_imgs.append(patch.to(device))

            out_imgs.append(p_imgs)
            out_bboxes.append(p_bboxes)
            out_labels.append(p_labels)
            out_metas.append(p_metas)

            #### change for sgdet
            # poly2obb(out_bboxes, self.version)
            return out_imgs, out_bboxes, out_labels, out_metas

    elif mode == 'test':
        p_imgs = []
        p_metas = []
        img = imgs.cpu().squeeze(0)
        # patch
        info = dict()
        info['labels'] = np.array(torch.tensor([], device='cpu'))
        info['ann'] = {'bboxes': {}}
        info['width'] = img.shape[2]
        info['height'] = img.shape[1]

        sizes = [patch_shape[0]]
        # gaps=[0]
        windows = get_sliding_window(info, sizes, gaps, img_rate_thr)
        patchs, patch_infos = crop_img_withoutann(info, windows, img,
                                                  no_padding=False,
                                                  padding_value=padding_value)

        # 对每张大图分解成的子图集合中的每张子图遍历
        for i, patch_info in enumerate(patch_infos):
            p_metas.append({'x_start': torch.tensor(patch_info['x_start'], device=device),
                            'y_start': torch.tensor(patch_info['y_start'], device=device),
                            'shape': patch_shape, 'img_shape': patch_shape, 'scale_factor': 1})

            patch = patchs[i]
            p_imgs.append(patch.to(device))

        out_imgs.append(p_imgs)
        out_metas.append(p_metas)

        return out_imgs, out_metas

    return out_imgs, out_bboxes, out_labels, out_metas

def list2tensor(img_lists):
    '''
    images: list of list of tensor images
    '''
    inputs = []
    for img in img_lists:
        inputs.append(img.cpu())
    inputs = torch.stack(inputs, dim=0).cpu()
    return inputs

def relocate(idx, local_bboxes, patch_meta):
    # 二阶段的bboxes为array
    # put patches' local bboxes to full img via patch_meta
    meta = patch_meta[idx]
    top = meta['y_start']
    left = meta['x_start']

    for in_class in range(len(local_bboxes)):
        local_bboxes_tmp = local_bboxes[in_class]
      
        for i in range(len(local_bboxes_tmp)):
            bbox = local_bboxes_tmp[i]
            # print('local_bboxes[i]:',bbox)
            bbox[0] += left
            bbox[1] += top
            

    return


# 从Global的信息整理成forward格式
def Collect_Global(g_img_infos, img_metas, length_thr):
    g_gt_boxes = []
    g_gt_labels = []

    for idx in range(len(g_img_infos)):
        g_gt_boxes.append(g_img_infos[idx]['gt_box'].squeeze(0))
        g_gt_labels.append(g_img_infos[idx]['labels'].squeeze(0))
        g_img_infos[idx]['img_shape'] = img_metas[0]['img_shape']
        g_img_infos[idx]['pad_shape'] = img_metas[0]['pad_shape']
        g_img_infos[idx]['scale_factor'] = 1.0

    # TODO:标签中会存在负值?
    # 各层按阈值进行标签分配(过滤)
    g_gt_boxes, g_gt_labels=filter_small_ann(g_gt_boxes, g_gt_labels, length_thr, g_img_infos)  # 这里进行标签过滤

    return g_gt_boxes, g_gt_labels, g_img_infos


def filter_small_ann(gt_bboxes, gt_labels, length_thr, g_img_infos=None):
    # 针对resize后图像中长度小于阈值的标签不保留
    # length_thr = 15

    gt_bboxes_global = []
    gt_labels_global = []
    gt_bboxes_global_ignore = []
    gt_labels_global_ignore = []
    # TODO:剔除resize后过小的标签,查看效果
    for gt, (bbox, label) in enumerate(zip(gt_bboxes, gt_labels)):
        # down_ratio = g_img_infos[gt]
        tmp_boxes = gt_bboxes[gt].clone()
        # gt_prepare = tmp_boxes[0].unsqueeze(0)  # 无gt时候补
        # gt_label_prepare = gt_labels[gt][[0]]
        gt_prepare = torch.zeros((0, 5), device=tmp_boxes.device)  # 无符合条件gt时来候补
        gt_label_prepare = torch.tensor([], device=tmp_boxes.device)
        # 根据长度阈值进行筛选
        mask = (tmp_boxes[:, 2] < length_thr) & (tmp_boxes[:, 3] < length_thr)

        tmp_boxes_out_ignore = tmp_boxes[mask]
        keeps_ignore = torch.nonzero(mask).squeeze(1)
        tmp_boxes_out = tmp_boxes[~mask]
        keeps = torch.nonzero(~mask).squeeze(1)

        tmp_labels_out = label[keeps]
        tmp_labels_out_ignore = label[keeps_ignore]

        if len(tmp_boxes_out) < 1:
            gt_bboxes_global.append(gt_prepare)
            gt_labels_global.append(gt_label_prepare)
        else:
            gt_bboxes_global.append(tmp_boxes_out)
            gt_labels_global.append(tmp_labels_out)

        gt_bboxes_global_ignore.append(tmp_boxes_out_ignore)
        gt_labels_global_ignore.append(tmp_labels_out_ignore)
    return gt_bboxes_global, gt_labels_global

@ROTATED_DETECTORS.register_module()
class RotatedTwoStageDetectorCrop(RotatedBaseDetector):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedTwoStageDetectorCrop, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.ite = 0

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 5).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def Test_Patches_Img(self,img,patch_shape,gaps, p_bs, proposals, rescale=False):
        """
        对输入的img按patch_shape,gaps决定的窗口进行切块检测
        """
        # Crop full img into patches
        gt_bboxes=[]
        gt_labels=[]
        device=get_device()
        local_bboxes_lists=[]
        pathches_cls_scores = []
        l_all_box_cls = []
        p_imgs, p_metas = FullImageCrop(self, img, gt_bboxes, gt_labels,
                                        patch_shape=patch_shape,
                                        gaps=gaps, mode='test')
        temp = img.shape[0]
        H =  img.shape[2]
        W = img.shape[3]
        
        img = img.cpu()
        del img
        
        
        for i in range(temp):
            j = 0
            patches = list2tensor(p_imgs[i])  # list to tensor
            patches_meta = p_metas[i]

            # patch batchsize
            while j < len(p_imgs[i]):
                if (j+p_bs) >= len(p_imgs[i]):
                    patch = patches[j:]
                    patch_meta = patches_meta[j:]
                else:
                    patch = patches[j:j + p_bs]
                    patch_meta = patches_meta[j:j + p_bs]  # x_start and y_start

                with torch.no_grad():
                    # fea_l_neck = self.extract_feat(patch)
                    patch=patch.to(device)
                    
                    # print(patch.shape)
                    x = self.extract_feat(patch)
                    
                    ##
                    # torch.cuda.empty_cache()
                    patch = patch.cpu()
                    del patch
                    
                    ##
                    if proposals is None:
                        proposal_list = self.rpn_head.simple_test_rpn(x, patch_meta)
                    else:
                        proposal_list = proposals
                    # 这里输出Local的预测结果
                    # outs_local = self.bbox_head(fea_l_neck)
                    # local的meta设置为1,因为这里未经过缩放
                    local_bbox_list= self.roi_head.simple_test(
                        x, proposal_list, patch_meta, rescale=rescale)
                    
                    
                    torch.cuda.empty_cache()
                    x = tuple(tensor.cpu() for tensor in x)
                    del x
                    
                    
                    # 将每个patch的local boxes放置到大图上对应的位置
                    for idx, res_list in enumerate(local_bbox_list):
                        det_bboxes = res_list
                        relocate(idx, det_bboxes, patch_meta)
                    local_bboxes_lists.append(local_bbox_list)

                    # local_bboxes_lists.append([local_bbox_list,local_label_list])

                j = j+p_bs
                
            
        bbox_list,p_keeps = merge_results_two_stage(local_bboxes_lists,iou_thr=0.4,flag=2)

        out_list = []
        for tt in bbox_list:
            if tt.shape[-1] !=6:
                out_list.append(torch.zeros((0, 6)).cpu().numpy())
            else:
                out_list.append(tt)


        return out_list
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
     #   import pdb; pdb.set_trace()
        
        # self.ite = self.ite+1 
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        
        losses.update(roi_losses)
        # if self.ite > 49:
        # print(losses)
        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""                               

        
        global_shape_h = img.shape[2]
        global_shape_w = img.shape[3]
        
        p_bs_2 = 2
      
        gaps = [200]
        patch_shape = (1024, 1024)
        

        local_bboxes_list= self.Test_Patches_Img(img, patch_shape, gaps, p_bs_2, proposals, rescale=False)
        local_bboxes = [local_bboxes_list]
        all = local_bboxes

        all_nms, all_keeps = merge_results_two_stage(all, iou_thr=0.4,flag=3)

        #####
        all_nms_list = []
        for tt in all_nms:
            if tt.shape[-1] !=6:
                all_nms_list.append(torch.zeros((0, 6)).cpu().numpy())
            else:
                all_nms_list.append(tt)
        # all_nms_list[-1][:,-1] = all_nms_list[-1][:,-1] * 10
        # print(all_nms_list[-1])
        return [all_nms_list]


    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
