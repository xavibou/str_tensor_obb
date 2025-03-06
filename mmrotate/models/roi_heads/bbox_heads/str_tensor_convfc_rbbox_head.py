# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer
from mmrotate.core import multiclass_nms_rotated

from ...builder import ROTATED_HEADS
from .rotated_bbox_head import RotatedBBoxHead
from mmrotate.core import build_bbox_coder


@ROTATED_HEADS.register_module()
class STRotatedConvFCBBoxHead(RotatedBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg

    Args:
        num_shared_convs (int, optional): number of ``shared_convs``.
        num_shared_fcs (int, optional): number of ``shared_fcs``.
        num_cls_convs (int, optional): number of ``cls_convs``.
        num_cls_fcs (int, optional): number of ``cls_fcs``.
        num_reg_convs (int, optional): number of ``reg_convs``.
        num_reg_fcs (int, optional): number of ``reg_fcs``.
        conv_out_channels (int, optional): output channels of convolution.
        fc_out_channels (int, optional): output channels of fc.
        conv_cfg (dict, optional): Config of convolution.
        norm_cfg (dict, optional): Config of normalization.
        init_cfg (dict, optional): Config of initialization.
    """

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 num_angle_convs=0,  
                 num_angle_fcs=0,    # New parameter
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 angle_coder=dict(
                     type='STCoder',
                     angle_version='le90'),
                 loss_angle=dict(type='L1Loss', loss_weight=0.05),
                 separate_angle=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(STRotatedConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs + 
                num_angle_convs + num_angle_fcs > 0)  # Updated assertion
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_angle_convs = num_angle_convs
        self.num_angle_fcs = num_angle_fcs    
        self.angle_coder = build_bbox_coder(angle_coder)
        self.coding_len = self.angle_coder.encode_size
        self.loss_angle = loss_angle
        self.separate_angle = separate_angle

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        # add angle specific branch
        self.angle_convs, self.angle_fcs, self.angle_last_dim = \
            self._add_conv_fc_branch(
                self.num_angle_convs, self.num_angle_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area
            if self.num_angle_fcs == 0:
                self.angle_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (5 if self.reg_class_agnostic else 5 * self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)
                
            # Add angle prediction layer
            out_dim_angle = (3 if self.reg_class_agnostic else 3 * self.num_classes)
            self.fc_angle = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.angle_last_dim,
                out_features=out_dim_angle)  # 3 channels for angle prediction

            if init_cfg is None:
                self.init_cfg += [
                    dict(
                        type='Xavier',
                        layer='Linear',
                        override=[
                            dict(name='shared_fcs'),
                            dict(name='cls_fcs'),
                            dict(name='reg_fcs'),
                            dict(name='angle_fcs')  # Add initialization for angle layers
                        ])
                ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        x_angle = x 

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        
        for conv in self.angle_convs:
            x_angle = conv(x_angle)
        if x_angle.dim() > 2:
            if self.with_avg_pool:
                x_angle = self.avg_pool(x_angle)
            x_angle = x_angle.flatten(1)
        for fc in self.angle_fcs:
            x_angle = self.relu(fc(x_angle))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        angle_pred = self.fc_angle(x_angle) 

        return cls_score, bbox_pred, angle_pred
    
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             angle_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function.

        Args:
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            labels (torch.Tensor): Shape (n*bs, ).
            label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
            bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
            bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            
            if pos_inds.any():    
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    if not self.separate_angle:
                        _, _, angle_pred = self.angle_coder.decode(angle_pred)
                        bbox_pred = torch.cat((bbox_pred[:, :4], angle_pred[:, None]), dim=1)
                        pos_bbox_pred = bbox_pred.view(
                            bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                    else:
                        pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                        bbox_weights[:, -1] = 0 # set angle weight in bbox to 0
                        
                        # Decode bbox and obtain width, height and angle
                        #decoded_boxes = self.bbox_coder.decode(rois[:, 1:], bbox_targets)
                        #wh = decoded_boxes[:, 2:-1]
                        wh = rois[:,3:5] - rois[:,1:3]
                        #breakpoint()
                        angle_targets = self.angle_coder.encode(
                                                            bbox_targets[:, -1, None], 
                                                            wh[:, 0, None], 
                                                            wh[:, 1, None]
                                                            )
                        losses['angle_loss'] = self.loss_bbox(
                            angle_pred[pos_inds.type(torch.bool)],
                            angle_targets[pos_inds.type(torch.bool)],
                            bbox_weights[pos_inds.type(torch.bool)][:,:self.coding_len],
                            avg_factor=bbox_targets.size(0),
                            reduction_override=reduction_override)
                else:
                    if not self.separate_angle:
                        _, _, angle_pred = self.angle_coder.decode(angle_pred.view(-1, self.angle_coder.encode_size)).view(bbox_pred.size(0), -1,1)
                        pos_angle_pred = angle_pred.view(bbox_pred.size(0), -1,1)[pos_inds.type(torch.bool),labels[pos_inds.type(torch.bool)]]
                        pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,5)[pos_inds.type(torch.bool),labels[pos_inds.type(torch.bool)]]
                        pos_bbox_pred = torch.cat((pos_bbox_pred[:, :4], pos_angle_pred), dim=1)
                    else:
                        pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,5)[pos_inds.type(torch.bool),labels[pos_inds.type(torch.bool)]]
                        bbox_weights[:, -1] = 0 # set angle weight in bbox to 0
                        
                        # Decode bbox and obtain width, height and angle
                        #decoded_boxes = self.bbox_coder.decode(rois[:, 1:], bbox_targets)
                        #wh = decoded_boxes[:, 2:-1]
                        wh = rois[:,3:5]
                        angle_targets = self.angle_coder.encode(
                                                            bbox_targets[:, -1, None], 
                                                            wh[:, 0, None], 
                                                            wh[:, 1, None]
                                                            )
                        pos_angle_pred = angle_pred.view(bbox_pred.size(0), -1,self.angle_coder.encode_size)[pos_inds.type(torch.bool),labels[pos_inds.type(torch.bool)]]
                        # losses['angle_loss'] = self.loss_bbox( angle_pred[pos_inds.type(torch.bool)],angle_targets[pos_inds.type(torch.bool)],bbox_weights[pos_inds.type(torch.bool)][:,:self.coding_len],avg_factor=bbox_targets.size(0),reduction_override=reduction_override)
                        losses['angle_loss'] = self.loss_bbox( 
                            pos_angle_pred,
                            angle_targets[pos_inds.type(torch.bool)],
                            bbox_weights[pos_inds.type(torch.bool)][:,:self.coding_len],
                            avg_factor=bbox_targets.size(0),
                            reduction_override=reduction_override)
                        
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   angle_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 6) and last
                dimension 6 represent (cx, cy, w, h, a, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            #_, _, angle_pred = self.angle_coder.decode(angle_pred)
            #bbox_pred = torch.cat((bbox_pred[:, :4], angle_pred[:, None]), dim=1)
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
            #bbox_pred = bbox_pred.view(bbox_results['angle_pred'].size(0), -1)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = bboxes.view(bboxes.size(0), -1, 5)
            bboxes[..., :4] = bboxes[..., :4] / scale_factor
            bboxes = bboxes.view(bboxes.size(0), -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms_rotated(
                bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_labels

@ROTATED_HEADS.register_module()
class STRotatedShared2FCBBoxHead(STRotatedConvFCBBoxHead):
    """Shared2FC RBBox head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(STRotatedShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            num_angle_convs=0,  
            num_angle_fcs=0,    
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)



@ROTATED_HEADS.register_module()
class STRotatedKFIoUShared2FCBBoxHead(STRotatedConvFCBBoxHead):
    """KFIoU RoI head."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(STRotatedKFIoUShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            num_angle_convs=0,  
            num_angle_fcs=0,    
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             angle_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function."""
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.

            if pos_inds.any():
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                    pos_bbox_pred_decode = bbox_pred_decode.view(
                        bbox_pred_decode.size(0), 5)[pos_inds.type(torch.bool)]
                else:
            
                    _, _, angle_pred = self.angle_coder.decode(angle_pred.view(-1, self.angle_coder.encode_size))
                    pos_angle_pred = angle_pred.view(bbox_pred.size(0), -1,1)[pos_inds.type(torch.bool),labels[pos_inds.type(torch.bool)]]
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,5)[pos_inds.type(torch.bool),labels[pos_inds.type(torch.bool)]]
                    pos_bbox_pred = torch.cat((pos_bbox_pred[:, :4], pos_angle_pred), dim=1)

                    pos_bbox_pred_decode = self.bbox_coder.decode(rois[pos_inds.type(torch.bool), 1:], pos_bbox_pred)
                    bbox_targets_decode = self.bbox_coder.decode(rois[pos_inds.type(torch.bool), 1:], bbox_targets[pos_inds.type(torch.bool)])
                    


                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    pred_decode=pos_bbox_pred_decode,
                    targets_decode=bbox_targets_decode,
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses