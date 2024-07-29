# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean

from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from ..builder import ROTATED_HEADS, build_loss
from .rotated_anchor_free_head import RotatedAnchorFreeHead
from mmrotate.core.bbox.transforms import norm_angle
INF = 1e8

import matplotlib.pyplot as plt
import numpy as np

def plot_oriented_bbox(ax, bbox, edgecolor='r'):
    """
    Plot an oriented bounding box.
    
    bbox: [x_center, y_center, width, height, angle]
    """
    x_center, y_center, width, height, angle_rad = bbox
    
    # Get the 4 corners of the bounding box
    corners = np.array([
        [-width / 2, -height / 2],
        [width / 2, -height / 2],
        [width / 2, height / 2],
        [-width / 2, height / 2]
    ])
    
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # Rotate the corners
    rotated_corners = corners @ rotation_matrix.T
    
    # Translate the corners to the center
    translated_corners = rotated_corners + np.array([x_center, y_center])
    
    # Plot the bounding box
    ax.plot(*np.vstack([translated_corners, translated_corners[0]]).T, edgecolor)

def plot_horizontal_bbox(ax, bbox, edgecolor='b'):
    """
    Plot a horizontal bounding box.
    
    bbox: [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = bbox
    corners = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ])
    
    # Plot the bounding box
    ax.plot(*np.vstack([corners, corners[0]]).T, edgecolor)

def plot_boxes(target, target_projected):
    # Create plot
    fig, ax = plt.subplots()

    for i in range(len(target)):
        plot_oriented_bbox(ax, target[i], edgecolor='r')
        plot_horizontal_bbox(ax, target_projected[i], edgecolor='b')

    # Set limits and show plot
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_aspect('equal')
    plt.savefig('boxes.png')


def plot_str_tensor(str_tensor, center_x, center_y, bbox):
    # Create plot
    fig, ax = plt.subplots()

    # Plot the oriented bounding box
    plot_oriented_bbox(ax, bbox, edgecolor='r')

    # Plot the structure tensor
    for i in range(len(str_tensor)):
        plot_oriented_bbox(ax, [center_x[i], center_y[i], 2 * str_tensor[i, 0], 2 * str_tensor[i, 1], str_tensor[i, 2]], edgecolor='b')

    # Set limits and show plot
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_aspect('equal')
    plt.savefig('str_tensor.png')


def compute_bbox_from_structure_tensor(a, b, c, center_x, center_y):
    """
    Compute the bounding box from the structure tensor components.
    
    a, b, c: Coefficients of the structure tensor.
    center_x, center_y: Center coordinates.
    
    Returns:
    bbox: [x1, y1, x2, y2]
    """
    # Build the structure tensor [[a, c], [c, b]]
    str_tensor = torch.tensor([[a, c], [c, b]])
    
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = torch.linalg.eigh(str_tensor)
    eigvals = eigvals.real  # Since the structure tensor is real, eigenvalues should be real
    eigvecs = eigvecs.real  # Eigenvectors will also be real
    
    # Sort eigenvalues and eigenvectors in descending order
    eigvals_sorted, indices = eigvals.sort(dim=-1, descending=True)
    eigvecs_sorted = eigvecs[:, indices]
    
    # Scale eigenvectors by the square root of the eigenvalues
    scaled_eigvecs = eigvecs_sorted * torch.sqrt(eigvals_sorted)
    
    # Compute the four corners of the bounding box
    corners = torch.tensor([
        [center_x, center_y] + scaled_eigvecs[:, 0].numpy(),
        [center_x, center_y] - scaled_eigvecs[:, 0].numpy(),
        [center_x, center_y] + scaled_eigvecs[:, 1].numpy(),
        [center_x, center_y] - scaled_eigvecs[:, 1].numpy()
    ])
    
    # Get the bounding box coordinates
    x_min = corners[:, 0].min()
    y_min = corners[:, 1].min()
    x_max = corners[:, 0].max()
    y_max = corners[:, 1].max()
    
    return [x_min, y_min, x_max, y_max]

def plot_structure_tensors_with_bboxes(structure_tensors, bboxes, center_x, center_y):
    """
    Plots horizontal bounding boxes and their corresponding structure tensors.
    
    structure_tensors: Tensor of shape [N, 3] containing the a, b, c coefficients of the structure matrix.
    bboxes: Tensor of shape [N, 4] containing the horizontal bounding boxes in the format [x1, y1, x2, y2].
    center_x: Tensor of shape [N] containing the x coordinates of the centers.
    center_y: Tensor of shape [N] containing the y coordinates of the centers.
    """
    N = structure_tensors.size(0)
    
    fig, ax = plt.subplots()
    
    for i in range(N):
        # Get the structure tensor coefficients
        a, b, c = structure_tensors[i]
        
        # Get the corresponding bounding box and center coordinates
        bbox = bboxes[i]
        cx, cy = center_x[i], center_y[i]
        
        # Plot the horizontal bounding box
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        
        # Compute the bounding box for the structure tensor
        #bbox_tensor = compute_bbox_from_structure_tensor(a, b, c, cx, cy)
        #x1t, y1t, x2t, y2t = bbox_tensor
        #rect_tensor = plt.Rectangle((x1t, y1t), x2t - x1t, y2t - y1t, edgecolor='red', facecolor='none')
        #ax.add_patch(rect_tensor)
        
        # Plot the eigenvectors multiplied by the square root of the eigenvalues
        str_tensor = torch.tensor([[a, c], [c, b]])
        eigvals, eigvecs = torch.linalg.eig(str_tensor)
        eigvals = eigvals.real
        eigvecs = eigvecs.real
        
        eigvals_sorted, indices = eigvals.sort(dim=-1, descending=True)
        eigvecs_sorted = eigvecs[:, indices]
        
        for j in range(2):
            vec = eigvecs_sorted[:, j] * torch.sqrt(eigvals_sorted[j])
            ax.arrow(cx, cy, vec[0].item(), vec[1].item(), head_width=0.1, head_length=0.1, fc='red', ec='red')
            ax.arrow(cx, cy, -vec[0].item(), -vec[1].item(), head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    ax.set_aspect('equal')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Horizontal Bounding Boxes and Structure Tensors')
    plt.savefig('structure_tensors_with_bboxes.png')


@ROTATED_HEADS.register_module()
class H2RBoxV2PHeadStuctureTensorBackToOriginal(RotatedAnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.
    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        separate_angle (bool): If true, angle prediction is separated from
            bbox regression loss. Default: False.
        scale_angle (bool): If true, add scale to angle pred branch. Default: True.
        h_bbox_coder (dict): Config of horzional bbox coder, only used when separate_angle is True.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_angle (dict): Config of angle loss, only used when separate_angle is True.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
    Example:
        >>> self = RotatedFCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, angle_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 scale_angle=False,
                 square_cls=[],
                 resize_cls=[],
                 angle_coder=dict(
                     type='PSCCoder', 
                     angle_version='le90', 
                     dual_freq=False, 
                     thr_mod=0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_ss_symmetry=dict(
                     type='SmoothL1Loss', loss_weight=0.2, beta=0.1),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.is_scale_angle = scale_angle
        self.square_cls = square_cls
        self.resize_cls = resize_cls
        self.angle_coder = build_bbox_coder(angle_coder)
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_ss_symmetry = build_loss(loss_ss_symmetry)

    def rbox2hbox(self, rbboxes):
        w = rbboxes[:, 2::5]
        h = rbboxes[:, 3::5]
        a = rbboxes[:, 4::5].detach()
        cosa = torch.cos(a).abs()
        sina = torch.sin(a).abs()
        hbbox_w = cosa * w + sina * h
        hbbox_h = sina * w + cosa * h
        dx = rbboxes[..., 0]
        dy = rbboxes[..., 1]
        dw = hbbox_w.reshape(-1)
        dh = hbbox_h.reshape(-1)
        x1 = dx - dw / 2
        y1 = dy - dh / 2
        x2 = dx + dw / 2
        y2 = dy + dh / 2
        return torch.stack((x1, y1, x2, y2), -1)

    def nested_projection(self, pred, target):
        target_xy1 = target[..., 0:2] - target[..., 2:4] / 2
        target_xy2 = target[..., 0:2] + target[..., 2:4] / 2
        target_projected = torch.cat((target_xy1, target_xy2), -1)
        pred_xy = pred[..., 0:2]
        pred_wh = pred[..., 2:4]
        da = pred[..., 4] - target[..., 4]
        cosa = torch.cos(da).abs()
        sina = torch.sin(da).abs()
        pred_wh = torch.matmul(
            torch.stack((cosa, sina, sina, cosa), -1).view(*cosa.shape, 2, 2),
            pred_wh[..., None])[..., 0]
        pred_xy1 = pred_xy - pred_wh / 2
        pred_xy2 = pred_xy + pred_wh / 2
        pred_projected = torch.cat((pred_xy1, pred_xy2), -1)
        return pred_projected, target_projected

    def rotate_str_tensor(self, str_tensor, theta):
        '''
        Rotate the str_tensor of shape [N, 3], where the last dimension contains the coefficients [a, b, c] of the structure tensor
        [[a, c], [c, b]] by an angle theta in radians.
        '''
        # Define the rotation matrix R_theta
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        R_theta = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]], device=str_tensor.device)
        
        # Create structure matrix of shape [N, 2, 2] from str_tensor: [N, 3]
        str_matrix = torch.stack([str_tensor[:, 0], str_tensor[:, 2], str_tensor[:, 2], str_tensor[:, 1]], dim=1).reshape(-1, 2, 2)

        # Rotate the structure matrix
        R_theta_T = R_theta.transpose(0, 1)  # Transpose of the rotation matrix
        rotated_matrix = torch.matmul(R_theta.unsqueeze(0), torch.matmul(str_matrix, R_theta_T.unsqueeze(0)))

        # Convert the rotated structure matrix back to [a, b, c] format
        rotated_str_tensor = torch.stack([rotated_matrix[:, 0, 0], rotated_matrix[:, 1, 1], rotated_matrix[:, 0, 1]], dim=1)

        return rotated_str_tensor
    
    def flip_str_tensor_vertical(self, str_tensor):
        '''
        Vertically flip the str_tensor of shape [N, 3], where the last dimension contains the coefficients [a, b, c] of the structure tensor
        [[a, c], [c, b]].
        '''
        # Create the vertical flip matrix F_vertical
        F_vertical = torch.tensor([[1, 0], [0, -1]], dtype=str_tensor.dtype, device=str_tensor.device)

        # Create structure matrix of shape [N, 2, 2] from str_tensor: [N, 3]
        str_matrix = torch.stack([str_tensor[:, 0], str_tensor[:, 2], str_tensor[:, 2], str_tensor[:, 1]], dim=1).reshape(-1, 2, 2)

        # Perform the vertical flip: F_vertical @ str_matrix @ F_vertical
        flipped_str_matrix = torch.matmul(torch.matmul(F_vertical, str_matrix), F_vertical)

        # Return the flipped structure matrix as 3 values a, b, c: [N, 2, 2] --> [N, 3]
        return torch.stack([flipped_str_matrix[:, 0, 0], flipped_str_matrix[:, 1, 1], flipped_str_matrix[:, 0, 1]], dim=1)

    
    def str_tensor_to_obb(self, center, str_tensor, angle_range='le90'):
        # Extract individual components from str_tensor
        a = str_tensor[:, 0]  # shape [N]
        b = str_tensor[:, 1]  # shape [N]
        c = str_tensor[:, 2]  # shape [N]

        # Construct the structure tensors
        structure_tensors = torch.stack([
            torch.stack([a, c], dim=-1),  # shape [N, 2]
            torch.stack([c, b], dim=-1)   # shape [N, 2]
        ], dim=-2)  # shape [N, 2, 2]

        # Calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(structure_tensors)  # eigenvalues shape [N, 2], eigenvectors shape [N, 2, 2]

        # Extract the real parts of the eigenvalues and eigenvectors
        eigenvalues = torch.abs(eigenvalues.real)  # shape [N, 2]
        eigenvectors = eigenvectors.real  # shape [N, 2, 2]

        scale = 1.0  # Scale factor for the width and height
        w = scale * eigenvalues[:, 0]  # shape [N]
        h = scale * eigenvalues[:, 1]  # shape [N]
        a = torch.atan2(eigenvectors[:, 1, 1], eigenvectors[:, 0, 1])  # shape [N]
        a = norm_angle(a, angle_range)

        # Construct the obb tensor [center_x, center_y, width, height, angle]
        obb = torch.stack([center[:, 0], center[:, 1], 2 * w, 2 * h, a], dim=-1)  # shape [N, 5]

        return obb

    
    def obb_to_circumscribed_hbb(self, obb):
        cx, cy, w, h, a = obb[:, 0], obb[:, 1], obb[:, 2], obb[:, 3], obb[:, 4]
        
        # Compute the corners relative to the center and dimensions
        corners = torch.stack([
            torch.stack([-w/2, -h/2], dim=1),  # bottom-left
            torch.stack([w/2, -h/2], dim=1),   # bottom-right
            torch.stack([w/2, h/2], dim=1),    # top-right
            torch.stack([-w/2, h/2], dim=1)    # top-left
        ], dim=1)

        # Create the rotation matrix
        rotation_matrix = torch.stack([
            torch.cos(a), -torch.sin(a),
            torch.sin(a), torch.cos(a)
        ], dim=-1).view(-1, 2, 2)  # shape [N, 2, 2]

        # Rotate the corners
        rotated_corners = torch.matmul(corners, rotation_matrix.transpose(1, 2))  # shape [N, 4, 2]

        # Translate the corners to the center point
        rotated_corners[:, :, 0] += cx.unsqueeze(1)  # shape [N, 4]
        rotated_corners[:, :, 1] += cy.unsqueeze(1)  # shape [N, 4]

        # Find the min and max x and y coordinates
        min_x, _ = torch.min(rotated_corners[:, :, 0], dim=1)  # shape [N]
        max_x, _ = torch.max(rotated_corners[:, :, 0], dim=1)  # shape [N]
        min_y, _ = torch.min(rotated_corners[:, :, 1], dim=1)  # shape [N]
        max_y, _ = torch.max(rotated_corners[:, :, 1], dim=1)  # shape [N]

        # Calculate the new width and height
        w_r = max_x - min_x  # shape [N]
        h_r = max_y - min_y  # shape [N]

        # Construct the circumscribed horizontal bounding boxes [cx, cy, w_r, h_r]
        hbox = torch.stack([cx, cy, w_r, h_r], dim=1)  # shape [N, 4]

        return hbox

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_angle = nn.Conv2d(
            self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.is_scale_angle:
            self.scale_angle = Scale(1.0)

    def forward(self, feats):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                angle_preds (list[Tensor]): Box angle for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.
        Returns:
            tuple: scores for each class, bbox predictions, angle predictions \
                and centerness predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)


        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        angle_pred = self.conv_angle(reg_feat)

        if self.is_scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()
        
        return cls_score, bbox_pred, angle_pred, centerness

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            angle_preds (list[Tensor]): Box angle for each scale level, \
                each is a 4D-tensor, the channel number is num_points * 1.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        assert len(cls_scores) == len(bbox_preds) \
               == len(angle_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets, angle_targets, bid_targets = self.get_targets(
            all_level_points, gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, self.angle_coder.encode_size)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        flatten_bid_targets = torch.cat(bid_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        pos_labels = flatten_labels[pos_inds]
        pos_bid_targets = flatten_bid_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]

            #pos_decoded_angle_preds = self.angle_coder.decode(
            #    pos_angle_preds, keepdim=True).detach()
            
            square_mask = torch.zeros_like(pos_labels, dtype=torch.bool)
            for c in self.square_cls:
                square_mask = torch.logical_or(square_mask, pos_labels == c)
            #pos_decoded_angle_preds[square_mask] = 0
            target_mask = torch.abs(
                pos_angle_targets[square_mask]) < torch.pi / 4
            pos_angle_targets[square_mask] = torch.where(
                target_mask, 0, -torch.pi / 2)
            
            #centers = pos_points + pos_bbox_preds[:, :2]
            angles_pred = self.str_tensor_to_obb(pos_points, pos_angle_preds)[..., 4].detach()
            angles_pred[square_mask] = 0
            
            # Extract angle from predicted structure tensors and add to bbox predictions
            pos_bbox_preds = torch.cat([pos_bbox_preds, angles_pred[..., None]], dim=-1)
            pos_bbox_targets = torch.cat([pos_bbox_targets, pos_angle_targets], dim=-1)

            # Decode bbox predictions and targets (from offsets to coordinates)
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_bbox_targets = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)

            loss_bbox = self.loss_bbox(
                    *self.nested_projection(pos_decoded_bbox_preds,
                                            pos_decoded_bbox_targets),
                    weight=pos_centerness_targets,
                    avg_factor=centerness_denorm)
            
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
                        
            # Self-supervision
            # Aggregate targets of the same bbox based on their identical bid
            bid, idx = torch.unique(pos_bid_targets, return_inverse=True)
            compacted_bid_targets = torch.empty_like(bid).index_reduce_(
                0, idx, pos_bid_targets, 'mean', include_self=False)
            
            # Generate a mask to eliminate bboxes without correspondence
            # (bcnt is supposed to be 2, for original and transformed)
            _, bidx, bcnt = torch.unique(
                compacted_bid_targets.long(),
                return_inverse=True,
                return_counts=True)
            bmsk = bcnt[bidx] == 2


            # The reduce all sample points of each object
            ss_info = img_metas[0]['ss']
            rot = ss_info[1]
            pair_angle_preds = torch.empty(
                *bid.shape, pos_angle_preds.shape[-1],
                device=bid.device).index_reduce_(
                    0, idx, pos_angle_preds, 'mean',
                    include_self=False)[bmsk].view(-1, 2,
                                                pos_angle_preds.shape[-1])

            pair_labels = torch.empty(
                *bid.shape, dtype=pos_labels.dtype,
                device=bid.device).index_reduce_(
                    0, idx, pos_labels, 'mean',
                    include_self=False)[bmsk].view(-1, 2)[:, 0]
            square_mask = torch.zeros_like(pair_labels, dtype=torch.bool)
            for c in self.square_cls:
                square_mask = torch.logical_or(square_mask, pair_labels == c)

            #angle_ori_preds = self.angle_coder.decode(
            #    pair_angle_preds[:, 0], keepdim=True)
            #angle_trs_preds = self.angle_coder.decode(
            #    pair_angle_preds[:, 1], keepdim=True)
            #angle_ori_preds = pair_angle_preds[:, 0]
            #angle_trs_preds = pair_angle_preds[:, 1]
            B, N, _ = pair_angle_preds.shape
            centers = torch.ones(B, 2, device=pair_angle_preds.device)
            angle_ori_preds = self.str_tensor_to_obb(centers, pair_angle_preds[:, 0])[..., 4]
            angle_trs_preds = self.str_tensor_to_obb(centers, pair_angle_preds[:, 1])[..., 4]

            if len(pair_angle_preds):
                if ss_info[0] == 'rot':
                    d_ang = angle_trs_preds - angle_ori_preds - rot
                else:
                    d_ang = angle_ori_preds + angle_trs_preds
                d_ang = (d_ang + torch.pi / 2) % torch.pi - torch.pi / 2
                d_ang[square_mask] = 0
                loss_ss = self.loss_ss_symmetry(d_ang, torch.zeros_like(d_ang))
            else:
                loss_ss = pair_angle_preds.sum()
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            loss_ss = pos_bbox_preds.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_ss=loss_ss)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
                concat_lvl_angle_targets (list[Tensor]): Angle targets of \
                    each level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, \
            angle_targets_list, bid_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        angle_targets_list = [
            angle_targets.split(num_points, 0)
            for angle_targets in angle_targets_list
        ]
        bid_targets_list = [
            bid_targets.split(num_points, 0)
            for bid_targets in bid_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        concat_lvl_bid_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            bid_targets = torch.cat(
                [bid_targets[i] for bid_targets in bid_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
            concat_lvl_bid_targets.append(bid_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_angle_targets, concat_lvl_bid_targets)

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression, classification and angle targets for a single
        image."""
        gt_bids = gt_bboxes[:, 5]
        gt_bboxes = gt_bboxes[:, :5]

        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1)), \
                   gt_bboxes.new_zeros((num_points, ))

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                    inside_gt_bbox_mask)

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        angle_targets = gt_angle[range(num_points), min_area_inds]
        bid_targets = gt_bids[min_area_inds]

        return labels, bbox_targets, angle_targets, bid_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)
        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   angle_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            angle_preds (list[Tensor]): Box angle for each scale level \
                with shape (N, num_points * 1, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the 6-th
                column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            angle_pred_list = [
                angle_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 angle_pred_list,
                                                 centerness_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           angle_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            angle_preds (list[Tensor]): Box angle for a single scale level \
                with shape (N, num_points * 1, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 1, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 6), where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the
                6-th column is a score between 0 and 1.
        """

        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, angle_pred, centerness, points in zip(
                cls_scores, bbox_preds, angle_preds, centernesses,
                mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, self.angle_coder.encode_size)
            
            #angle_pred = self.angle_coder.decode(angle_pred, keepdim=True)
            # Compute angle from structure tensor
            pred_obb = self.str_tensor_to_obb(points, angle_pred)
            angle_pred = pred_obb[:, 4].unsqueeze(-1)

            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = self.bbox_coder.decode(
                points, bbox_pred, max_shape=img_shape)
            #bboxes[:, 4] = angle_pred.squeeze(-1)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        for id in self.square_cls:
            det_bboxes[det_labels == id, 4] = 0
        for id in self.resize_cls:
            det_bboxes[det_labels == id, 2:4] *= 0.85
        return det_bboxes, det_labels

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centerness'))
    def refine_bboxes(self, cls_scores, bbox_preds, angle_preds, centernesses):
        """This function will be used in S2ANet, whose num_anchors=1."""
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        # device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            angle_pred = angle_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 4)
            angle_pred = angle_pred.permute(0, 2, 3, 1)
            angle_pred = angle_pred.reshape(num_imgs, -1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=-1)

            points = mlvl_points[lvl]

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(points, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list
