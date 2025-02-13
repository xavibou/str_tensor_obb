# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np

import torch
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder
from torch import Tensor
from mmrotate.core.bbox.transforms import norm_angle
from ..builder import ROTATED_BBOX_CODERS


@ROTATED_BBOX_CODERS.register_module()
class CSLCoder(BaseBBoxCoder):
    """Circular Smooth Label Coder.

    `Circular Smooth Label (CSL)
    <https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40>`_ .

    Args:
        angle_version (str): Angle definition.
        omega (float, optional): Angle discretization granularity.
            Default: 1.
        window (str, optional): Window function. Default: gaussian.
        radius (int/float): window radius, int type for
            ['triangle', 'rect', 'pulse'], float type for
            ['gaussian']. Default: 6.
    """

    def __init__(self, angle_version, omega=1, window='gaussian', radius=6):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['oc', 'le90', 'le135']
        assert window in ['gaussian', 'triangle', 'rect', 'pulse']
        self.angle_range = 90 if angle_version == 'oc' else 180
        self.angle_offset_dict = {'oc': 0, 'le90': 90, 'le135': 45}
        self.angle_offset = self.angle_offset_dict[angle_version]
        self.omega = omega
        self.window = window
        self.radius = radius
        self.coding_len = int(self.angle_range // omega)

    def encode(self, angle_targets):
        """Circular Smooth Label Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level
                Has shape (num_anchors * H * W, 1)

        Returns:
            list[Tensor]: The csl encoding of angle offset for each
                scale level. Has shape (num_anchors * H * W, coding_len)
        """

        # radius to degree
        angle_targets_deg = angle_targets * (180 / math.pi)
        # empty label
        smooth_label = torch.zeros_like(angle_targets).repeat(
            1, self.coding_len)
        angle_targets_deg = (angle_targets_deg +
                             self.angle_offset) / self.omega
        # Float to Int
        angle_targets_long = angle_targets_deg.long()

        if self.window == 'pulse':
            radius_range = angle_targets_long % self.coding_len
            smooth_value = 1.0
        elif self.window == 'rect':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.coding_len
            smooth_value = 1.0
        elif self.window == 'triangle':
            base_radius_range = torch.arange(
                -self.radius, self.radius, device=angle_targets_long.device)
            radius_range = (base_radius_range +
                            angle_targets_long) % self.coding_len
            smooth_value = 1.0 - torch.abs(
                (1 / self.radius) * base_radius_range)

        elif self.window == 'gaussian':
            base_radius_range = torch.arange(
                -self.angle_range // 2,
                self.angle_range // 2,
                device=angle_targets_long.device)

            radius_range = (base_radius_range +
                            angle_targets_long) % self.coding_len
            smooth_value = torch.exp(-torch.pow(base_radius_range, 2) /
                                     (2 * self.radius**2))

        else:
            raise NotImplementedError

        if isinstance(smooth_value, torch.Tensor):
            smooth_value = smooth_value.unsqueeze(0).repeat(
                smooth_label.size(0), 1)

        return smooth_label.scatter(1, radius_range, smooth_value)

    def decode(self, angle_preds):
        """Circular Smooth Label Decoder.

        Args:
            angle_preds (Tensor): The csl encoding of angle offset
                for each scale level.
                Has shape (num_anchors * H * W, coding_len)

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1)
        """
        angle_cls_inds = torch.argmax(angle_preds, dim=1)
        angle_pred = ((angle_cls_inds + 0.5) *
                      self.omega) % self.angle_range - self.angle_offset
        return angle_pred * (math.pi / 180)


@ROTATED_BBOX_CODERS.register_module()
class PSCCoder(BaseBBoxCoder):
    """Phase-Shifting Coder.

    `Phase-Shifting Coder (PSC)
    <https://arxiv.org/abs/2211.06368>`.

    Args:
        angle_version (str): Angle definition.
            Only 'le90' is supported at present.
        dual_freq (bool, optional): Use dual frequency. Default: True.
        num_step (int, optional): Number of phase steps. Default: 3.
        thr_mod (float): Threshold of modulation. Default: 0.47.
    """

    def __init__(self,
                 angle_version: str,
                 dual_freq: bool = True,
                 num_step: int = 3,
                 thr_mod: float = 0.47):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['le90']
        self.dual_freq = dual_freq
        self.num_step = num_step
        self.thr_mod = thr_mod
        if self.dual_freq:
            self.encode_size = 2 * self.num_step
        else:
            self.encode_size = self.num_step

        self.coef_sin = torch.tensor(
            tuple(
                torch.sin(torch.tensor(2 * k * math.pi / self.num_step))
                for k in range(self.num_step)))
        self.coef_cos = torch.tensor(
            tuple(
                torch.cos(torch.tensor(2 * k * math.pi / self.num_step))
                for k in range(self.num_step)))

    def encode(self, angle_targets: Tensor) -> Tensor:
        """Phase-Shifting Encoder.

        Args:
            angle_targets (Tensor): Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1)

        Returns:
            list[Tensor]: The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
        """
        phase_targets = angle_targets * 2
        phase_shift_targets = tuple(
            torch.cos(phase_targets + 2 * math.pi * x / self.num_step)
            for x in range(self.num_step))

        # Dual-freq PSC for square-like problem
        if self.dual_freq:
            phase_targets = angle_targets * 4
            phase_shift_targets += tuple(
                torch.cos(phase_targets + 2 * math.pi * x / self.num_step)
                for x in range(self.num_step))

        return torch.cat(phase_shift_targets, axis=-1)

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        """Phase-Shifting Decoder.

        Args:
            angle_preds (Tensor): The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (num_anchors * H * W, encode_size)
            keepdim (bool): Whether the output tensor has dim retained or not.

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1) when keepdim is true,
                (num_anchors * H * W) otherwise
        """
        self.coef_sin = self.coef_sin.to(angle_preds)
        self.coef_cos = self.coef_cos.to(angle_preds)

        phase_sin = torch.sum(
            angle_preds[:, 0:self.num_step] * self.coef_sin,
            dim=-1,
            keepdim=keepdim)
        phase_cos = torch.sum(
            angle_preds[:, 0:self.num_step] * self.coef_cos,
            dim=-1,
            keepdim=keepdim)
        phase_mod = phase_cos**2 + phase_sin**2
        phase = -torch.atan2(phase_sin, phase_cos)  # In range [-pi,pi)

        if self.dual_freq:
            phase_sin = torch.sum(
                angle_preds[:, self.num_step:(2 * self.num_step)] *
                self.coef_sin,
                dim=-1,
                keepdim=keepdim)
            phase_cos = torch.sum(
                angle_preds[:, self.num_step:(2 * self.num_step)] *
                self.coef_cos,
                dim=-1,
                keepdim=keepdim)
            phase_mod = phase_cos**2 + phase_sin**2
            phase2 = -torch.atan2(phase_sin, phase_cos) / 2

            # Phase unwarpping, dual freq mixing
            # Angle between phase and phase2 is obtuse angle
            idx = torch.cos(phase) * torch.cos(phase2) + torch.sin(
                phase) * torch.sin(phase2) < 0
            # Add pi to phase2 and keep it in range [-pi,pi)
            phase2[idx] = phase2[idx] % (2 * math.pi) - math.pi
            phase = phase2

        # Set the angle of isotropic objects to zero
        phase[phase_mod < self.thr_mod] *= 0
        angle_pred = phase / 2
        return angle_pred

@ROTATED_BBOX_CODERS.register_module()
class PseudoAngleCoder(BaseBBoxCoder):
    """Pseudo Angle Coder."""

    encode_size = 1

    def encode(self, angle_targets: Tensor) -> Tensor:
        return angle_targets

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        if keepdim:
            return angle_preds
        else:
            return angle_preds.squeeze(-1)


@ROTATED_BBOX_CODERS.register_module()
class STCoder(BaseBBoxCoder):
    """
    Structure Tensor Coder (ST)
    Args:
        angle_version (str): Angle definition.
            Only 'le90' is supported at present.
    """

    def __init__(self,
                 angle_version: str,
                 anisotropy: int = 2
                 ):
        super().__init__()
        self.angle_version = angle_version
        self.encode_size = 3
        assert angle_version in ['le90']
        self.anisotropy = anisotropy

        if anisotropy == 1:
            self.width = 0.5
            self.height = 0.5
        elif anisotropy == 2:
            self.width = 1
            self.height = 0.5
        elif anisotropy == 4:
            self.width = 2
            self.height = 0.5
        else:
            raise NotImplementedError

    def encode(self, angle, width, height):
        # Ensure input tensors are of shape [N, 1]
        '''
        if len(angle.shape) > 1:
            angle = angle.squeeze(1)
        if len(width.shape) > 1:
            width = width.squeeze(1)
        if len(height.shape) > 1:
            height = height.squeeze(1)
        '''
        angle = angle.squeeze(1)
        width = width.squeeze(1)
        height = height.squeeze(1)

        equal_indices = width - height < 1e-2
        mod_angle = norm_angle(angle, 'le45')
        angle[equal_indices] = mod_angle[equal_indices]

        # Compute cosine and sine of the angles
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)

        # Create the rotation matrix for each angle
        R = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=1),
            torch.stack([sin_theta, cos_theta], dim=1)
        ], dim=1)

        # Eigenvalues (width and height) need to be of shape [N, 2]
        eigenvalues = torch.stack([width * self.width, height * self.height], dim=1)

        # Create diagonal matrix of eigenvalues for each batch
        Lambda = torch.stack([
            torch.diag(eigenvalues[i])
            for i in range(eigenvalues.shape[0])
        ])

        # Compute structure tensor T for each angle
        T = torch.bmm(R, torch.bmm(Lambda, R.transpose(1, 2)))

        # Extract the components of the structure tensor
        a = T[:, 0, 0]
        b = T[:, 1, 1]
        c = T[:, 0, 1]  # Since T is symmetric, T[0, 1] == T[1, 0]

        return torch.stack([a, b, c], dim=1)

    def decode(self, str_tensor, angle_range='le90'):
        # Extract individual components from str_tensor
        a = str_tensor[:, 0]  # shape [N]
        b = str_tensor[:, 1]  # shape [N]
        c = str_tensor[:, 2]  # shape [N]

        # Build structure tensor as a matrix
        structure_tensors = torch.stack([
            torch.stack([a, c], dim=-1),  # shape [N, 2]
            torch.stack([c, b], dim=-1)   # shape [N, 2]
        ], dim=-2)  # shape [N, 2, 2]

        # Calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(structure_tensors)

        # Extract the real parts of the eigenvalues and eigenvectors
        eigenvalues = torch.abs(eigenvalues.real)  # shape [N, 2]
        eigenvectors = eigenvectors.real  # shape [N, 2, 2]

        w =  eigenvalues[:, 0]
        h =  eigenvalues[:, 1]
        a = torch.atan2(eigenvectors[:, 1, 1], eigenvectors[:, 0, 1])  # shape [N]
        a = norm_angle(a, angle_range)

        # Construct the obb tensor [center_x, center_y, width, height, angle]
        #obb = torch.stack([w, h, a], dim=-1)  # shape [N, 5]

        return w, h, a

def rotated_box_to_poly(rrects):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    n = rrects.shape[0]
    if n == 0:
        return torch.zeros([0,8])
    x_ctr  = rrects[:, 0]
    y_ctr  = rrects[:, 1]
    width  = rrects[:, 2]
    height = rrects[:, 3]
    angle  = rrects[:, 4]
    tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
    rect = torch.stack([tl_x, br_x, br_x, tl_x, tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y, tl_y, tl_y, br_y, br_y], 1).reshape([n, 2, 8])
    c = torch.cos(angle)
    s = torch.sin(angle)
    R = torch.stack([c, c, c, c, s, s, s, s, -s, -s, -s, -s, c, c, c, c], 1).reshape([n, 2, 8])
    offset = torch.stack([x_ctr, x_ctr, x_ctr, x_ctr, y_ctr, y_ctr, y_ctr, y_ctr], 1)
    poly = ((R * rect).sum(1) + offset).reshape([n, 2, 4]).permute([0,2,1]).reshape([n, 8])
    return poly

def norm_angle_cobb(angle, range=[float(-np.pi / 4), float(np.pi)]):
    ret = (angle - range[0]) % range[1] + range[0]
    return ret

def poly_to_rotated_box(polys):
    """
    polys:n*8
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rrect:[x_ctr,y_ctr,w,h,angle]
    """
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)

    edge1 = torch.sqrt(
        (pt1[..., 0] - pt2[..., 0])**2 + (pt1[..., 1] - pt2[..., 1])**2)
    edge2 = torch.sqrt(
        (pt2[..., 0] - pt3[..., 0])**2 + (pt2[..., 1] - pt3[..., 1])**2)

    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]), (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]), (pt4[..., 0] - pt1[..., 0]))
    angles = torch.zeros_like(angles1)
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]

    angles = norm_angle_cobb(angles, range=[float(-np.pi / 2), float(np.pi / 2)])
    #angles = norm_angle(angles, angle_range)

    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0

    edges = torch.stack([edge1, edge2], dim=1)
    width = torch.max(edges, 1)[0]
    height = torch.min(edges, 1)[0]

    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


@ROTATED_BBOX_CODERS.register_module()
class COBBCoder(BaseBBoxCoder):
    """
    Structure Tensor Coder (ST)
    Args:
        angle_version (str): Angle definition.
            Only 'le90' is supported at present.
    """

    def __init__(self,
                 angle_version: str,
                 pow_iou: float = 1.,
                 ratio_type: str = 'sig',
                 ):
        super().__init__()
        self.angle_version = angle_version
        assert angle_version in ['le90']
        self.encode_size = 4
        self.pow_iou = pow_iou
        self.ratio_type = ratio_type

    @torch.no_grad()
    def build_iou_matrix(self, hbboxes:torch.Tensor, ratio_pred:torch.Tensor):
        assert hbboxes.size(1) == 4
        assert ratio_pred.shape[0] == hbboxes.shape[0]

        min_x = hbboxes[:, 0]
        min_y = hbboxes[:, 1]
        max_x = hbboxes[:, 2]
        max_y = hbboxes[:, 3]
        w = max_x - min_x
        h = max_y - min_y
        w_large = w > h
        w_large_ratio = ratio_pred[w_large] / 4
        w_large_w = w[w_large]
        w_large_h = h[w_large]
        h_large = torch.logical_not(w_large)
        h_large_ratio = ratio_pred[h_large] / 4
        h_large_w = w[h_large]
        h_large_h = h[h_large]

        x_ratio = torch.zeros_like(ratio_pred)
        y_ratio = torch.zeros_like(ratio_pred)

        # x(1-x)=r --> x^2-x+r=0
        h_large_delta_x = torch.sqrt(1 - 4 * h_large_ratio)
        x_ratio[h_large] = (1 - h_large_delta_x) / 2
        # h^2y(1-y) = w^2r --> y^2-y+(w^2/h^2)r=0
        h_large_delta_y = torch.sqrt(1 - 4 * (h_large_w*h_large_w/(h_large_h*h_large_h)) * h_large_ratio)
        y_ratio[h_large] = (1 - h_large_delta_y) / 2

        # y(1-y)=r --> y^2-y+r=0
        w_large_delta_y = torch.sqrt(1 - 4 * w_large_ratio)
        y_ratio[w_large] = (1 - w_large_delta_y) / 2
        # w^2x(1-x) = h^2r --> x^2-x+(h^2/w^2)r=0
        w_large_delta_x = torch.sqrt(1 - 4 * (w_large_h*w_large_h/(w_large_w*w_large_w)) * w_large_ratio)
        x_ratio[w_large] = (1 - w_large_delta_x) / 2
        iou_self = torch.zeros_like(ratio_pred)

        # type0 shape: l01, l02
        # type1 shape: l03, l04
        l_01 = torch.sqrt((x_ratio * w)**2       + (y_ratio * h)**2)
        l_02 = torch.sqrt(((1 - x_ratio) * w)**2 + ((1 - y_ratio) * h)**2)
        l_03 = torch.sqrt((x_ratio * w)**2       + ((1 - y_ratio) * h)**2)
        l_04 = torch.sqrt(((1 - x_ratio) * w)**2 + (y_ratio * h)**2)
        # (yh)t/(xw) + (1-y)ht/(xw) = (1-2x)w --> t=(1-2x)xw^2/h
        # intersection = (1 - t / ((1-y)h)) * l02 * l01 = (1 - (1-2x)xw^2/((1-y)h^2)) * l01 * l02
        i_01 = (1 - (1 - 2 * x_ratio) * x_ratio * w * w / ((1 - y_ratio) * h * h)) * l_01 * l_02
        iou_01 = i_01 / (l_01 * l_02 + l_03 * l_04 - i_01)
        i_02 = (1 - (1 - 2 * y_ratio) * y_ratio * h * h / ((1 - x_ratio) * w * w)) * l_01 * l_02
        iou_02 = i_02 / (l_01 * l_02 + l_03 * l_04 - i_02)
        # i_03 = 2 * x_ratio * y_ratio * w * h
        i_03 = (x_ratio + y_ratio - 2 * x_ratio * y_ratio)**2 / ((1 - x_ratio) * (1 - y_ratio)) * w * h / 2
        iou_03 = torch.zeros_like(iou_02)
        nzero = l_01 > 1e-5
        iou_03[nzero] = i_03[nzero] / (l_01[nzero] * l_02[nzero] * 2 - i_03[nzero])

        h1 = 0.5 * w - (0.5 - y_ratio) / (1 - y_ratio) * w * x_ratio
        h2 = 0.5 * h - (0.5 - x_ratio) / (1 - x_ratio) * h * y_ratio
        s2 = h1**2 + h2**2
        tana = (0.5 - x_ratio) / (1 - x_ratio) * l_04 / (0.5 / (1 - y_ratio) * l_03)
        tanb = (0.5 - y_ratio) / (1 - y_ratio) * l_03 / (0.5 / (1 - x_ratio) * l_04)
        nzero = tana + tanb > 1e-8
        i_12_tpye1 = torch.zeros_like(i_03)
        i_12_tpye1[nzero] = tana[nzero] * tanb[nzero] / (tana[nzero] + tanb[nzero])
        i_12 = i_12_tpye1 * s2 * 2 + h1 * h2 * 2
        iou_12 = i_12 / (l_03 * l_04 * 2 - i_12)
        iou_self = torch.ones_like(ratio_pred)
        iou0 = torch.stack([iou_self, iou_01, iou_02, iou_03], dim=-1)
        iou1 = torch.stack([iou_01, iou_self, iou_12, iou_02], dim=-1)
        iou2 = torch.stack([iou_02, iou_12, iou_self, iou_01], dim=-1)
        iou3 = torch.stack([iou_03, iou_02, iou_01, iou_self], dim=-1)
        return torch.stack([iou0, iou1, iou2, iou3], dim=-2)

    def encode(self, rbboxes:torch.Tensor):
        assert rbboxes.shape[1] == 5

        polys = rotated_box_to_poly(rbboxes)

        max_x, max_x_idx = torch.max(polys[:, ::2],   dim=1)
        min_x, min_x_idx = torch.min(polys[:, ::2],   dim=1)
        max_y, max_y_idx = torch.max(polys[:, 1::2],  dim=1)
        min_y, min_y_idx = torch.min(polys[:, 1::2],  dim=1)
        hbboxes = torch.stack([min_x, min_y, max_x, max_y], dim=1)

        polys = polys.view(-1, 4, 2)
        w = hbboxes[:, 2] - hbboxes[:, 0]
        h = hbboxes[:, 3] - hbboxes[:, 1]
        x_ind = torch.argsort(polys[:, :, 0], dim=1)
        y_ind = torch.argsort(polys[:, :, 1], dim=1)
        polys_x = polys[:, :, 0]
        polys_y = polys[:, :, 1]
        s_x = polys_x[(torch.arange(polys.shape[0]), x_ind[:, 1])]
        s_y = polys_y[(torch.arange(polys.shape[0]), y_ind[:, 1])]
        dx = (s_x - hbboxes[:, 0]) / w
        dy = (s_y - hbboxes[:, 1]) / h

        w_large = w > h
        h_large_dx = dx[torch.logical_not(w_large)]
        w_large_dy = dy[w_large]
        ratio = torch.zeros_like(max_x)
        ratio[torch.logical_not(w_large)] = h_large_dx * (1 - h_large_dx) * 4
        ratio[w_large] = w_large_dy * (1 - w_large_dy) * 4

        assert torch.all(ratio <= 1.0 + 1e-5)

        ious = self.build_iou_matrix(hbboxes, ratio)
        is_type13 = torch.logical_or(x_ind[:, 1] == y_ind[:, 2], x_ind[:, 1] == y_ind[:, 3])
        is_type23 = torch.logical_or(x_ind[:, 0] == y_ind[:, 2], x_ind[:, 0] == y_ind[:, 3])
        rtype = is_type23.long() * 2 + is_type13.long()
        ious = ious[(torch.arange(ious.shape[0]), rtype)]

        ious = torch.pow(ious, self.pow_iou)

        if self.ratio_type == 'sig':
            ratio = 1 - torch.sqrt(1 - ratio)
        elif self.ratio_type == 'ln':
            ratio = (1 - torch.sqrt(1 - ratio)) / 2
            square_like = torch.logical_or(rtype == 1, rtype == 2)
            ratio[square_like] = 1 - ratio[square_like]
            ratio = 1 + torch.log2(ratio)
        else:
            raise NotImplementedError

        return ratio[:, None], ious

    # FIXME bboxes_pred should be ratio_pred
    def decode(self, hbboxes:torch.Tensor, bboxes_pred:torch.Tensor, rotated_scores:torch.Tensor, max_shape=None):
        assert hbboxes.shape[0] == hbboxes.shape[0] == rotated_scores.shape[0]
        
        bboxes_pred = torch.squeeze(bboxes_pred, dim=-1)
        if self.ratio_type == 'sig':
            #bboxes_pred = torch.clamp(bboxes_pred) # WEIRD this is the identity according to the jittor doc
            bboxes_pred = 1 - (1 - bboxes_pred)**2
        elif self.ratio_type == 'ln':
            square_like = bboxes_pred > 0
            bboxes_pred = torch.clamp((bboxes_pred - 1)**2, min_v=0., max_v=1.)
            bboxes_pred[square_like] = 1 - bboxes_pred[square_like]
            bboxes_pred = 1 - (1 - bboxes_pred * 2)**2
        else:
            raise NotImplementedError

        assert rotated_scores.size(1) == 4
        rbboxes_list = self.build_polypairs(hbboxes, bboxes_pred)
        rbboxes_proposals = torch.concat([rbboxes[:, None, :] for rbboxes in rbboxes_list], dim=1)
        rbboxes_proposals = rbboxes_proposals.reshape(-1, rbboxes_proposals.shape[-1])

        num_hbboxes = hbboxes.size(0)
        best_index = torch.argmax(rotated_scores, dim=-1, keepdims=False) + \
            torch.arange(num_hbboxes, dtype=torch.int32, device=rotated_scores.device) * 4

        best_rbboxes = rbboxes_proposals[best_index, :]
        return best_rbboxes

    def build_polypairs(self, hbboxes:torch.Tensor, ratio_pred: torch.Tensor):
        assert hbboxes.shape[1] == 4
        assert ratio_pred.shape[0] == hbboxes.shape[0]
        min_x = hbboxes[:, 0]
        min_y = hbboxes[:, 1]
        max_x = hbboxes[:, 2]
        max_y = hbboxes[:, 3]
        
        # (left, bottom, right, top)
        w = max_x - min_x
        h = max_y - min_y
        w_large = w > h
        w_large_ratio = ratio_pred[w_large] / 4
        w_large_w = w[w_large]
        w_large_h = h[w_large]
        h_large = torch.logical_not(w_large)
        h_large_ratio = ratio_pred[h_large] / 4
        h_large_w = w[h_large]
        h_large_h = h[h_large]
        x1 = torch.zeros_like(ratio_pred)
        x2 = torch.zeros_like(ratio_pred)
        y1 = torch.zeros_like(ratio_pred)
        y2 = torch.zeros_like(ratio_pred)

        # x(1-x)=r --> x^2-x+r=0
        h_large_delta_x = torch.sqrt(1 - 4 * h_large_ratio)
        x1[h_large] = (1 - h_large_delta_x) / 2 * h_large_w
        x2[h_large] = (1 + h_large_delta_x) / 2 * h_large_w
        # h^2y(1-y) = w^2r --> y^2-y+(w^2/h^2)r=0
        h_large_delta_y = torch.sqrt(1 - 4 * (h_large_w*h_large_w/(h_large_h*h_large_h)) * h_large_ratio)
        y1[h_large] = (1 - h_large_delta_y) / 2 * h_large_h
        y2[h_large] = (1 + h_large_delta_y) / 2 * h_large_h

        # y(1-y)=r --> y^2-y+r=0
        w_large_delta_y = torch.sqrt(1 - 4 * w_large_ratio)
        y1[w_large] = (1 - w_large_delta_y) / 2 * w_large_h
        y2[w_large] = (1 + w_large_delta_y) / 2 * w_large_h
        # w^2x(1-x) = h^2r --> x^2-x+(h^2/w^2)r=0
        w_large_delta_x = torch.sqrt(1 - 4 * (w_large_h*w_large_h/(w_large_w*w_large_w)) * w_large_ratio)
        x1[w_large] = (1 - w_large_delta_x) / 2 * w_large_w
        x2[w_large] = (1 + w_large_delta_x) / 2 * w_large_w

        poly1 = torch.stack([min_x+x1, min_y,
                             max_x, min_y+y2,
                             max_x-x1, max_y,
                             min_x, max_y-y2], dim=-1)
        poly2 = torch.stack([min_x+x2, min_y,
                             max_x, min_y+y2,
                             max_x-x2, max_y,
                             min_x, max_y-y2], dim=-1)
        poly3 = torch.stack([min_x+x1, min_y,
                             max_x, min_y+y1,
                             max_x-x1, max_y,
                             min_x, max_y-y1], dim=-1)
        poly4 = torch.stack([min_x+x2, min_y,
                             max_x, min_y+y1,
                             max_x-x2, max_y,
                             min_x, max_y-y1], dim=-1)

        return [poly_to_rotated_box(poly1),
                poly_to_rotated_box(poly2),
                poly_to_rotated_box(poly3),
                poly_to_rotated_box(poly4)]



# Write a dummy test for the COBBCoder, to verify it can encode and decode the angle correctly

def test_cobb_coder():
    coder = COBBCoder(angle_version='le90')
    center = torch.tensor([[0, 0], [0, 0]])
    wh = torch.tensor([[1, 1], [1, 1]])
    angle = torch.tensor([[np.pi/4], [10 * np.pi/180]])
    ratio, ious = coder.encode(torch.cat([center, wh, angle], dim=-1))
    print("ratio: ", ratio)
    print("ious: ", ious)

    # Decode the ratio and ious back to angle
    x1 = center[:,0] - wh[:,0]/2
    y1 = center[:,1] - wh[:,1]/2
    x2 = center[:,0] + wh[:,0]/2
    y2 = center[:,1] + wh[:,1]/2

    # Compute the HBBOX from the rotated bbox by applying a rotation matrix


    
    xy_min_max_bbox = torch.stack([x1, y1, x2, y2], dim=-1)
    decoded_bbox = coder.decode(xy_min_max_bbox, ratio, ious)
    
    print('Input bbox 1: ', torch.cat([center, wh, angle], dim=-1)[0])
    print("Decoded bbox 1: ", decoded_bbox[0])
    print()
    print('Input bbox 2: ', torch.cat([center, wh, angle], dim=-1)[1])
    print("Decoded bbox 2: ", decoded_bbox[1])

test_cobb_coder()