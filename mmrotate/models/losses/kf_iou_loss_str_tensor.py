# Copyright (c) SJTU. All rights reserved.
import torch
from mmdet.models.losses.utils import weighted_loss
from torch import nn

from ..builder import ROTATED_LOSSES

def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


def make_sigma(str_tensor, bbox):
    """Convert structure tensor angle to 2-D Gaussian distribution Sigma.

    Args:
        str_tensor (torch.Tensor): structure tensor angle with shape (N, 3), corresponding to the a,b,c of the structure tensor.

    Returns:
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2) built with a,b and c as follows: [[a,c],[c,b]].
    """
    wh = bbox[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = str_tensor[:, 2]
    
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,1))
    
    return sigma


@weighted_loss
def kfiou_loss(pred,
               target,
               pred_str_tensor=None,
               targets_str_tensor=None,
               fun=None,
               beta=1.0 / 9.0,
               eps=1e-6):
    """Kalman filter IoU loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        pred_decode (torch.Tensor): Predicted decode bboxes.
        targets_decode (torch.Tensor): Corresponding gt decode bboxes.
        fun (str): The function applied to distance. Defaults to None.
        beta (float): Defaults to 1.0/9.0.
        eps (float): Defaults to 1e-6.

    Returns:
        loss (torch.Tensor)
    """
    xy_p = pred[:, :2]
    xy_t = target[:, :2]
    
    #Sigma_p = make_sigma(pred_str_tensor, pred)
    #Sigma_t = make_sigma(targets_str_tensor, target)

    _, Sigma_p = xy_wh_r_2_xy_sigma(pred_str_tensor)
    _, Sigma_t = xy_wh_r_2_xy_sigma(targets_str_tensor)


    # verify there are no nans
    assert not torch.isnan(Sigma_p).any()
    assert not torch.isnan(Sigma_t).any()

    # Smooth-L1 norm
    diff = torch.abs(xy_p - xy_t)
    xy_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                          diff - 0.5 * beta).sum(dim=-1)
    Vb_p = 4 * Sigma_p.det().sqrt()
    Vb_t = 4 * Sigma_t.det().sqrt()
    K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
    Sigma = Sigma_p - K.bmm(Sigma_p)
    Vb = 4 * Sigma.det().sqrt()
    Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
    KFIoU = Vb / (Vb_p + Vb_t - Vb + eps)

    if fun == 'ln':
        kf_loss = -torch.log(KFIoU + eps)
    elif fun == 'exp':
        kf_loss = torch.exp(1 - KFIoU) - 1
    else:
        kf_loss = 1 - KFIoU

    loss = (xy_loss + kf_loss).clamp(0)

    return loss


@ROTATED_LOSSES.register_module()
class STKFLoss(nn.Module):
    """Kalman filter based loss.

    Args:
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 fun='none',
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(STKFLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['none', 'ln', 'exp']
        self.fun = fun
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                pred_str_tensor=None,
                targets_str_tensor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            pred_decode (torch.Tensor): Predicted structure tensor angle.
            targets_str_tensor (torch.Tensor): Corresponding gt angle as a structure tensor.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.

        Returns:
            loss (torch.Tensor)
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        
        # if contains nan, return 0
        if torch.isnan(pred).any() or torch.isnan(target).any():
            return torch.tensor(0.0)

        return kfiou_loss(
            pred,
            target,
            fun=self.fun,
            weight=weight,
            avg_factor=avg_factor,
            pred_str_tensor=pred_str_tensor,
            targets_str_tensor=targets_str_tensor,
            reduction=reduction,
            **kwargs) * self.loss_weight
