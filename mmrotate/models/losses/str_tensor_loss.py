import torch
import torch.nn as nn
from ..builder import ROTATED_LOSSES
from mmrotate.core import build_bbox_coder

@ROTATED_LOSSES.register_module()
class StructureTensorLoss(nn.Module):
    def __init__(self, beta=100, gamma=3e4, loss_weight=1.0):
        """
        Custom loss for structure tensors.

        Args:
            beta (float): Controls how fast alpha decays with the aspect ratio.
        """
        super(StructureTensorLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.beta = beta
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.angle_coder = build_bbox_coder(dict(type='STCoder', angle_version='le90'))

    def calculate_alpha(self, width, height):
        """
        Calculate the alpha factor based on the aspect ratio (r = max(width, height) / min(width, height)).

        Args:
            width (Tensor): The width (eigenvalue 1).
            height (Tensor): The height (eigenvalue 2).

        Returns:
            alpha (Tensor): The scaling factor alpha based on aspect ratio.
        """
        # Compute the aspect ratio r = max(width, height) / min(width, height)
        aspect_ratio = torch.max(width, height) / torch.min(width, height)
        # Apply the exponential decay for alpha
        alpha = self.gamma * torch.exp(-self.beta * (aspect_ratio - 1))
        return alpha

    def forward(self, str_tensor_target, str_tensor_pred):
        """
        Forward pass to compute the loss.

        Args:
            str_tensor_target (Tensor): Target structure tensor (shape [N, 3]).
            str_tensor_pred (Tensor): Predicted structure tensor (shape [N, 3]).
            width (Tensor): Eigenvalue corresponding to width (shape [N]).
            height (Tensor): Eigenvalue corresponding to height (shape [N]).

        Returns:
            loss (Tensor): The computed loss.
        """

        
        # Calculate L1 loss between the structure tensors
        l1_loss_value = self.l1_loss(str_tensor_target, str_tensor_pred)

        # Extract the eigenvalues from the structure tensors
        width, height, _ = self.angle_coder.decode(str_tensor_target)

        # Extract the off-diagonal element T^01 (which is the 3rd element in each tensor)
        off_diag_loss = torch.abs(str_tensor_pred[:, 2] - str_tensor_target[:, 2])

        # Calculate alpha based on the aspect ratio
        alpha = self.calculate_alpha(width, height)

        # Compute the overall loss: L1 loss + alpha-weighted off-diagonal loss
        loss = l1_loss_value + torch.mean(alpha * off_diag_loss)
        return self.loss_weight * loss
