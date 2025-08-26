import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from .loss_util import weighted_loss

from MDIQA.utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def cross_entropy(pred, target):
    return F.cross_entropy(pred, target, reduction='none')


@weighted_loss
def binary_cross_entropy(pred, target):
    return F.binary_cross_entropy(pred, target, reduction='none')


@weighted_loss
def binary_cross_entropy_with_logits(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target, reduction='none')


@weighted_loss
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: float,
    with_logits=True,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    if with_logits:
        inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (batch_size + eps)
    return loss


@weighted_loss
def nll_loss(pred, target):
    return F.nll_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CrossEntropyLoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * cross_entropy(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class BinaryCrossEntropyLoss(nn.Module):
    """Binary Cross Entropy loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', with_logits=True):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        if with_logits:
            self.func = binary_cross_entropy_with_logits
        else:
            self.func = binary_cross_entropy

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (B, 1, H, W). Predicted tensor. (After Sigmoid)
            target (Tensor): of shape (B, 1, H, W). Ground truth tensor. (0 or 1)
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * self.func(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class DICELoss(nn.Module):
    """DICE loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', with_logits=True):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.with_logits = with_logits

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (B, 1, H, W). Predicted tensor. (After Sigmoid)
            target (Tensor): of shape (B, 1, H, W). Ground truth tensor. (0 or 1)
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * dice_loss(pred, target, batch_size=pred.shape[0], with_logits=self.with_logits)


@LOSS_REGISTRY.register()
class NLLLoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * nll_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class IntraRankingLoss(nn.Module):
    """Rank loss.
    区别于自带的ranking loss
    用于local training
    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, loss_weight_consistency=0., reduction='mean', margin=0.5, threshold=0., concistency_threshold=0.):
        super(IntraRankingLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.loss_weight_consistency = loss_weight_consistency
        self.reduction = reduction
        self.margin = margin
        self.threshold = threshold
        self.concistency_threshold = concistency_threshold

    def forward(self, pred, target):
        """
        Compute ranking loss for tensor pred and target.
        
        Parameters:
        pred (torch.Tensor): Predicted tensor of shape (B, h, w)
        target (torch.Tensor): Ground truth tensor of shape (B, h, w)
        margin (float): Margin for margin ranking loss
        
        Returns:
        torch.Tensor: Loss value
        """
        pred, target = pred.flatten(1), target.flatten(1)
        B, d = pred.shape

        # Expand dimensions to enable broadcasting for pairwise comparison
        pred_expanded_1 = pred.unsqueeze(2).expand(B, d, d)
        pred_expanded_2 = pred.unsqueeze(1).expand(B, d, d)
        
        target_expanded_1 = target.unsqueeze(2).expand(B, d, d)
        target_expanded_2 = target.unsqueeze(1).expand(B, d, d)
        
        # Compute differences
        pred_diff = pred_expanded_1 - pred_expanded_2
        target_diff = target_expanded_1 - target_expanded_2
        
        # Create a binary mask where target_diff is positive        
        mask = (target_diff > self.threshold).float()
        mask_consistency = (target_diff.abs() < self.concistency_threshold).float()
        mask_consistency = torch.triu(mask_consistency, diagonal=1)
        
        # Compute the ranking loss using margin ranking loss
        loss = F.margin_ranking_loss(pred_diff, torch.zeros_like(pred_diff), mask, margin=self.margin, reduction='none')
        loss_consistency = F.mse_loss(pred_diff * mask_consistency, torch.zeros_like(pred_diff), reduction='none')

        # Sum over all pairs and normalize by the number of pairs
        if self.reduction in ['mean', 'none']:
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
            loss_consistency = loss_consistency.sum() / (mask_consistency.sum() + 1e-6)
            return self.loss_weight * loss + self.loss_weight_consistency * loss_consistency
        else:
            loss = (loss * mask).sum()
            loss_consistency = loss_consistency.sum()
            return self.loss_weight * loss + self.loss_weight_consistency * loss_consistency


@LOSS_REGISTRY.register()
class SynthesisInterRankLoss(nn.Module):
    """Monotonicity regularization loss, will be zero when rankings of pred are as expected."""

    def __init__(self, loss_weight=1.0, num_severity=5, num_ratio=4):
        super(SynthesisInterRankLoss, self).__init__()
        self.loss_weight = loss_weight
        self.num_severity = num_severity
        self.num_ratio = num_ratio

    def forward(self, pred):
        # Ensure pred is of shape (batch_size, 20)
        batch_size = pred.size(0)

        # First dimension: p[4i] > p[4i+1] > p[4i+2] > p[4i+3]
        pred_reshaped = pred.view(batch_size, self.num_severity, self.num_ratio)
        diff1 = pred_reshaped[:, :, 1:] - pred_reshaped[:, :, :-1]  # hope diff < 0
        loss1 = F.relu(diff1).mean()
        
        # Second dimension: p[i] > p[i+4] > p[i+8] > p[i+12] > p[i+16]
        pred_transposed = pred.view(batch_size, self.num_severity, self.num_ratio).transpose(1, 2)
        diff2 = pred_transposed[:, :, 1:] - pred_transposed[:, :, :-1]
        loss2 = F.relu(diff2).mean()

        loss = loss1 + loss2
        
        return self.loss_weight * loss
    

@LOSS_REGISTRY.register()
class FidelityLoss(nn.Module):
    """
    LIQA Fidelty Loss for pair wise learning to rank
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FidelityLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (B). Predicted tensor.
            target (Tensor): of shape (B). Ground truth tensor.
        """
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
        pred = pred - pred.t()
        target = target - target.t()
        triu_indices = torch.triu_indices(pred.shape[0], pred.shape[0], offset=1)
        pred = pred[triu_indices[0], triu_indices[1]]
        target = target[triu_indices[0], triu_indices[1]]
        g = 0.5 * (torch.sign(target) + 1)
        constant = torch.sqrt(torch.Tensor([2.])).to(pred.device)
        p = 0.5 * (1 + torch.erf(pred / constant))  # torch.erf: Gaussian cdf
        g = g.view(-1, 1)
        p = p.view(-1, 1)
        if self.reduction == 'sum':
            return self.loss_weight * torch.sum((1 - (torch.sqrt(p * g + 1e-8) + torch.sqrt((1 - p) * (1 - g) + 1e-8))))
        else:
            return self.loss_weight *torch.mean((1 - (torch.sqrt(p * g + 1e-8) + torch.sqrt((1 - p) * (1 - g) + 1e-8))))


@LOSS_REGISTRY.register()
class CLSLoss(nn.Module):
    """CLS loss.
    分类loss 对每个patch是否存在degradation分类
    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', threshold=0.5):
        super(CLSLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.criterion = torch.nn.BCELoss(reduction=reduction)
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, score_curr_map, gt, weight=None, **kwargs):
        """
        Args:
            score_curr_map (Tensor): of shape (B, H, W).
            gt (Tensor, optional): of shape (B, H, W).  1:clear 0:blur
        """
        _, h, w = score_curr_map.shape
        clear_mask = (gt > self.threshold).float()  # patch-wise clear/blur classification gt
        score_curr_map = torch.sigmoid(score_curr_map)  # B,H,W
        loss = self.criterion(score_curr_map, clear_mask)
        return self.loss_weight * loss
    