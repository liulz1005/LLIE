import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss


#新增----------------------------------------------------------------------
from torchvision.models import vgg19, VGG19_Weights

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


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
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

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
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
    


#新增--------------------------------------------------------------------
class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2'], 
                 weights=[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.weights = weights
        self.feature_layers = feature_layers
        
        # 加载VGG19模型（不指定device，后续自动适配输入设备）
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        for param in vgg.parameters():
            param.requires_grad = False  # 冻结参数
        
        # 修正层命名逻辑：按VGG实际结构生成conv1_1、conv1_2等名称
        self.layers = nn.ModuleDict()
        current_layer = []
        conv_block = 1  # 当前卷积块编号（1-5）
        conv_in_block = 0  # 块内卷积层编号（1-2/3/4）
        
        for module in vgg:
            current_layer.append(module)
            if isinstance(module, nn.Conv2d):
                conv_in_block += 1
                layer_name = f'conv{conv_block}_{conv_in_block}'
                self.layers[layer_name] = nn.Sequential(*current_layer)
            elif isinstance(module, nn.MaxPool2d):
                layer_name = f'pool{conv_block}'
                self.layers[layer_name] = nn.Sequential(*current_layer)
                current_layer = []  # 池化后重置当前层
                conv_block += 1  # 进入下一个卷积块
                conv_in_block = 0  # 重置块内计数
        
        # 检查指定的特征层是否存在
        available_layers = list(self.layers.keys())
        for layer in feature_layers:
            if layer not in available_layers:
                raise ValueError(f"无效的特征层: {layer}，可选层: {available_layers}")
        
        # 图像归一化参数（不预先指定device，forward时自动匹配）
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.criterion = nn.MSELoss()

    def forward(self, pred, gt):
        # 自动匹配输入设备
        device = pred.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        
        # 输入归一化（假设输入范围为[0,1]，先转换为[0,255]再归一化）
        pred = pred * 255.0  # 从[0,1]转为[0,255]
        gt = gt * 255.0
        pred = (pred - mean) / std
        gt = (gt - mean) / std
        
        total_loss = 0.0
        for i, layer_name in enumerate(self.feature_layers):
            # 提取特征（层自动转移到输入设备）
            pred_feat = self.layers[layer_name](pred)
            gt_feat = self.layers[layer_name](gt)
            total_loss += self.weights[i] * self.criterion(pred_feat, gt_feat)
        
        return total_loss


class GradientLoss(nn.Module):
    """高频损失：基于梯度差异（Sobel算子）"""
    def __init__(self, loss_weight=1.0, reduction='mean', kernel_size=3):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

        # Sobel算子（水平和垂直方向）
        if kernel_size == 3:
            self.kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
            self.kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        else:
            raise NotImplementedError(f"不支持的核大小: {kernel_size}")

    def forward(self, pred, target):
        # 扩展核以匹配输入通道数
        b, c, h, w = pred.shape
        kernel_x = self.kernel_x.repeat(c, 1, 1, 1).to(pred.device)
        kernel_y = self.kernel_y.repeat(c, 1, 1, 1).to(pred.device)

        # 计算梯度
        pred_grad_x = F.conv2d(pred, kernel_x, padding=1, groups=c)
        pred_grad_y = F.conv2d(pred, kernel_y, padding=1, groups=c)
        target_grad_x = F.conv2d(target, kernel_x, padding=1, groups=c)
        target_grad_y = F.conv2d(target, kernel_y, padding=1, groups=c)

        # 梯度差异损失（L1或MSE）
        loss = F.l1_loss(pred_grad_x, target_grad_x, reduction=self.reduction) + \
               F.l1_loss(pred_grad_y, target_grad_y, reduction=self.reduction)

        return self.loss_weight * loss
    
class CombinedLoss(nn.Module):
    """组合损失函数: CharbonnierLoss + 0.1*PerceptualLoss + 0.01*GradientLoss"""
    def __init__(self, 
                 charbonnier_eps=1e-3,
                 perceptual_layers=['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2'],
                 perceptual_weights=[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0],
                 gradient_kernel_size=3,
                 device='cuda'):
        super(CombinedLoss, self).__init__()
        # 初始化各子损失函数
        self.charbonnier = CharbonnierLoss(eps=charbonnier_eps)
        self.perceptual = PerceptualLoss(
            feature_layers=perceptual_layers,
            weights=perceptual_weights,
            device=device
        )
        self.gradient = GradientLoss(kernel_size=gradient_kernel_size)
        
        # 损失权重
        self.perceptual_weight = 0.1
        self.gradient_weight = 0.01

    def forward(self, pred, target):
        # 计算各子损失
        l_char = self.charbonnier(pred, target)
        l_percep = self.perceptual(pred, target)
        l_grad = self.gradient(pred, target)
        
        # 组合损失
        total_loss = l_char + self.perceptual_weight * l_percep + self.gradient_weight * l_grad
        return total_loss
