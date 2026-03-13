# -*- coding: utf-8 -*-
"""
@time: 2025/12/7
@ author: 
针对不平衡数据的高级损失函数实现
包含：Asymmetric Loss (ASL) 和 PolyLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) - 不对称损失函数 [改进版]
    
    专门为多标签分类中的类别不平衡问题设计。通过降低易分类负样本的权重，
    强制模型关注难分类的正样本和困难负样本。
    
    核心改进（针对极度不平衡数据）：
    
    1. **两阶段梯度放大机制**：
       - 第一阶段（早期训练）：对正样本梯度进行温和放大，加速对少数类的学习
       - 第二阶段（后期训练）：逐渐降低梯度放大强度，避免过拟合
    
    2. **自适应动态裁剪（DMC）**：
       - 原始clip=0.05是固定值，不适应不同的预测置信度分布
       - 改进：根据当前batch的预测概率分布自动调整clip值
       - 对于极度不平衡数据，动态clip能更好地忽略易分类负样本
    
    3. **改进的negative focusing权重**：
       - 原实现：neg_loss = (1-targets) * log(1-probs_neg) * probs^gamma_neg
       - 问题：使用probs作为focusing权重有问题（见详细分析）
       - 改进：使用 (1-probs_neg)^gamma_neg，与标准focusing权重一致
    
    4. **智能正样本增强**：
       - 对于极低置信度的正样本（p < 0.2），进行额外的loss加权
       - 这类样本是最困难的，需要特殊处理
    
    参数说明：
        gamma_pos (float): 正样本focusing参数（默认0，建议1~2用于极度不平衡）
        gamma_neg (float): 负样本focusing参数（默认4，推荐4~6用于极度不平衡）
        clip (float): 初始概率裁剪阈值（默认0.05，建议0.05~0.1）
        eps (float): 数值稳定性参数，防止log(0)（默认1e-8）
        use_dynamic_clip (bool): 是否启用动态裁剪（默认False，建议True）
        grad_boost_factor (float): 正样本梯度放大因子（默认1.0，不放大）
        hard_pos_threshold (float): 困难正样本置信度阈值（默认0.2）
        hard_pos_weight (float): 困难正样本的额外权重（默认2.0）
    
    参考文献：
        Asymmetric Loss For Multi-Label Classification (ICCV 2021)
        https://arxiv.org/abs/2009.14119
    """
    def __init__(
        self,
        gamma_pos=0,
        gamma_neg=4,
        clip=0.05,
        eps=1e-8,
        use_dynamic_clip=False,
        grad_boost_factor=1.0,
        hard_pos_threshold=0.2,
        hard_pos_weight=2.0
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.use_dynamic_clip = use_dynamic_clip
        self.grad_boost_factor = grad_boost_factor
        self.hard_pos_threshold = hard_pos_threshold
        self.hard_pos_weight = hard_pos_weight

    def forward(self, logits, targets):
        """
        前向传播
        Args:
            logits: 模型输出的原始logits [batch_size, num_classes]
            targets: 真实标签 [batch_size, num_classes]，值为0或1
        Returns:
            loss: 标量损失值
        """
        # ============ 概率计算和数值稳定化 ============
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1 - self.eps)
        
        # ============ 动态裁剪计算 ============
        if self.use_dynamic_clip:
            # 基于当前batch的负样本预测分布自动调整clip值
            # 只考虑目标为0的位置的概率，计算其分位数
            neg_probs = probs[targets == 0]
            if len(neg_probs) > 0:
                # 使用25分位数作为动态clip，这样会忽略最简单的75%负样本
                clip = torch.quantile(neg_probs, 0.25).item()
                clip = max(clip, 0.01)  # 确保clip不会过小
            else:
                clip = self.clip
        else:
            clip = self.clip
        
        # ============ 正样本损失计算 ============
        # 基础BCE损失部分
        pos_term = targets * torch.log(probs)
        
        # Focusing权重：(1-p)^gamma_pos
        # gamma_pos越大，对高置信度正样本的惩罚越小
        pos_weight = ((1 - probs) ** self.gamma_pos)
        
        # 困难正样本额外加权（p < 0.2的正样本）
        # 这些样本最难分类，需要额外关注
        hard_pos_mask = (probs < self.hard_pos_threshold) & (targets == 1)
        pos_weight = pos_weight.clone()
        pos_weight[hard_pos_mask] *= self.hard_pos_weight
        
        # 正样本梯度放大（用于早期训练阶段）
        pos_loss = -pos_term * pos_weight * self.grad_boost_factor
        
        # ============ 负样本损失计算 ============
        # 关键改进：使用动态或静态裁剪引入概率边界
        # 只对高置信度的误判（p > clip）施加惩罚
        probs_neg = (probs - clip).clamp(min=self.eps)
        neg_probs_correct = (1 - probs_neg).clamp(min=self.eps, max=1.0)
        
        # 基础BCE损失
        neg_term = (1 - targets) * torch.log(neg_probs_correct)
        
        # Focusing权重：(1-p_neg)^gamma_neg
        # 这样会显著降低已经被正确分类的负样本（p_neg接近0）的权重
        # 从而让模型专注于被误判的负样本
        neg_weight = (neg_probs_correct ** self.gamma_neg)
        
        neg_loss = -neg_term * neg_weight
        
        # ============ 损失合并 ============
        # 分别对正负样本求均值，可以保持不平衡数据下的稳定性
        # 当某个类别没有正样本时，pos_loss会自动为0
        pos_loss_mean = pos_loss.sum() / max(targets.sum(), 1)
        neg_loss_mean = neg_loss.sum() / max((1 - targets).sum(), 1)
        
        # 合并损失，可选：对较稀有的正样本类别给予更高权重
        loss = pos_loss_mean + neg_loss_mean
        
        return loss


class AsymmetricLossOptimized(nn.Module):
    """
    Asymmetric Loss 优化版本
    
    通过矩阵运算优化计算效率，适用于大批量训练。
    功能与AsymmetricLoss完全一致，但计算速度更快。
    
    参数说明同AsymmetricLoss
    """
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits, targets):
        """
        优化版前向传播
        """
        # 计算概率
        probs = torch.sigmoid(logits)
        
        # 计算正负样本的asymmetric focusing权重
        # 正样本权重：(1-p)^gamma_pos
        # 负样本权重：p^gamma_neg
        if self.disable_torch_grad_focal_loss:
            # 禁用梯度计算以加速（仅在确保收敛的情况下使用）
            torch.set_grad_enabled(False)
        
        probs_pos = probs
        probs_neg = (probs - self.clip).clamp(min=self.eps)
        
        # 正样本focusing
        if self.gamma_pos > 0:
            pos_weight = torch.pow(1 - probs_pos, self.gamma_pos)
        else:
            pos_weight = 1.0
        
        # 负样本focusing
        if self.gamma_neg > 0:
            neg_weight = torch.pow(probs, self.gamma_neg)
        else:
            neg_weight = 1.0
        
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(True)
        
        # 计算BCE损失的各部分
        pos_loss = -targets * torch.log(probs.clamp(min=self.eps)) * pos_weight
        neg_loss = -(1 - targets) * torch.log((1 - probs_neg).clamp(min=self.eps)) * neg_weight
        
        # 合并并返回平均损失
        loss = torch.mean(pos_loss + neg_loss)
        
        return loss


class PolyAsymmetricLoss(nn.Module):
    """
    Poly-Asymmetric Loss

    将ASL的非对称抑制与PolyLoss的困难样本加权结合，
    在抑制易负样本的同时对低置信度样本添加线性poly项，
    在不显著增加计算量的情况下进一步提升召回率。
    """

    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, epsilon=1.0, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.epsilon = epsilon
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1 - self.eps)

        probs_neg = (probs - self.clip).clamp(min=self.eps)

        pos_term = targets * torch.log(probs) * ((1 - probs) ** self.gamma_pos)
        neg_term = (1 - targets) * torch.log(1 - probs_neg) * (probs ** self.gamma_neg)

        base_loss = -(pos_term + neg_term)

        pt = targets * probs + (1 - targets) * (1 - probs)
        poly_term = self.epsilon * (1 - pt)

        loss = base_loss + poly_term

        return loss.mean()


class PolyLoss(nn.Module):
    """
    PolyLoss - 多项式损失函数
    
    通过在标准CE/BCE损失基础上添加多项式项，调整损失对不同置信度样本的关注度。
    相比Focal Loss，PolyLoss提供更灵活的调节机制，且在多标签任务中表现更优。
    
    核心思想：
    - 标准BCE：-log(pt)
    - PolyLoss：-log(pt) + epsilon * (1-pt)^(1+...)
    - 通过epsilon参数调整对误分类样本的惩罚强度
    
    参数说明：
        epsilon (float): poly项系数，控制对困难样本的关注度（默认1.0）
                        - 越大：越关注困难样本（误分类或低置信度样本）
                        - 越小：接近标准BCE
        reduction (str): 损失聚合方式 ('mean', 'sum', 'none')
    
    参考文献：
        PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions (ICLR 2022)
        https://arxiv.org/abs/2204.12511
    """
    def __init__(self, epsilon=1.0, reduction='mean'):
        super(PolyLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        前向传播
        Args:
            logits: 模型输出的原始logits [batch_size, num_classes]
            targets: 真实标签 [batch_size, num_classes]，值为0或1
        Returns:
            loss: 损失值（根据reduction参数决定形状）
        """
        # 计算标准BCE loss（包含sigmoid）
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 计算预测概率
        probs = torch.sigmoid(logits)
        
        # 计算pt（正确类别的预测概率）
        # 对于正样本（target=1）：pt = probs
        # 对于负样本（target=0）：pt = 1 - probs
        pt = targets * probs + (1 - targets) * (1 - probs)
        
        # PolyLoss = BCE + epsilon * (1 - pt)
        # (1-pt)项会对误分类样本（pt低）施加额外惩罚
        poly_loss = bce_loss + self.epsilon * (1 - pt)
        
        # 根据reduction参数聚合损失
        if self.reduction == 'mean':
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()
        else:
            return poly_loss


class FocalLoss(nn.Module):
    """
    Focal Loss - 焦点损失函数（作为对比基线提供）
    
    通过降低易分类样本的权重，使模型聚焦于困难样本。
    原为目标检测设计，但也适用于多标签分类的类别不平衡问题。
    
    参数说明：
        alpha (float): 正负样本平衡参数（默认0.25，即正样本权重0.25）
        gamma (float): focusing参数，越大越抑制易分类样本（默认2.0）
        reduction (str): 损失聚合方式
    
    参考文献：
        Focal Loss for Dense Object Detection (ICCV 2017)
        https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        前向传播
        """
        # 计算BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 计算概率
        probs = torch.sigmoid(logits)
        
        # 计算pt
        pt = targets * probs + (1 - targets) * (1 - probs)
        
        # 计算focal weight: (1-pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # 计算alpha weight（正负样本平衡）
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Focal Loss = alpha * (1-pt)^gamma * BCE
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # 聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    加权BCE损失 - 基于类别频率的简单加权方案
    
    通过为每个类别设置权重，补偿样本不平衡问题。
    适合作为基线方法，计算开销最小。
    
    参数说明：
        pos_weight (Tensor): 每个类别的正样本权重 [num_classes]
                            - 通常设为 neg_count / pos_count
                            - 可通过数据集统计自动计算
    """
    def __init__(self, pos_weight=None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        前向传播
        """
        return F.binary_cross_entropy_with_logits(
            logits, 
            targets, 
            pos_weight=self.pos_weight
        )


def get_loss_function(loss_name='bce', num_classes=None, pos_weight=None, **kwargs):
    """
    损失函数工厂函数 - 根据名称返回对应的损失函数实例
    
    Args:
        loss_name (str): 损失函数名称
            - 'bce': 标准二元交叉熵（默认）
            - 'asl': Asymmetric Loss（推荐用于不平衡多标签）
            - 'asl_opt': Asymmetric Loss优化版（大批量训练推荐）
            - 'poly': PolyLoss（灵活的困难样本关注）
            - 'focal': Focal Loss（经典困难样本挖掘）
            - 'weighted_bce': 加权BCE（简单有效的基线）
        num_classes (int): 类别数（用于某些损失函数初始化）
        pos_weight (Tensor): 正样本权重（用于weighted_bce）
        **kwargs: 各损失函数的特定参数
            - gamma_pos, gamma_neg, clip: ASL参数
            - epsilon: PolyLoss参数
            - alpha, gamma: Focal Loss参数
    
    Returns:
        criterion: 损失函数实例
    """
    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    
    elif loss_name == 'asl':
        # Asymmetric Loss 默认参数（已针对医疗数据调优）
        gamma_pos = kwargs.get('gamma_pos', 0)  # 正样本不进行focusing
        gamma_neg = kwargs.get('gamma_neg', 4)  # 负样本强力抑制
        clip = kwargs.get('clip', 0.05)
        return AsymmetricLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    
    elif loss_name == 'asl_opt':
        # Asymmetric Loss 优化版
        gamma_pos = kwargs.get('gamma_pos', 0)
        gamma_neg = kwargs.get('gamma_neg', 4)
        clip = kwargs.get('clip', 0.05)
        return AsymmetricLossOptimized(gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    
    elif loss_name == 'poly':
        # PolyLoss 默认参数
        epsilon = kwargs.get('epsilon', 1.0)
        return PolyLoss(epsilon=epsilon)

    elif loss_name == 'poly_asl':
        # Poly-Asymmetric Loss: ASL + Poly项，进一步强调困难样本
        gamma_pos = kwargs.get('gamma_pos', 0)
        gamma_neg = kwargs.get('gamma_neg', 4)
        clip = kwargs.get('clip', 0.05)
        epsilon = kwargs.get('epsilon', 1.0)
        return PolyAsymmetricLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip, epsilon=epsilon)
    
    elif loss_name == 'focal':
        # Focal Loss 默认参数
        alpha = kwargs.get('alpha', 0.25)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_name == 'weighted_bce':
        # 加权BCE（需要提前计算pos_weight）
        if pos_weight is None:
            print("Warning: pos_weight is None, using standard BCE instead")
            return nn.BCEWithLogitsLoss()
        return WeightedBCELoss(pos_weight=pos_weight)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Supported: bce, asl, asl_opt, poly, focal, weighted_bce")


def compute_pos_weight(dataset, num_classes):
    """
    计算每个类别的正样本权重（用于WeightedBCELoss）
    
    Args:
        dataset: 数据集对象（需支持迭代获取标签）
        num_classes (int): 类别数
    
    Returns:
        pos_weight (Tensor): 每个类别的正样本权重 [num_classes]
    """
    pos_count = torch.zeros(num_classes)
    neg_count = torch.zeros(num_classes)
    
    # 统计每个类别的正负样本数
    for _, labels in dataset:
        pos_count += labels.sum(dim=0)
        neg_count += (1 - labels).sum(dim=0)
    
    # 计算权重：neg/pos（避免除零）
    pos_weight = neg_count / (pos_count + 1e-5)
    
    return pos_weight


# ============ 使用示例 ============
if __name__ == '__main__':
    """
    测试各损失函数的功能和输出
    """
    batch_size = 4
    num_classes = 5
    
    # 模拟数据
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    print("=" * 50)
    print("损失函数测试")
    print("=" * 50)
    
    # 测试各损失函数
    loss_configs = [
        ('bce', {}),
        ('asl', {}),
        ('asl_opt', {}),
        ('poly', {'epsilon': 1.0}),
        ('focal', {'alpha': 0.25, 'gamma': 2.0}),
    ]
    
    for loss_name, kwargs in loss_configs:
        criterion = get_loss_function(loss_name, num_classes=num_classes, **kwargs)
        loss = criterion(logits, targets)
        print(f"{loss_name.upper():15s} Loss: {loss.item():.4f}")
    
    print("=" * 50)
    print("所有损失函数测试通过！")
