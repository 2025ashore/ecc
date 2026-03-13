# -*- coding: utf-8 -*-
"""
双曲空间标签嵌入模块 (Hyperbolic Label Embedding)

核心创新：
    1. 在Poincaré球模型中学习ECG标签嵌入
    2. 利用双曲空间的层级表示能力建模心律失常的父-子类关系
    3. 双曲注意力机制实现标签感知的特征聚合
    4. 双曲距离度量用于多标签分类

理论背景：
    - 双曲空间天然适合表示树状/层级结构
    - PTB-XL标签存在 superdiagnostic → diagnostic → subdiagnostic 层级
    - 双曲空间的"边缘膨胀"特性使其能更高效地嵌入层级数据

参考文献：
    - Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical Representations", NeurIPS 2017
    - Ganea et al., "Hyperbolic Neural Networks", NeurIPS 2018
    - Chen et al., "Hyperbolic Vision Transformers", CVPR 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== Poincaré球基础运算 ====================

class PoincareOperations:
    """Poincaré球模型的基本数学运算
    
    Poincaré球: B^n = {x ∈ R^n : ||x|| < 1}
    曲率: c > 0 (默认c=1)
    """
    
    EPS = 1e-5  # 数值稳定性常数
    MAX_NORM = 1 - 1e-3  # 防止点落在球面边界上
    
    @staticmethod
    def project(x, c=1.0):
        """将欧几里得向量投影到Poincaré球内
        
        Args:
            x: 欧几里得向量 (*, d)
            c: 曲率参数
        Returns:
            投影后的双曲向量
        """
        norm = torch.clamp(x.norm(dim=-1, keepdim=True), min=PoincareOperations.EPS)
        max_norm = PoincareOperations.MAX_NORM / math.sqrt(c)
        cond = norm > max_norm
        projected = x / norm * max_norm
        return torch.where(cond, projected, x)
    
    @staticmethod
    def mobius_add(x, y, c=1.0):
        """Möbius加法：双曲空间中的加法操作
        
        公式: x ⊕_c y = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) / 
                        (1 + 2c<x,y> + c²||x||²||y||²)
        
        Args:
            x, y: Poincaré球中的点 (*, d)
            c: 曲率
        Returns:
            x ⊕_c y
        """
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        
        return PoincareOperations.project(num / torch.clamp(denom, min=PoincareOperations.EPS), c)
    
    @staticmethod
    def mobius_matvec(M, x, c=1.0):
        """Möbius矩阵-向量乘法：双曲空间中的线性变换
        
        Args:
            M: 变换矩阵 (d_out, d_in)
            x: Poincaré球中的点 (*, d_in)
            c: 曲率
        Returns:
            变换后的点 (*, d_out)
        """
        x_norm = torch.clamp(x.norm(dim=-1, keepdim=True), min=PoincareOperations.EPS)
        Mx = F.linear(x, M)  # 在切空间做线性变换
        Mx_norm = torch.clamp(Mx.norm(dim=-1, keepdim=True), min=PoincareOperations.EPS)
        
        # 使用指数映射和对数映射的近似
        result = Mx / Mx_norm * torch.tanh(Mx_norm / x_norm * torch.atanh(torch.clamp(x_norm, max=1-PoincareOperations.EPS)))
        
        return PoincareOperations.project(result, c)
    
    @staticmethod
    def expmap0(v, c=1.0):
        """从原点的指数映射：切空间 → Poincaré球
        
        公式: exp_0^c(v) = tanh(√c ||v||) * v / (√c ||v||)
        
        Args:
            v: 切空间中的向量 (*, d)
            c: 曲率
        Returns:
            Poincaré球中的点
        """
        sqrt_c = math.sqrt(c)
        v_norm = torch.clamp(v.norm(dim=-1, keepdim=True), min=PoincareOperations.EPS)
        return PoincareOperations.project(torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm), c)
    
    @staticmethod
    def logmap0(y, c=1.0):
        """到原点的对数映射：Poincaré球 → 切空间
        
        公式: log_0^c(y) = arctanh(√c ||y||) * y / (√c ||y||)
        
        Args:
            y: Poincaré球中的点 (*, d)
            c: 曲率
        Returns:
            切空间中的向量
        """
        sqrt_c = math.sqrt(c)
        y_norm = torch.clamp(y.norm(dim=-1, keepdim=True), min=PoincareOperations.EPS)
        return torch.atanh(torch.clamp(sqrt_c * y_norm, max=1-PoincareOperations.EPS)) * y / (sqrt_c * y_norm)
    
    @staticmethod
    def poincare_distance(x, y, c=1.0):
        """Poincaré球中的双曲距离
        
        公式: d_c(x, y) = (2/√c) * arctanh(√c ||−x ⊕_c y||)
        
        Args:
            x, y: Poincaré球中的点 (*, d)
            c: 曲率
        Returns:
            双曲距离标量
        """
        sqrt_c = math.sqrt(c)
        mob_add = PoincareOperations.mobius_add(-x, y, c)
        mob_norm = torch.clamp(mob_add.norm(dim=-1), min=PoincareOperations.EPS, max=1-PoincareOperations.EPS)
        return (2 / sqrt_c) * torch.atanh(sqrt_c * mob_norm)
    
    @staticmethod
    def lambda_x(x, c=1.0):
        """共形因子 λ_x^c = 2 / (1 - c||x||²)
        
        用于计算切空间的缩放
        """
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        return 2 / torch.clamp(1 - c * x2, min=PoincareOperations.EPS)


# ==================== 双曲神经网络层 ====================

class HyperbolicLinear(nn.Module):
    """双曲线性层：在Poincaré球中进行线性变换
    
    实现: y = exp_0(W · log_0(x) + b)
    即先映射到切空间，做欧几里得线性变换，再映射回双曲空间
    """
    
    def __init__(self, in_features, out_features, c=1.0, bias=True):
        super(HyperbolicLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=0.01)  # 小初始化防止梯度爆炸
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        """
        Args:
            x: 双曲空间中的点 (B, *, in_features)
        Returns:
            变换后的点 (B, *, out_features)
        """
        # 映射到切空间
        x_tan = PoincareOperations.logmap0(x, self.c)
        
        # 欧几里得线性变换
        out_tan = F.linear(x_tan, self.weight, self.bias)
        
        # 映射回双曲空间
        out = PoincareOperations.expmap0(out_tan, self.c)
        
        return out


class HyperbolicLabelEmbedding(nn.Module):
    """双曲标签嵌入层
    
    为每个ECG标签在Poincaré球中学习一个嵌入向量
    层级关系通过双曲距离自然编码（父类靠近原点，子类靠近边缘）
    """
    
    def __init__(self, num_classes, embed_dim=128, c=1.0, init_scale=0.01):
        super(HyperbolicLabelEmbedding, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.c = c
        
        # 在切空间初始化嵌入，然后映射到双曲空间
        self.label_embeds_tan = nn.Parameter(torch.randn(num_classes, embed_dim) * init_scale)
    
    def forward(self):
        """获取所有标签的双曲嵌入
        
        Returns:
            label_embeds: (num_classes, embed_dim) 在Poincaré球中
        """
        return PoincareOperations.expmap0(self.label_embeds_tan, self.c)
    
    def get_label_embed(self, label_ids):
        """获取指定标签的双曲嵌入
        
        Args:
            label_ids: 标签索引 (*)
        Returns:
            embeds: 对应的双曲嵌入 (*, embed_dim)
        """
        all_embeds = self.forward()
        return all_embeds[label_ids]


class HyperbolicAttention(nn.Module):
    """双曲注意力机制
    
    在Poincaré球中计算Query-Key-Value注意力
    使用双曲距离替代欧几里得点积
    """
    
    def __init__(self, embed_dim, num_heads=4, c=1.0, dropout=0.1):
        super(HyperbolicAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.c = c
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # 在切空间做投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, return_attn=False):
        """
        Args:
            query: 查询向量，双曲空间 (B, Q, D)
            key: 键向量，双曲空间 (B, K, D)
            value: 值向量，双曲空间 (B, K, D)
        Returns:
            output: 注意力输出 (B, Q, D)
            attn_weights: (可选) 注意力权重 (B, num_heads, Q, K)
        """
        B, Q, D = query.shape
        K = key.shape[1]
        
        # 映射到切空间进行投影
        q_tan = PoincareOperations.logmap0(query, self.c)
        k_tan = PoincareOperations.logmap0(key, self.c)
        v_tan = PoincareOperations.logmap0(value, self.c)
        
        # 线性投影
        q = self.q_proj(q_tan).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_tan).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_tan).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 映射回双曲空间计算距离
        q_hyp = PoincareOperations.expmap0(q, self.c)
        k_hyp = PoincareOperations.expmap0(k, self.c)
        
        # 计算双曲注意力分数（基于负距离）
        # 距离越近，注意力越高
        # attn_scores[b, h, i, j] = -d(q[b,h,i], k[b,h,j])
        attn_scores = -self._pairwise_distance(q_hyp, k_hyp)  # (B, H, Q, K)
        attn_scores = attn_scores / self.scale
        
        # Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 在切空间聚合value（使用欧几里得加权和作为近似）
        attn_output = torch.matmul(attn_weights, v)  # (B, H, Q, head_dim)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Q, D)
        
        # 输出投影
        output_tan = self.out_proj(attn_output)
        
        # 映射回双曲空间
        output = PoincareOperations.expmap0(output_tan, self.c)
        
        if return_attn:
            return output, attn_weights
        return output
    
    def _pairwise_distance(self, x, y):
        """计算双曲空间中的成对距离
        
        Args:
            x: (B, H, Q, d)
            y: (B, H, K, d)
        Returns:
            dist: (B, H, Q, K)
        """
        B, H, Q, d = x.shape
        K = y.shape[2]
        
        # 扩展维度以计算成对距离
        x_exp = x.unsqueeze(3).expand(B, H, Q, K, d)  # (B, H, Q, K, d)
        y_exp = y.unsqueeze(2).expand(B, H, Q, K, d)  # (B, H, Q, K, d)
        
        # 计算双曲距离
        dist = PoincareOperations.poincare_distance(
            x_exp.reshape(-1, d), 
            y_exp.reshape(-1, d), 
            self.c
        )
        
        return dist.view(B, H, Q, K)


# ==================== 双曲标签分类头 ====================

class HyperbolicLabelClassifier(nn.Module):
    """双曲标签分类器
    
    核心思想：
        1. 特征和标签都嵌入到同一双曲空间
        2. 分类基于特征与标签嵌入的双曲距离
        3. 距离越近，属于该标签的概率越高
    
    优势：
        - 自然建模标签层级关系
        - 更好的少样本标签泛化能力
        - 可解释的标签相似性（通过双曲距离）
    """
    
    def __init__(
        self, 
        input_dim, 
        num_classes, 
        hyperbolic_dim=128,
        c=1.0,
        num_attn_heads=4,
        dropout=0.1,
        use_label_hierarchy=True,
        hierarchy_loss_weight=0.1,
    ):
        super(HyperbolicLabelClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hyperbolic_dim = hyperbolic_dim
        self.c = c
        self.use_label_hierarchy = use_label_hierarchy
        self.hierarchy_loss_weight = hierarchy_loss_weight
        
        # 特征投影到双曲空间
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hyperbolic_dim),
            nn.LayerNorm(hyperbolic_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hyperbolic_dim, hyperbolic_dim),
        )
        
        # 双曲标签嵌入
        self.label_embedding = HyperbolicLabelEmbedding(
            num_classes=num_classes,
            embed_dim=hyperbolic_dim,
            c=c,
            init_scale=0.05,
        )
        
        # 双曲注意力：特征作为query，标签作为key/value
        self.hyper_attention = HyperbolicAttention(
            embed_dim=hyperbolic_dim,
            num_heads=num_attn_heads,
            c=c,
            dropout=dropout,
        )
        
        # 距离到logits的映射（可学习温度参数）
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # 残差分类头（与双曲距离分类并行）
        self.residual_classifier = nn.Linear(hyperbolic_dim, num_classes)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
        # 标签层级约束（可选）
        if use_label_hierarchy:
            # 层级关系矩阵：hierarchy_mask[i,j]=1 表示标签i是标签j的父类
            # 需要在外部设置，这里先初始化为空
            self.register_buffer('hierarchy_mask', torch.zeros(num_classes, num_classes))
    
    def set_label_hierarchy(self, hierarchy_dict):
        """设置标签层级关系
        
        Args:
            hierarchy_dict: {parent_idx: [child_idx1, child_idx2, ...], ...}
        """
        mask = torch.zeros(self.num_classes, self.num_classes)
        for parent, children in hierarchy_dict.items():
            for child in children:
                mask[parent, child] = 1
        self.hierarchy_mask = mask.to(self.label_embedding.label_embeds_tan.device)
    
    def forward(self, features, return_distances=False):
        """
        Args:
            features: 特征向量 (B, input_dim)
            return_distances: 是否返回双曲距离
        Returns:
            logits: 分类logits (B, num_classes)
            distances: (可选) 双曲距离 (B, num_classes)
        """
        B = features.shape[0]
        
        # 1. 特征投影到切空间，再映射到双曲空间
        feat_tan = self.feature_proj(features)  # (B, hyperbolic_dim)
        feat_hyp = PoincareOperations.expmap0(feat_tan, self.c)  # (B, hyperbolic_dim)
        
        # 2. 获取所有标签的双曲嵌入
        label_embeds = self.label_embedding()  # (num_classes, hyperbolic_dim)
        
        # 3. 计算特征与每个标签嵌入的双曲距离
        # feat_hyp: (B, D), label_embeds: (C, D)
        feat_exp = feat_hyp.unsqueeze(1).expand(B, self.num_classes, self.hyperbolic_dim)
        label_exp = label_embeds.unsqueeze(0).expand(B, self.num_classes, self.hyperbolic_dim)
        
        distances = PoincareOperations.poincare_distance(
            feat_exp.reshape(-1, self.hyperbolic_dim),
            label_exp.reshape(-1, self.hyperbolic_dim),
            self.c
        ).view(B, self.num_classes)  # (B, C)
        
        # 4. 距离转换为logits（距离越小，logits越大）
        distance_logits = -distances / torch.clamp(self.temperature, min=0.1)
        
        # 5. 残差分类（欧几里得空间的补充）
        residual_logits = self.residual_classifier(feat_tan)
        
        # 6. 融合双曲距离logits和残差logits
        logits = distance_logits + self.residual_scale * residual_logits
        
        if return_distances:
            return logits, distances
        return logits
    
    def compute_hierarchy_loss(self):
        """计算层级约束损失
        
        约束：父类标签应比子类标签更靠近原点（双曲空间中心）
        """
        if not self.use_label_hierarchy or self.hierarchy_mask.sum() == 0:
            return torch.tensor(0.0, device=self.label_embedding.label_embeds_tan.device)
        
        label_embeds = self.label_embedding()  # (C, D)
        
        # 计算每个标签到原点的距离（范数）
        origin = torch.zeros_like(label_embeds[0])
        distances_to_origin = PoincareOperations.poincare_distance(
            label_embeds,
            origin.unsqueeze(0).expand_as(label_embeds),
            self.c
        )  # (C,)
        
        # 层级约束：父类距离 < 子类距离
        # loss = max(0, parent_dist - child_dist + margin)
        margin = 0.1
        loss = 0.0
        count = 0
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if self.hierarchy_mask[i, j] > 0:  # i是j的父类
                    # 父类i应该比子类j更靠近原点
                    loss += F.relu(distances_to_origin[i] - distances_to_origin[j] + margin)
                    count += 1
        
        if count > 0:
            loss = loss / count
        
        return self.hierarchy_loss_weight * loss
    
    def get_label_similarities(self):
        """获取标签间的双曲相似性矩阵（用于可视化）
        
        Returns:
            sim_matrix: (num_classes, num_classes) 标签相似性
        """
        label_embeds = self.label_embedding()  # (C, D)
        C, D = label_embeds.shape
        
        # 计算成对距离
        dist_matrix = torch.zeros(C, C, device=label_embeds.device)
        for i in range(C):
            for j in range(C):
                dist_matrix[i, j] = PoincareOperations.poincare_distance(
                    label_embeds[i], label_embeds[j], self.c
                )
        
        # 距离转相似性
        sim_matrix = torch.exp(-dist_matrix)
        
        return sim_matrix


# ==================== 组合模块：双曲标签感知特征聚合 ====================

class HyperbolicLabelConditionedFusion(nn.Module):
    """双曲标签条件特征融合
    
    将视图特征和标签嵌入都映射到双曲空间，
    每个标签作为query从视图特征中聚合相关信息
    """
    
    def __init__(
        self,
        feat_dim=128,
        num_classes=9,
        num_views=7,
        c=1.0,
        num_heads=4,
        dropout=0.1,
    ):
        super(HyperbolicLabelConditionedFusion, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_views = num_views
        self.c = c
        
        # 视图特征投影到双曲空间
        self.view_proj = nn.Linear(feat_dim, feat_dim)
        
        # 双曲标签嵌入
        self.label_embedding = HyperbolicLabelEmbedding(
            num_classes=num_classes,
            embed_dim=feat_dim,
            c=c,
        )
        
        # 双曲注意力
        self.hyper_attn = HyperbolicAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            c=c,
            dropout=dropout,
        )
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(feat_dim)
        
        # 每标签分类器
        self.label_classifiers = nn.Parameter(torch.zeros(num_classes, feat_dim))
        self.label_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.normal_(self.label_classifiers, mean=0, std=0.02)
    
    def forward(self, view_features, return_attn=False):
        """
        Args:
            view_features: 视图特征 (B, num_views, feat_dim)
        Returns:
            logits: 标签条件logits (B, num_classes)
            attn_weights: (可选) 注意力权重 (B, num_heads, num_classes, num_views)
        """
        B = view_features.shape[0]
        
        # 1. 视图特征投影到双曲空间
        view_tan = self.view_proj(view_features)  # (B, V, D)
        view_hyp = PoincareOperations.expmap0(view_tan, self.c)  # (B, V, D)
        
        # 2. 获取标签嵌入并扩展batch维度
        label_embeds = self.label_embedding()  # (C, D)
        label_hyp = label_embeds.unsqueeze(0).expand(B, -1, -1)  # (B, C, D)
        
        # 3. 双曲注意力：标签作为query，视图作为key/value
        label_feat, attn_weights = self.hyper_attn(
            query=label_hyp,
            key=view_hyp,
            value=view_hyp,
            return_attn=True,
        )  # label_feat: (B, C, D)
        
        # 4. 映射回切空间做分类
        label_feat_tan = PoincareOperations.logmap0(label_feat, self.c)
        label_feat_tan = self.output_norm(label_feat_tan)
        
        # 5. 每标签独立分类
        # logits[b, c] = <label_feat[b, c], classifier[c]> + bias[c]
        logits = (label_feat_tan * self.label_classifiers.unsqueeze(0)).sum(dim=-1) + self.label_bias
        
        if return_attn:
            return logits, attn_weights
        return logits


# ==================== 层级一致性损失 ====================

class HierarchyConsistencyLoss(nn.Module):
    """层级一致性损失
    
    确保父类标签的预测概率不小于子类标签的预测概率。
    
    原理：
        - 如果预测某个子类为正，则父类也应该为正
        - loss = max(0, p_child - p_parent + margin)
        - 这保持了标签的层级一致性
    
    Args:
        parent_child_pairs: 父子关系列表 [(parent_idx, child_idx), ...]
        margin: 边距参数 (默认0.1)
    """
    
    def __init__(self, parent_child_pairs, margin=0.1):
        super(HierarchyConsistencyLoss, self).__init__()
        self.parent_child_pairs = parent_child_pairs
        self.margin = margin
    
    def forward(self, probs, targets=None):
        """
        Args:
            probs: 预测概率 (B, C)
            targets: 真实标签 (可选，用于仅对正样本计算)
        
        Returns:
            loss: 层级一致性损失
        """
        if len(self.parent_child_pairs) == 0:
            return torch.tensor(0.0, device=probs.device)
        
        total_loss = 0.0
        count = 0
        
        for parent_idx, child_idx in self.parent_child_pairs:
            # 子类概率不应超过父类概率（加边距）
            # loss = relu(p_child - p_parent + margin)
            violation = F.relu(probs[:, child_idx] - probs[:, parent_idx] + self.margin)
            
            # 如果提供targets，仅对子类为正的样本计算
            if targets is not None:
                mask = targets[:, child_idx] > 0.5
                if mask.sum() > 0:
                    violation = violation[mask]
                else:
                    continue
            
            total_loss = total_loss + violation.mean()
            count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=probs.device)
        
        return total_loss / count


# ==================== PTB-XL 标签层级定义 ====================

def get_ptbxl_label_hierarchy():
    """获取PTB-XL数据集的标签层级结构
    
    层级结构（示例，需要根据实际数据集调整）：
        - superdiagnostic (5类): NORM, MI, STTC, CD, HYP
            - diagnostic (23类): ...
                - subdiagnostic (更细分类)
    
    Returns:
        hierarchy_dict: {parent_idx: [child_idx1, ...], ...}
    """
    # 这是一个示例结构，需要根据实际PTB-XL标签映射调整
    # 返回空字典表示不使用层级约束
    hierarchy = {}
    
    # 示例：如果使用superdiagnostic实验
    # hierarchy = {
    #     0: [5, 6, 7],  # NORM 的子类
    #     1: [8, 9, 10, 11],  # MI 的子类
    #     ...
    # }
    
    return hierarchy


if __name__ == "__main__":
    # 测试代码
    print("Testing Hyperbolic Label Embedding Module...")
    
    # 测试Poincaré球操作
    x = torch.randn(2, 128) * 0.1
    y = torch.randn(2, 128) * 0.1
    
    x_proj = PoincareOperations.project(x)
    print(f"Projected x norm: {x_proj.norm(dim=-1)}")  # 应该 < 1
    
    dist = PoincareOperations.poincare_distance(x_proj, PoincareOperations.project(y))
    print(f"Poincaré distance: {dist}")
    
    # 测试双曲标签分类器
    classifier = HyperbolicLabelClassifier(
        input_dim=128,
        num_classes=9,
        hyperbolic_dim=128,
    )
    
    features = torch.randn(4, 128)
    logits, distances = classifier(features, return_distances=True)
    print(f"Logits shape: {logits.shape}")  # (4, 9)
    print(f"Distances shape: {distances.shape}")  # (4, 9)
    
    # 测试标签相似性
    sim = classifier.get_label_similarities()
    print(f"Label similarity matrix shape: {sim.shape}")  # (9, 9)
    
    # 测试双曲标签条件融合
    fusion = HyperbolicLabelConditionedFusion(
        feat_dim=128,
        num_classes=9,
        num_views=7,
    )
    
    view_feats = torch.randn(4, 7, 128)
    logits, attn = fusion(view_feats, return_attn=True)
    print(f"Fusion logits shape: {logits.shape}")  # (4, 9)
    print(f"Attention shape: {attn.shape}")  # (4, num_heads, 9, 7)
    
    print("\nAll tests passed!")
