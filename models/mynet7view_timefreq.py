# -*- coding: utf-8 -*-
"""
@time: 2026/1/5
@description: MyNet6View + 时频融合视图 + Sobel形态梯度视图 = MyNet7ViewTimeFreq (8视图)
    
核心设计：
    - 保留原有6个视图分支（多导联分组）
    - 第7个视图：时频融合分支（全12导联）
    - 第8个视图：Sobel形态梯度分支（全12导联一阶微分）← 新增创新点
    - 统一融合机制：8个视图的自适应权重融合
    
架构图：
    12导联 ECG 输入
    ├─ 视图1-6: 导联分组（MyNet分支） → 128维特征
    ├─ 视图7: 全导联时频 (LearnableSTFT + SpectrogramCNN) → 128维特征
    └─ 视图8: Sobel形态梯度 (Sobel Conv → MyNet) → 128维特征 ← 新增
         ↓
    8个视图的自适应权重
         ↓
    加权融合 → 128维融合特征
         ↓
    分类头 → num_classes 输出
    
Sobel形态梯度视图创新点：
    心电信号的多标签诊断（束支阻滞、ST段异常等）高度依赖波形的斜率变化。
    Sobel算子本质上是高效的一阶微分，通过不可学习的 [-1, 0, 1] 卷积核对每个
    导联独立提取梯度信号，再经MyNet骨干网络提取深层特征。
    这是传统信号处理先验（微分算子）与深度学习的有机结合，
    强制模型从"变化率"角度审视心电图，而非仅依赖原始幅值。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import CoordAtt, CrossModalAttention
from models.model import Mish, Res2Block, AdaptiveWeight, MyNet
from models.timefreq_fusion import LearnableSTFT, SpectrogramCNN
from models.label_graph import LabelGraphRefiner, FeatureLabelGCN


class TimeFreqView(nn.Module):
    """时频融合视图分支：输入12导联→输出128维特征
    
    设计思路：
        1. LearnableSTFT：多尺度频谱分析
        2. SpectrogramCNN：频谱特征提取
        3. 特征压缩：降到128维，匹配其他视图输出
    """
    
    def __init__(self, input_channels=12, output_dim=128,
                 use_lead_attention=True, lead_attention_reduction=4,
                 lead_attention_spectral_spatial=True, lead_attention_spatial_kernel=7):
        super(TimeFreqView, self).__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        
        # 多尺度STFT（输入12导联）
        self.stft = LearnableSTFT(
            n_fft=512,
            hop_length=128,
            n_scales=3
        )
        
        # 谱图CNN分支（输入通道=12，输出特征=512）
        self.spectrogram_cnn = SpectrogramCNN(
            input_channels=input_channels,
            num_classes=output_dim,  # 直接输出目标维度，避免额外映射
            hidden_dims=[64, 128, 256, 512],
            use_lead_attention=use_lead_attention,
            lead_attention_reduction=lead_attention_reduction,
            lead_attention_spectral_spatial=lead_attention_spectral_spatial,
            lead_attention_spatial_kernel=lead_attention_spatial_kernel,
        )
        
        # 特征投影层（512→128维）
        self.feature_projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
        
        # 全局特征处理（确保稳定性）
        self.output_norm = nn.BatchNorm1d(output_dim)
    
    def forward(self, x):
        """
        参数：
            x: 输入信号，形状 (B, 12, L)
        返回：
            feat: 时频特征，形状 (B, 128)
        """
        # 获取多尺度谱图
        fused_spec, _ = self.stft(x)  # (B, 12, n_freqs, T)
        
        # 谱图CNN提取特征
        spec_feat, _, lead_weights = self.spectrogram_cnn(fused_spec)  # (B, 512)
        
        # 特征投影到128维
        feat = self.feature_projection(spec_feat)  # (B, 128)
        
        # 归一化输出
        feat = self.output_norm(feat)
        
        return feat


class SobelMorphologicalView(nn.Module):
    """Sobel引导的形态梯度视图：通过一阶微分算子提取ECG波形斜率特征
    
    核心思想：
        心电信号的多标签诊断（束支阻滞、ST段异常等）高度依赖波形的斜率变化。
        Sobel算子本质上是高效的一阶微分，可以强制模型从"变化率"视角审视心电图，
        而非仅依赖原始幅值。
    
    实现细节：
        1. 不可学习的1D Sobel卷积 [-1, 0, 1]，对每个导联独立提取一阶梯度
        2. 梯度信号通过 MyNet 骨干网络提取128维深层特征表示
    
    创新性：
        传统信号处理先验（微分算子）与深度学习的有机结合，
        为模型提供显式的"形态变化率"视角，与幅值域视图形成互补。
    """
    
    def __init__(self, input_channels=12, output_dim=128, num_classes=9):
        super(SobelMorphologicalView, self).__init__()
        self.input_channels = input_channels
        
        # ===== 不可学习的1D Sobel卷积 =====
        # 使用 F.conv1d + register_buffer，而非 nn.Conv1d + nn.Parameter
        # 这样 Sobel 核不会出现在 model.parameters() 中，不会被优化器跟踪
        sobel_kernel = torch.tensor([-1.0, 0.0, 1.0]).reshape(1, 1, 3)
        sobel_kernel = sobel_kernel.repeat(input_channels, 1, 1)  # (12, 1, 3)
        self.register_buffer('sobel_kernel', sobel_kernel)  # 自动跟随 .to(device)
        
        # ===== 【关键修复】梯度信号归一化 =====
        # Sobel输出的一阶差分信号统计分布与原始ECG幅值信号差异巨大：
        #   - 原始信号：均值非零，方差取决于ECG幅度
        #   - 梯度信号：近零均值，方差极小（平坦段）或突变（QRS波群）
        # 不归一化会导致骨干网络第一层conv1(kernel=25)收到分布错误的输入，
        # 使得整个Sobel分支学出噪声特征，反向污染融合结果。
        # BatchNorm1d 逐通道（逐导联）归一化梯度信号到标准分布。
        self.gradient_norm = nn.BatchNorm1d(input_channels)
        
        # ===== 特征提取骨干：复用MyNet架构处理梯度信号 =====
        # 输入为12通道归一化梯度信号，输出128维特征
        self.backbone = MyNet(
            input_channels=input_channels,
            single_view=True,
            num_classes=num_classes
        )
    
    def forward(self, x, return_seq=False):
        """前向传播
        
        参数：
            x: 原始12导联ECG信号，形状 (B, 12, L)
            return_seq: 是否返回池化前的序列特征（用于跨模态融合）
        返回：
            feat: 形态梯度特征，形状 (B, 128)
            seq_feat: （可选）池化前序列特征，形状 (B, 128, L'')
        """
        # 第一步：Sobel一阶微分 → 提取各导联的斜率/梯度特征图
        # 使用 F.conv1d + buffer（不经过 nn.Conv1d，避免权重被优化器跟踪）
        gradient = F.conv1d(
            x, self.sobel_kernel, bias=None,
            stride=1, padding=1, groups=self.input_channels
        )  # (B, 12, L)
        
        # 第二步：【关键】归一化梯度信号，使其分布适配骨干网络
        gradient = self.gradient_norm(gradient)  # (B, 12, L)
        
        # 第三步：通过骨干网络提取深层特征
        return self.backbone(gradient, return_seq=return_seq)


class MyNet7ViewTimeFreq(nn.Module):
    """多视图ECG分类模型：6个导联视图 + 1个时频融合视图 + 1个Sobel形态梯度视图
    
    视图划分：
        视图1: 1导联（I）
        视图2: 2导联（aVR, aVL）
        视图3: 2导联（V1, V2）
        视图4: 2导联（V3, V4）
        视图5: 2导联（V5, V6）
        视图6: 3导联（II, III, aVF）
        视图7: 12导联（全导联时频融合）
        视图8: 12导联（Sobel形态梯度） ← 新增
    
    融合策略：
        - 每个视图生成自适应权重
        - 加权融合8个视图特征
        - 最终分类
    """
    
    def __init__(
            self,
            num_classes=9,
            use_label_graph_refiner=False,
            label_graph_hidden=64,
            label_graph_learnable_adj=True,
            label_graph_dropout=0.1,
            use_feature_label_gcn=True,
            feature_label_gcn_hidden=64,
            feature_label_gcn_layers=2,
            feature_label_gcn_dropout=0.1,
            feature_label_gcn_learnable_adj=True,
            feature_label_gcn_init_gate=-2.0,
            feature_label_gcn_adj_init_off_diag=0.1,
            use_view_transformer_fusion=True,
            view_transformer_layers=1,
            view_transformer_heads=4,
            view_transformer_dropout=0.1,
            view_transformer_residual_scale=0.1,
            use_cross_modal_fusion=True,
            cross_modal_heads=4,
            cross_modal_dropout=0.1,
            cross_modal_tokens=32,
            use_lead_attention=True,
            lead_attention_reduction=4,
            lead_attention_spectral_spatial=True,
            lead_attention_spatial_kernel=7,
    ):
        super(MyNet7ViewTimeFreq, self).__init__()
        self.num_classes = num_classes
        self.use_view_transformer_fusion = use_view_transformer_fusion
        self.view_transformer_residual_scale = view_transformer_residual_scale
        self.use_cross_modal_fusion = use_cross_modal_fusion
        
        # ========== 原有6个视图分支 ==========
        # 视图1-6的MyNet分支（保持原有设计，single_view=True输出特征）
        self.MyNet1 = MyNet(input_channels=1, single_view=True, num_classes=num_classes)   # 1导联
        self.MyNet2 = MyNet(input_channels=2, single_view=True, num_classes=num_classes)   # 2导联
        self.MyNet3 = MyNet(input_channels=2, single_view=True, num_classes=num_classes)   # 2导联
        self.MyNet4 = MyNet(input_channels=2, single_view=True, num_classes=num_classes)   # 2导联
        self.MyNet5 = MyNet(input_channels=2, single_view=True, num_classes=num_classes)   # 2导联
        self.MyNet6 = MyNet(input_channels=3, single_view=True, num_classes=num_classes)   # 3导联
        
        # ========== 新增：第7个视图 - 时频融合 ==========
        self.MyNet7_TimeFreq = TimeFreqView(
            input_channels=12, output_dim=128,
            use_lead_attention=use_lead_attention,
            lead_attention_reduction=lead_attention_reduction,
            lead_attention_spectral_spatial=lead_attention_spectral_spatial,
            lead_attention_spatial_kernel=lead_attention_spatial_kernel,
        )
        
        # ========== 新增：第8个视图 - Sobel形态梯度 ==========
        self.MyNet8_Sobel = SobelMorphologicalView(
            input_channels=12, output_dim=128, num_classes=num_classes
        )
        # 【关键】Sobel视图渐进式门控：初始化为极小贡献
        # sigmoid(-3) ≈ 0.047，意味着训练初期Sobel视图仅贡献~5%的融合权重，
        # 随着骨干网络学到有用的梯度特征，门控自动打开，
        # 避免训练初期噪声特征污染其他7个已知视图的融合。
        self.sobel_gate = nn.Parameter(torch.tensor(-3.0))
        
        # ========== 8个视图的自适应权重模块 ==========
        self.fuse_weight_1 = AdaptiveWeight(128)
        self.fuse_weight_2 = AdaptiveWeight(128)
        self.fuse_weight_3 = AdaptiveWeight(128)
        self.fuse_weight_4 = AdaptiveWeight(128)
        self.fuse_weight_5 = AdaptiveWeight(128)
        self.fuse_weight_6 = AdaptiveWeight(128)
        self.fuse_weight_7 = AdaptiveWeight(128)  # 时频视图的权重
        self.fuse_weight_8 = AdaptiveWeight(128)  # 新增：Sobel形态梯度视图的权重

        # ========== 新增：视图级 Transformer 融合（8个视图token） ==========
        self.view_pos_embed = None
        self.view_transformer = None
        self.view_transformer_norm = None
        if self.use_view_transformer_fusion:
            self.view_pos_embed = nn.Parameter(torch.zeros(1, 8, 128))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=128,
                nhead=view_transformer_heads,
                dim_feedforward=256,
                dropout=view_transformer_dropout,
                batch_first=True,
                activation='gelu',
                norm_first=True,
            )
            self.view_transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=view_transformer_layers,
            )
            self.view_transformer_norm = nn.LayerNorm(128)

        # ========== 跨模态中融合 (Cross-Modal Mid-Fusion) ==========
        # 在网络中层引入时域-频域双向交叉注意力
        self.cross_modal_attn = None
        self.freq_seq_encoder = None
        self.time_seq_downsample = None
        self.cross_modal_gate = None
        if self.use_cross_modal_fusion:
            # 双向交叉注意力模块
            self.cross_modal_attn = CrossModalAttention(
                d_model=128,
                nhead=cross_modal_heads,
                dropout=cross_modal_dropout,
            )
            # 频域序列编码器：从 STFT 谱图中提取频域 token 序列
            self.freq_seq_encoder = nn.Sequential(
                nn.Conv2d(12, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            # 时域序列降采样：每个视图的序列压缩到固定 token 数，控制 cross-attention 计算开销
            self.time_seq_downsample = nn.AdaptiveAvgPool1d(cross_modal_tokens)
            # 可学习的融合门控（初始化为较小值，训练初期让原始特征主导）
            self.cross_modal_gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2)≈0.12

        # ========== Feature-level 标签图 GCN（新增，作用于 fused_feat 上） ==========
        self.feature_label_gcn = None
        if use_feature_label_gcn:
            self.feature_label_gcn = FeatureLabelGCN(
                feat_dim=128,
                num_classes=num_classes,
                gcn_hidden=feature_label_gcn_hidden,
                num_gcn_layers=feature_label_gcn_layers,
                dropout=feature_label_gcn_dropout,
                learnable_adj=feature_label_gcn_learnable_adj,
                init_gate=feature_label_gcn_init_gate,
                adj_init_off_diag=feature_label_gcn_adj_init_off_diag,
            )

        # 最终分类头
        self.fc = nn.Linear(128, num_classes)

        # Logits-level 标签图细化模块（可选，原有机制保留）
        self.label_refiner = None
        if use_label_graph_refiner:
            self.label_refiner = LabelGraphRefiner(
                num_classes=num_classes,
                hidden=label_graph_hidden,
                dropout=label_graph_dropout,
                learnable_adj=label_graph_learnable_adj,
            )
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_intermediate=False):
        """前向传播：8视图特征提取→自适应加权融合→分类
        
        参数：
            x: 输入12导联ECG信号，形状 (batch_size, 12, seq_len)
            return_intermediate: 是否返回中间特征（用于可视化）
        
        返回：
            logits: 分类输出，形状 (batch_size, num_classes)
            intermediate: （可选）中间特征字典
        """
        
        # ========== 准备各视图的输入 ==========
        x_view1 = x[:, 3, :].unsqueeze(1)                                               # 导联3 (I)
        x_view2 = torch.cat((x[:, 0, :].unsqueeze(1), x[:, 4, :].unsqueeze(1)), dim=1)  # 导联0,4 (aVR, aVL)
        x_view3 = x[:, 6:8, :]                                                           # 导联6-7 (V1, V2)
        x_view4 = x[:, 8:10, :]                                                          # 导联8-9 (V3, V4)
        x_view5 = x[:, 10:12, :]                                                         # 导联10-11 (V5, V6)
        x_view6 = torch.cat((x[:, 1:3, :], x[:, 5, :].unsqueeze(1)), dim=1)             # 导联1-2,5 (II, III, aVF)

        # ========== 第一步：特征提取 + 跨模态中融合 (Mid-Fusion) ==========
        if self.use_cross_modal_fusion and self.cross_modal_attn is not None:
            # --- 时域视图：同时获取池化后特征和池化前序列特征 ---
            view1_feat, time_seq_1 = self.MyNet1(x_view1, return_seq=True)
            view2_feat, time_seq_2 = self.MyNet2(x_view2, return_seq=True)
            view3_feat, time_seq_3 = self.MyNet3(x_view3, return_seq=True)
            view4_feat, time_seq_4 = self.MyNet4(x_view4, return_seq=True)
            view5_feat, time_seq_5 = self.MyNet5(x_view5, return_seq=True)
            view6_feat, time_seq_6 = self.MyNet6(x_view6, return_seq=True)

            # --- Sobel形态梯度视图：全导联一阶微分特征 ---
            view8_feat, time_seq_8 = self.MyNet8_Sobel(x, return_seq=True)

            # --- 频域视图：计算 STFT 一次并复用（避免重复计算） ---
            fused_spec, _ = self.MyNet7_TimeFreq.stft(x)
            spec_feat, _, _ = self.MyNet7_TimeFreq.spectrogram_cnn(fused_spec)
            view7_feat = self.MyNet7_TimeFreq.feature_projection(spec_feat)
            view7_feat = self.MyNet7_TimeFreq.output_norm(view7_feat)

            # --- 跨模态交叉注意力 (Cross-Modal Mid-Fusion) ---
            # 时域: 降采样各视图序列（含Sobel视图），拼接为时域 token 序列
            time_seqs = [time_seq_1, time_seq_2, time_seq_3,
                         time_seq_4, time_seq_5, time_seq_6, time_seq_8]
            time_seqs_ds = [self.time_seq_downsample(seq) for seq in time_seqs]
            T_ds = time_seqs_ds[0].shape[2]  # 每个视图降采样后的 token 数
            time_concat = torch.cat(time_seqs_ds, dim=2)         # (B, 128, 7*T_ds)
            time_tokens = time_concat.permute(0, 2, 1)           # (B, 7*T_ds, 128)

            # 频域: 从缓存的 STFT 谱图编码频域 token 序列
            freq_map = self.freq_seq_encoder(fused_spec)          # (B, 128, F', T')
            freq_tokens = freq_map.flatten(2).permute(0, 2, 1)   # (B, S, 128)

            # 双向交叉注意力：
            #   频域Q → 时域K/V: 节律异常 → 寻找对应异常波形
            #   时域Q → 频域K/V: 波形形态 → 寻找对应频率成分
            time_enhanced, freq_enhanced = self.cross_modal_attn(time_tokens, freq_tokens)

            # 将增强后的 token 池化为修正向量，通过可学习门控注入原始特征
            gate = torch.sigmoid(self.cross_modal_gate)

            # 时域修正: 拆回7个时域视图（6个导联视图+1个Sobel视图），分别池化为修正向量
            time_enh_split = time_enhanced.permute(0, 2, 1)       # (B, 128, 7*T_ds)
            time_view_chunks = torch.split(time_enh_split, T_ds, dim=2)
            view1_feat = view1_feat + gate * time_view_chunks[0].mean(dim=2)
            view2_feat = view2_feat + gate * time_view_chunks[1].mean(dim=2)
            view3_feat = view3_feat + gate * time_view_chunks[2].mean(dim=2)
            view4_feat = view4_feat + gate * time_view_chunks[3].mean(dim=2)
            view5_feat = view5_feat + gate * time_view_chunks[4].mean(dim=2)
            view6_feat = view6_feat + gate * time_view_chunks[5].mean(dim=2)
            view8_feat = view8_feat + gate * time_view_chunks[6].mean(dim=2)  # Sobel时域修正

            # 频域修正
            view7_feat = view7_feat + gate * freq_enhanced.permute(0, 2, 1).mean(dim=2)
        else:
            # --- 原始路径（无跨模态中融合，纯后融合） ---
            view1_feat = self.MyNet1(x_view1)
            view2_feat = self.MyNet2(x_view2)
            view3_feat = self.MyNet3(x_view3)
            view4_feat = self.MyNet4(x_view4)
            view5_feat = self.MyNet5(x_view5)
            view6_feat = self.MyNet6(x_view6)
            view7_feat = self.MyNet7_TimeFreq(x)
            view8_feat = self.MyNet8_Sobel(x)  # Sobel形态梯度视图

        # ========== 第二步：计算8个视图的自适应权重 ==========
        view_features = [view1_feat, view2_feat, view3_feat, view4_feat,
                         view5_feat, view6_feat, view7_feat, view8_feat]

        fuse_weight_1 = self.fuse_weight_1(view1_feat)  # (batch_size, 1)
        fuse_weight_2 = self.fuse_weight_2(view2_feat)
        fuse_weight_3 = self.fuse_weight_3(view3_feat)
        fuse_weight_4 = self.fuse_weight_4(view4_feat)
        fuse_weight_5 = self.fuse_weight_5(view5_feat)
        fuse_weight_6 = self.fuse_weight_6(view6_feat)
        fuse_weight_7 = self.fuse_weight_7(view7_feat)  # 时频视图权重
        fuse_weight_8 = self.fuse_weight_8(view8_feat)  # Sobel形态梯度视图权重
        
        fuse_weights = [
            fuse_weight_1, fuse_weight_2, fuse_weight_3, fuse_weight_4,
            fuse_weight_5, fuse_weight_6, fuse_weight_7, fuse_weight_8
        ]

        # ========== 第三步：融合8个视图 ==========
        # Sobel渐进式门控：控制第8视图对融合的贡献比例
        sobel_gate = torch.sigmoid(self.sobel_gate)  # 初始≈ 0.047, 渐进增大
        
        # 先计算原始加权融合（作为主干融合特征）
        fused_sum = (fuse_weight_1 * view1_feat +
                     fuse_weight_2 * view2_feat +
                     fuse_weight_3 * view3_feat +
                     fuse_weight_4 * view4_feat +
                     fuse_weight_5 * view5_feat +
                     fuse_weight_6 * view6_feat +
                     fuse_weight_7 * view7_feat +
                     sobel_gate * fuse_weight_8 * view8_feat)  # (B, 128)  门控叠加在Sobel视图上

        if self.use_view_transformer_fusion and self.view_transformer is not None:
            # (B, 8, 128) 视图token，Sobel视图经门控缩放
            view_features_gated = [
                view1_feat, view2_feat, view3_feat, view4_feat,
                view5_feat, view6_feat, view7_feat,
                sobel_gate * view8_feat  # 门控也作用于Transformer token
            ]
            view_tokens = torch.stack(view_features_gated, dim=1)

            # 使用softmax权重做"温和缩放"，避免token幅值差异过大导致不稳定
            weights_raw = torch.cat(fuse_weights, dim=1)  # (B, 8)
            weights = F.softmax(weights_raw, dim=1)  # (B, 8)
            token_scale = 0.5 + 0.5 * weights  # (B, 8) in [0.5, 1.0]
            view_tokens = view_tokens * token_scale.unsqueeze(-1)

            # 视图位置编码
            if self.view_pos_embed is not None:
                view_tokens = view_tokens + self.view_pos_embed

            # Transformer 编码得到修正量（delta），再用残差方式叠加回 fused_sum
            view_tokens = self.view_transformer(view_tokens)
            delta = view_tokens.mean(dim=1)  # (B, 128)
            delta = self.view_transformer_norm(delta)
            fused_feat = fused_sum + self.view_transformer_residual_scale * delta
        else:
            fused_feat = fused_sum
        
        # ========== 第四步：Feature-level 标签图建模 + 最终分类 ==========
        # 先在特征空间做标签共现图卷积（如果启用）
        if self.feature_label_gcn is not None:
            fused_feat = self.feature_label_gcn(fused_feat)  # (B, 128) → 标签图增强 → (B, 128)

        logits = self.fc(fused_feat)  # (batch_size, num_classes)

        # Logits-level 后处理（如果同时启用）
        if self.label_refiner is not None:
            logits = self.label_refiner(logits)
        
        # ========== 返回结果 ==========
        if return_intermediate:
            intermediate = {
                'view_features': view_features,
                'fuse_weights': fuse_weights,
                'fused_feat': fused_feat,
                'view_names': ['view1_I', 'view2_aVR_aVL', 'view3_V1_V2', 
                              'view4_V3_V4', 'view5_V5_V6', 'view6_II_III_aVF', 
                              'view7_TimeFreq', 'view8_SobelMorphology'],
                'cross_modal_gate': torch.sigmoid(self.cross_modal_gate).item() if self.use_cross_modal_fusion and self.cross_modal_gate is not None else None,
                'sobel_gate': sobel_gate.item(),
            }
            return logits, intermediate
        
        return logits
    
    def get_view_weights(self, x):
        """获取8个视图的融合权重分布（用于可视化）
        
        返回：
            weights_np: numpy数组，形状 (batch_size, 8)，每行sum=1
        """
        _, intermediate = self.forward(x, return_intermediate=True)
        
        # 提取权重并进行softmax归一化
        weights = torch.cat(intermediate['fuse_weights'], dim=1)  # (B, 8)
        weights = F.softmax(weights, dim=1)
        
        return weights.detach().cpu().numpy()


# ========== 便捷函数 ==========

def create_mynet7view_timefreq(num_classes=9, pretrained=False):
    """创建MyNet7ViewTimeFreq模型
    
    参数：
        num_classes: 分类类别数
        pretrained: 是否加载预训练权重
    
    返回：
        model: MyNet7ViewTimeFreq实例
    """
    model = MyNet7ViewTimeFreq(num_classes=num_classes)
    
    if pretrained:
        # TODO: 从checkpoints加载预训练权重
        pass
    
    return model
