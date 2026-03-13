# -*- coding: utf-8 -*-
'''
@time: 2021/4/16 18:45
@ author:
'''

class Config:
    """
    ECG分类实验配置类：集中管理所有超参数、路径和实验设置
    用于统一控制训练流程、模型选择、数据读取等，方便实验复现和参数调优
    """

    # 随机种子：固定所有随机性（模型初始化、数据划分等），保证实验可重复
    seed = 10

    # 数据集路径配置：支持3种常见ECG数据集切换
    datafolder = 'data/ptbxl/'  # PTB-XL：大规模公开ECG数据集（含多种心律失常标注）
    # datafolder = '../data/CPSC/'  # CPSC：中国心电学会数据集（适用于心律失常检测）
    # datafolder = '../data/hf/'    # HF：心力衰竭相关ECG数据集（聚焦心功能不全诊断）

    # 实验编号：区分不同实验方案（如基线实验、改进实验、消融实验等）
    '''
    实验标识说明：
    - exp0：基线实验（基础模型+标准流程）
    - exp1~exp3：改进实验（如多实例学习、多模态融合、特征增强等不同方案）
    - exp1.1、exp1.1.1：子实验（针对某一改进点的消融或参数优化）
    '''
    experiment = 'exp0'

    # 训练相关配置
    '''
    主模型选择：支持多种适用于ECG时序数据的深度学习模型
    模型说明：
    - MyNet6View：自定义多视角ECG模型（可能融合不同导联或特征维度）
    - MyNet7ViewTimeFreq：MyNet6View + 时频融合第7视图（✨ 推荐用于多标签分类）
    - resnet1d_wang：Wang等人提出的1D ResNet（适配ECG时序特征提取）
    - xresnet1d101：扩展版1D ResNet（更深层结构，提升特征表达能力）
    - inceptiontime：Inception-Time模型（多尺度卷积融合，捕捉ECG不同时长特征）
    - fcn_wang：Wang等人提出的1D FCN（轻量型模型，快速训练）
    - lstm：单向LSTM（捕捉ECG时序依赖关系）
    - lstm_bidir：双向LSTM（同时捕捉前后时序信息）
    - vit：Vision Transformer（将ECG序列转为patch，学习全局依赖）
    - mobilenetv3_small：轻量型MobileNetV3（适用于边缘设备部署）
    '''
    model_name = 'MyNet7ViewTimeFreq'  # 主训练模型（已切换到7视图时频融合版本）

    model_name2 = 'MyNet'  # 辅助模型（用于知识蒸馏、多模型融合等场景）

    batch_size = 128  # 批次大小：平衡训练速度与内存占用（ECG数据常用32/64/128）
    max_epoch = 30  # 最大训练轮数：防止过拟合，可结合早停机制调整
    lr = 0.001  # 初始学习率：ECG任务常用1e-3~1e-4，可搭配学习率调度器
    device_num = 1  # GPU设备编号：单GPU用0/1，多GPU可设为列表（如[0,1]）

    # 模型权重文件：命名包含模型名、实验编号、最优评价指标（AUC）
    # 用于加载预训练模型、继续训练或测试推理
    checkpoints = 'MyNet7ViewTimeFreq_exp0_checkpoint_best.pth'

    # 输出目录：Checkpoint和训练曲线CSV的保存根目录
    # 消融实验脚本会自动修改此路径以将各实验结果分别存放
    output_dir = '.'

    # 知识蒸馏相关参数（仅在使用蒸馏时生效）
    alpha = 0.5  # 损失平衡系数：alpha*蒸馏损失 + (1-alpha)*原始分类损失（0~1之间）
    temperature = 2  # 温度参数：软化教师模型输出分布，温度越高分布越平滑（通常1~10）


    # ========== 多尺度时频融合配置 (Scheme #7) ==========
    # 频域增强相关参数
    use_timefreq_fusion = False  # 是否使用时频融合模型（True切换到TimeFreqFusionNet）
    
    # STFT参数
    stft_n_fft = 512  # FFT大小（越大频率分辨率越高，但时间分辨率下降）
    stft_hop_length = 128  # 跳跃长度（越小时间分辨率越高）
    stft_n_scales = 3  # 多尺度数量（捕捉不同时间窗的特征）
    
    # 时频融合权重
    use_label_conditional_fusion = True  # 是否使用标签条件融合
    fusion_temperature = 1.0  # 标签条件的温度参数（影响时频权重的锐度）
    
    # 鲁棒性增强
    apply_time_freq_augmentation = True  # 是否应用时频增强（时间/频率裁剪、混合）
    augmentation_prob = 0.5  # 增强概率
    
    # 频域预处理
    normalize_spectrogram = True  # 是否对谱图进行归一化
    spectrogram_norm_type = 'instance'  # 归一化类型：'instance'/'batch'/'layer'

    # 标签依赖建模（GCN/图细化）
    use_label_graph_refiner = False  # 是否启用 logits 层标签图细化（已修复门控bug，可安全启用）
    label_graph_hidden = 64  # 图层隐藏维度
    label_graph_learnable_adj = True  # 邻接矩阵是否可学习
    label_graph_dropout = 0.1  # 图层dropout

    # Feature-level 标签图 GCN（新增：作用于 fused_feat，在分类头之前建模标签共现）
    use_feature_label_gcn = True           # 是否启用 feature 层标签图 GCN
    feature_label_gcn_hidden = 64          # 每标签隐状态维度（建议 32/64/128）
    feature_label_gcn_layers = 2           # GCN 层数（建议 1/2，3层有过平滑风险）
    feature_label_gcn_dropout = 0.1        # GCN 内部 dropout（建议 0.0~0.3）
    feature_label_gcn_learnable_adj = True  # 邻接矩阵是否可学习
    feature_label_gcn_init_gate = -2.0     # 门控初始值（sigmoid前）：控制GCN增强的初始强度
                                           #   -4.0 → sigmoid≈0.02（极保守，需更多epoch激活）
                                           #   -2.0 → sigmoid≈0.12（温和起步，推荐默认值）
                                           #   -1.0 → sigmoid≈0.27（较积极）
                                           #    0.0 → sigmoid=0.50（一半原始+一半GCN）
    feature_label_gcn_adj_init_off_diag = 0.1  # 邻接矩阵 off-diagonal 初始系数
                                                #   0.01 → 稀疏起步（标签独立性强时适用）
                                                #   0.1  → 默认均匀先验
                                                #   0.3  → 密集起步（标签共现频繁时适用）

    # ========== 导联注意力 (Lead Attention) ==========
    # 在频域分支 SpectrogramCNN 的多尺度卷积之前，对 12 导联频谱图施加通道注意力
    # 让网络自适应地放大含关键疾病频谱的导联，抑制无关导联噪声
    use_lead_attention = True              # 是否启用导联注意力（False 可做消融实验）
    lead_attention_reduction = 4           # FC 瓶颈降维比（mid = 12/r），建议 2/4/6
    lead_attention_spectral_spatial = True  # 是否启用第二级频率-时间空间注意力
    lead_attention_spatial_kernel = 7      # 空间注意力卷积核大小（奇数），建议 3/5/7

    # ========== 视图级 Transformer 融合（方案1） ==========
    # 将 7 个视图特征 (B,7,128) 作为 token，经轻量 Transformer 编码后再池化分类
    use_view_transformer_fusion = True
    view_transformer_layers = 1
    view_transformer_heads = 4
    view_transformer_dropout = 0.1
    view_transformer_residual_scale = 0.2

    # ========== 跨模态中融合 (Cross-Modal Mid-Fusion) ==========
    # 在网络中层引入时域-频域双向交叉注意力，让两个模态在特征图级别交换信息
    # 频域Q→时域K/V: 节律异常→寻找对应异常波形; 时域Q→频域K/V: 波形特征→寻找对应频率成分
    use_cross_modal_fusion = True   # 是否启用跨模态中融合（True 启用, False 退回纯后融合）
    cross_modal_heads = 4           # 交叉注意力头数
    cross_modal_dropout = 0.1       # 交叉注意力 Dropout
    cross_modal_tokens = 32         # 每个时域视图降采样到的 token 数（控制计算/内存开销）

# 实例化配置对象：供其他模块（训练、测试、数据加载）导入使用
config = Config()