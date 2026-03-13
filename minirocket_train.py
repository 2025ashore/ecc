# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb
# MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series
# Classification
# 论文链接: https://arxiv.org/abs/2012.08791
# 代码功能: 基于MiniRocket时间序列特征变换的ECG信号多标签分类
# 核心逻辑: MiniRocket通过预定义的随机卷积核生成强判别特征，再用简单线性模型分类，兼顾速度与性能

import utils  # 自定义工具函数库（包含TPR计算等评估指标）
from sklearn.metrics import roc_auc_score  # 多标签分类评估指标：ROC曲线下面积
import copy  # 用于深拷贝最优模型（避免引用传递导致的参数修改）
import numpy as np  # 数组运算与数据处理
import torch
import torch.nn as nn  # PyTorch神经网络模块
import torch.optim as optim  # PyTorch优化器模块
from models.minrocket import fit, transform  # MiniRocket核心函数：fit初始化参数，transform生成特征
from dataset import DownLoadECGData  # 数据集加载与预处理类（之前定义的数据集处理模块）
import random  # 随机数生成（用于固定种子）
from dataset import hf_dataset  # Hugging Face数据集加载函数（适配第三方数据集）


def setup_seed(seed):
    """
    固定随机种子，保证实验的可重复性

    参数：
        seed: int - 随机种子值（此处硬编码为7，也可通过配置文件传入）
    """
    print('seed: ', seed)
    torch.manual_seed(seed)  # 设置PyTorch CPU随机种子
    torch.cuda.manual_seed_all(seed)  # 设置PyTorch所有GPU随机种子（多GPU场景）
    np.random.seed(seed)  # 设置NumPy随机种子
    random.seed(seed)  # 设置Python原生随机种子
    torch.backends.cudnn.deterministic = True  # 禁用CUDA非确定性算法，确保结果一致


def train(num_classes, training_size, X_training, Y_training, X_validation, Y_validation, **kwargs):
    """
    训练流程：初始化MiniRocket特征变换参数 + 训练线性分类器

    参数：
        num_classes: int - 类别数量（从训练集标签维度获取）
        training_size: int - 训练集样本数量
        X_training: np.ndarray - 训练集ECG信号数据，形状[样本数, 时间步长, 导联数]
        Y_training: np.ndarray - 训练集标签（独热编码），形状[样本数, 类别数]
        X_validation: np.ndarray - 验证集ECG信号数据
        Y_validation: np.ndarray - 验证集标签（独热编码）
        **kwargs: dict - 可选超参数（用于覆盖默认配置）

    返回：
        parameters: dict - MiniRocket特征变换的参数（卷积核、偏置、 dilation等）
        best_model: nn.Sequential - 训练过程中验证损失最优的线性模型
        f_mean: np.ndarray - 训练集变换后特征的均值（用于归一化）
        f_std: np.ndarray - 训练集变换后特征的标准差（用于归一化）
    """
    # -- 初始化超参数 ----------------------------------------------------------
    # 默认超参数（适用于大多数时间序列分类任务，可通过kwargs覆盖）
    args = {
        "num_features": 10_000,  # MiniRocket生成的特征数量（10000个）
        "minibatch_size": 256,    # 训练批次大小
        "lr": 1e-4,               # 学习率（Adam优化器）
        "max_epochs": 50,         # 最大训练轮数
        "patience_lr": 5,         # 学习率调度器的耐心值（5个小批次无改善则降学习率）
        "patience": 10,           # 早停的耐心值（10个小批次无改善则停止训练）
        "cache_size": training_size  # 缓存大小（设为训练集大小以加速特征变换，0则不缓存）
    }
    # 用传入的kwargs更新超参数（优先级：用户传入 > 默认值）
    args = {**args, **kwargs}

    # MiniRocket的特征数必须是84的倍数（84是预定义的卷积核组合数量）
    _num_features = 84 * (args["num_features"] // 84)

    # -- 定义模型初始化函数 ----------------------------------------------------
    def init(layer):
        """初始化线性层的权重和偏置（均设为0）"""
        if isinstance(layer, nn.Linear):  # 仅对线性层生效
            nn.init.constant_(layer.weight.data, 0)  # 权重初始化为0
            nn.init.constant_(layer.bias.data, 0)    # 偏置初始化为0

    # -- 构建模型、损失函数、优化器 ----------------------------------------------
    # 模型：单层线性分类器（MiniRocket已提取强判别特征，无需复杂网络）
    model = nn.Sequential(nn.Linear(_num_features, num_classes))
    # 损失函数：BCEWithLogitsLoss（适用于多标签分类，直接接收logits输出，无需sigmoid）
    loss_function = nn.BCEWithLogitsLoss()
    # 优化器：Adam（适用于小批量训练，收敛稳定）
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    # 学习率调度器：根据验证损失降低学习率（损失无改善则降为原来的0.5，最小1e-8）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, min_lr=1e-8, patience=args["patience_lr"]
    )
    # 初始化模型参数
    model.apply(init)

    # -- 数据类型转换 ----------------------------------------------------------
    # 转换为float32以适配PyTorch和MiniRocket的计算精度要求
    X_training = X_training.astype(np.float32)
    Y_training = torch.FloatTensor(Y_training)  # 标签转为PyTorch张量
    X_validation = X_validation.astype(np.float32)
    Y_validation = torch.FloatTensor(Y_validation)

    # -- 训练循环 --------------------------------------------------------------
    minibatch_count = 0  # 小批次计数器（用于触发验证和早停）
    best_validation_loss = np.inf  # 最优验证损失（初始设为无穷大）
    stall_count = 0  # 验证损失无改善的计数器（用于早停）
    stop = False  # 早停标志位

    print("Training... (faster once caching is finished)")  # 缓存完成后训练会加速

    for epoch in range(args["max_epochs"]):  # 遍历每个训练轮数
        print(f"Epoch {epoch + 1}...".ljust(80, " "), end="\r", flush=True)  # 实时打印当前轮数

        # 第1个epoch：初始化MiniRocket参数并预处理验证集特征
        if epoch == 0:
            # fit函数：根据训练集数据初始化MiniRocket的卷积核、偏置、dilation等参数
            parameters = fit(X_training, args["num_features"])
            # 用初始化后的参数转换验证集特征（生成_num_features个特征）
            X_validation_transform = transform(X_validation, parameters)

        # 转换训练集特征（每个epoch都重新转换，确保数据增强生效）
        X_training_transform = transform(X_training, parameters)

        # 第1个epoch：计算训练集特征的均值和标准差（用于归一化，避免数据泄露）
        if epoch == 0:
            f_mean = X_training_transform.mean(0)  # 按特征维度求均值（形状：[num_features]）
            f_std = X_training_transform.std(0) + 1e-8  # 按特征维度求标准差（+1e-8避免除零）

            # 验证集特征归一化（用训练集的均值和标准差）
            X_validation_transform = (X_validation_transform - f_mean) / f_std
            X_validation_transform = torch.FloatTensor(X_validation_transform)  # 转为张量

        # 训练集特征归一化
        X_training_transform = (X_training_transform - f_mean) / f_std
        X_training_transform = torch.FloatTensor(X_training_transform)

        # 随机打乱训练集索引，按批次划分（实现批次随机化）
        minibatches = torch.randperm(len(X_training_transform)).split(args["minibatch_size"])

        # 遍历每个小批次进行训练
        for minibatch_index, minibatch in enumerate(minibatches):
            # 若已触发早停，停止当前epoch的训练
            if epoch > 0 and stop:
                break
            # 跳过最后一个不足批次大小的小批次（避免批次大小不一致导致的问题）
            if minibatch_index > 0 and len(minibatch) < args["minibatch_size"]:
                break

            # -- 模型训练步骤 --------------------------------------------------
            optimizer.zero_grad()  # 梯度清零（避免上一批次梯度累积）
            # 前向传播：线性模型接收变换后的特征，输出logits
            _Y_training = model(X_training_transform[minibatch])
            # 计算训练损失（基于模型输出和真实标签）
            training_loss = loss_function(_Y_training, Y_training[minibatch])
            training_loss.backward()  # 反向传播：计算梯度
            optimizer.step()  # 优化器更新：调整模型参数

            minibatch_count += 1  # 累积小批次计数器

            # 每10个小批次：验证模型性能、调整学习率、判断早停
            if minibatch_count % 10 == 0:
                # 验证集前向传播（无梯度计算，节省资源）
                _Y_validation = model(X_validation_transform)
                validation_loss = loss_function(_Y_validation, Y_validation)

                # 学习率调度器：根据验证损失调整学习率
                scheduler.step(validation_loss)

                # 早停逻辑：判断验证损失是否改善
                if validation_loss.item() >= best_validation_loss:
                    stall_count += 1  # 无改善，累积计数器
                    # 若累积次数达到patience，触发早停
                    if stall_count >= args["patience"]:
                        stop = True
                        print(f"\n<Stopped at Epoch {epoch + 1}>")  # 打印早停信息
                else:
                    # 验证损失改善，更新最优损失和最优模型
                    best_validation_loss = validation_loss.item()
                    best_model = copy.deepcopy(model)  # 深拷贝最优模型（避免后续修改）
                    stall_count = 0  # 重置无改善计数器

    return parameters, best_model, f_mean, f_std


def predict(parameters, model, f_mean, f_std, X_test, Y_test, **kwargs):
    """
    预测函数：用训练好的MiniRocket参数和模型预测测试集，计算评估指标

    参数：
        parameters: dict - MiniRocket特征变换参数（train函数返回）
        model: nn.Sequential - 最优线性模型（train函数返回）
        f_mean: np.ndarray - 训练集特征均值（用于归一化）
        f_std: np.ndarray - 训练集特征标准差（用于归一化）
        X_test: np.ndarray - 测试集ECG信号数据
        Y_test: np.ndarray - 测试集标签（独热编码）
        **kwargs: dict - 可选参数（预留扩展）
    """
    predictions = []  # 存储预测结果

    # 数据类型转换（适配MiniRocket和PyTorch）
    X_test = X_test.astype(np.float32)

    # -- 特征变换与归一化 ------------------------------------------------------
    # 用训练好的MiniRocket参数转换测试集特征
    X_test_transform = transform(X_test, parameters)
    # 用训练集的均值和标准差归一化测试集特征（避免数据泄露）
    X_test_transform = (X_test_transform - f_mean) / f_std
    # 转为PyTorch张量
    X_test_transform = torch.FloatTensor(X_test_transform)

    # -- 模型预测 --------------------------------------------------------------
    model.eval()  # 设置模型为评估模式（无Dropout等训练特定操作）
    with torch.no_grad():  # 禁用梯度计算，节省资源
        # 前向传播+Sigmoid激活（将logits转为0-1的概率，适配多标签分类）
        _predictions = torch.sigmoid(model(X_test_transform)).cpu().detach().numpy()
    predictions.append(_predictions)
    # 去除多余维度，得到最终预测结果（形状：[测试集样本数, 类别数]）
    predictions = np.array(predictions).squeeze(axis=0)

    # -- 计算评估指标 ----------------------------------------------------------
    # 计算macro-AUC（多标签分类，每个类别独立计算AUC后取平均）
    auc = roc_auc_score(Y_test, predictions, average='macro')
    # 计算TPR（真阳性率，自定义工具函数，适配多标签场景）
    TPR = utils.compute_TPR(Y_test, predictions)
    # 打印评估结果
    print("Test Set Evaluation:")
    print(f"AUC = {auc:.4f}, TPR = {TPR:.4f}")


def main(data_name='ptbxl'):
    """
    主函数：整合数据集加载、模型训练、预测评估的完整流程

    参数：
        data_name: str - 数据集名称（'ptbxl'/'cpsc'，默认'ptbxl'）
    """
    # 固定随机种子（保证可重复性）
    setup_seed(7)

    # -- 加载并预处理数据集 ----------------------------------------------------
    if data_name == 'ptbxl':
        # 加载PTB-XL数据集（任务：节律分类'rhythm'，实验名称'exp0'）
        ded = DownLoadECGData('exp0', 'rhythm', 'data/ptbxl/')
        # 预处理数据：加载原始数据、标签聚合、划分训练/验证/测试集、信号预处理
        X_training, Y_training, X_validation, Y_validation, X_test, Y_test = ded.preprocess_data()
    elif data_name == 'cpsc':
        # 加载CPSC数据集（任务：所有标签'all'，实验名称'exp_cpsc'）
        ded = DownLoadECGData('exp_cpsc', 'all', 'data/CPSC/')
        X_training, Y_training, X_validation, Y_validation, X_test, Y_test = ded.preprocess_data()
    else:
        # 加载Hugging Face数据集（适配第三方公开数据集）
        X_training, Y_training, X_validation, Y_validation, X_test, Y_test = hf_dataset()

    # -- 训练模型 --------------------------------------------------------------
    # 类别数 = 训练集标签的维度（Y_training.shape[1]）
    # 训练集大小 = 训练集样本数（len(X_training)）
    parameters, best_model, f_mean, f_std = train(
        num_classes=len(Y_training[0]),
        training_size=len(X_training),
        X_training=X_training,
        Y_training=Y_training,
        X_validation=X_validation,
        Y_validation=Y_validation
    )

    # -- 测试集预测与评估 ------------------------------------------------------
    predict(parameters, best_model, f_mean, f_std, X_test, Y_test)


# 主函数入口：默认使用PTB-XL数据集，可修改参数为'cpsc'切换数据集
if __name__ == '__main__':
    main(data_name='ptbxl')  # 可选：main(data_name='cpsc')