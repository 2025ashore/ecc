# -*- coding: utf-8 -*-
"""
@time: 2021/4/15 15:40
@author:
@description: 基于知识蒸馏（Knowledge Distillation, KD）的ECG信号分类训练代码
              核心逻辑：使用预训练的大模型（教师模型）指导小模型（学生模型）训练，
              兼顾模型性能与推理效率，支持多标签分类任务（指标包括AUC、TPR、损失值）
"""
import torch
import time
import os
# 导入自定义的模型库（包含各类ECG分类模型）、工具函数库（损失计算、指标计算等）
import models
import utils
from torch import optim  # PyTorch优化器模块
from dataset import load_datasets  # 导入数据集加载函数（对应之前的dataset.py）
from config import config  # 导入配置文件（包含训练参数、路径等）

# 导入评估指标：ROC曲线下面积（多标签分类常用指标）
from sklearn.metrics import roc_auc_score
import numpy as np  # 数组运算
import random  # 随机数生成
import pandas as pd  # 结果保存为CSV文件

# 设备选择：优先使用GPU（CUDA可用时），否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    """
    固定随机种子，保证实验的可重复性

    参数：
        seed: int - 随机种子值（从配置文件读取）
    """
    print('seed: ', seed)
    # 设置PyTorch的CPU随机种子
    torch.manual_seed(seed)
    # 设置PyTorch的所有GPU随机种子（多GPU场景）
    torch.cuda.manual_seed_all(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置Python原生的随机种子
    random.seed(seed)
    # 禁用CUDA的非确定性算法（确保相同输入得到相同输出）
    torch.backends.cudnn.deterministic = True


def save_distill_checkpoint(best_auc, model, optimizer, epoch):
    """
    保存知识蒸馏训练中的最佳学生模型参数

    参数：
        best_auc: float - 当前最佳的AUC指标值
        model: torch.nn.Module - 学生模型（待保存）
        optimizer: torch.optim.Optimizer - 优化器（保存其状态用于续训）
        epoch: int - 当前epoch数
    """
    print('Distilled Model Saving...')
    # 处理多GPU情况下的模型参数
    if config.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    # 确保保存目录存在
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    # 保存模型状态字典、优化器状态、当前epoch和最佳AUC
    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
    }, os.path.join('checkpoints', f"{config.model_name2}_{config.experiment}_distill_best.pth"))


def train_epoch(model_large, model_small, optimizer, criterion, train_dataloader):
    """
    训练一个epoch（单次完整遍历训练集）

    参数：
        model_large: torch.nn.Module - 教师模型（预训练大模型，不参与训练更新）
        model_small: torch.nn.Module - 学生模型（待训练小模型，优化目标）
        optimizer: torch.optim.Optimizer - 优化器（仅优化学生模型参数）
        criterion: 损失函数 - 知识蒸馏损失函数（融合学生模型与教师模型的预测）
        train_dataloader: DataLoader - 训练集数据加载器

    返回：
        train_avg_loss: float - 该epoch训练集平均损失
        train_auc: float - 该epoch训练集macro-AUC值
        train_TPR: float - 该epoch训练集真阳性率（针对多标签分类）
    """
    # 设置模型为训练模式：启用Dropout、BatchNorm更新等
    model_large.train()
    model_small.train()

    loss_meter = 0.0  # 累计损失值
    it_count = 0  # 迭代次数（批次数量）
    outputs = []  # 存储所有批次的模型预测结果（用于计算指标）
    targets = []  # 存储所有批次的真实标签（用于计算指标）

    # 遍历训练集的每个批次
    for inputs, target in train_dataloader:
        # 数据增强：给输入信号添加高斯噪声（均值0，标准差0.1），提升模型泛化能力
        inputs = inputs + torch.randn_like(inputs) * 0.1

        # 将数据和标签移到指定设备（GPU/CPU）
        inputs = inputs.to(device)
        target = target.to(device)

        # 梯度清零：避免上一轮迭代的梯度累积
        optimizer.zero_grad()

        # 教师模型推理（禁用梯度计算，仅用于生成蒸馏目标）
        with torch.no_grad():
            # 教师模型输出（作为学生模型的"软标签"）
            target1 = model_large(inputs)

        # 学生模型推理（启用梯度计算，用于反向传播）
        output = model_small(inputs)

        # 计算知识蒸馏损失：融合学生模型与真实标签的损失、学生与教师预测的损失
        loss = criterion(output, target, target1)

        # 反向传播：计算梯度
        loss.backward()
        # 优化器更新：更新学生模型参数
        optimizer.step()

        # 累计损失和迭代次数
        loss_meter += loss.item()  # 取出张量的数值（避免梯度跟踪）
        it_count += 1

        # 对学生模型输出做Sigmoid激活（多标签分类，每个类别独立预测概率）
        output = torch.sigmoid(output)
        # 收集当前批次的预测结果和真实标签（转移到CPU并转为numpy数组）
        for i in range(len(output)):
            outputs.append(output[i].cpu().detach().numpy())
            targets.append(target[i].cpu().detach().numpy())

    # 计算训练集的macro-AUC（多标签分类，对每个类别计算AUC后取平均）
    train_auc = roc_auc_score(targets, outputs, average='macro')
    # 计算训练集的TPR（真阳性率，自定义工具函数，适配多标签场景）
    train_TPR = utils.compute_TPR(targets, outputs)
    # 计算平均损失
    train_avg_loss = loss_meter / it_count

    # 打印该epoch的训练指标
    print('train_loss: %.4f,   macro_auc: %.4f,   TPR: %.4f' % (train_avg_loss, train_auc, train_TPR))
    return train_avg_loss, train_auc, train_TPR


def test_epoch(model_large, model_small, criterion, val_dataloader):
    """
    验证/测试一个epoch（单次完整遍历验证集/测试集）

    参数：
        model_large: torch.nn.Module - 教师模型（仅用于计算蒸馏损失）
        model_small: torch.nn.Module - 学生模型（核心评估对象）
        criterion: 损失函数 - 知识蒸馏损失函数（用于计算损失值，不参与优化）
        val_dataloader: DataLoader - 验证集/测试集数据加载器

    返回：
        test_avg_loss: float - 该epoch验证集/测试集平均损失
        test_auc: float - 该epoch验证集/测试集macro-AUC值
        test_TPR: float - 该epoch验证集/测试集真阳性率
    """
    # 设置模型为评估模式：禁用Dropout、固定BatchNorm参数等
    model_large.eval()
    model_small.eval()

    loss_meter = 0.0  # 累计损失值
    it_count = 0  # 迭代次数（批次数量）
    outputs = []  # 存储所有批次的模型预测结果
    targets = []  # 存储所有批次的真实标签

    # 禁用梯度计算（评估阶段不需要反向传播，节省显存和计算资源）
    with torch.no_grad():
        # 遍历验证集/测试集的每个批次
        for inputs, target in val_dataloader:
            # 数据增强：与训练阶段保持一致（添加高斯噪声）
            inputs = inputs + torch.randn_like(inputs) * 0.1

            # 将数据和标签移到指定设备
            inputs = inputs.to(device)
            target = target.to(device)

            # 教师模型推理（生成软标签）
            target1 = model_large(inputs)
            # 学生模型推理（生成预测结果）
            output = model_small(inputs)

            # 计算损失（仅用于监控，不更新参数）
            loss = criterion(output, target, target1)
            loss_meter += loss.item()
            it_count += 1

            # Sigmoid激活后收集预测结果和真实标签
            output = torch.sigmoid(output)
            for i in range(len(output)):
                outputs.append(output[i].cpu().detach().numpy())
                targets.append(target[i].cpu().detach().numpy())

        # 计算评估指标
        test_auc = roc_auc_score(targets, outputs, average='macro')
        test_TPR = utils.compute_TPR(targets, outputs)
        test_avg_loss = loss_meter / it_count

    # 打印评估指标
    print('test_loss: %.4f,   macro_auc: %.4f,   TPR: %.4f' % (test_avg_loss, test_auc, test_TPR))
    return test_avg_loss, test_auc, test_TPR


def train(config=config):
    """
    主训练流程：整合数据加载、模型初始化、训练迭代、指标监控、结果保存

    参数：
        config: 配置对象 - 包含所有训练参数（从config.py导入）
    """
    # 1. 固定随机种子（保证实验可复现）
    setup_seed(config.seed)
    # 打印GPU可用性（确认训练设备）
    print('torch.cuda.is_available:', torch.cuda.is_available())

    # 2. 加载数据集：训练集、验证集、测试集的DataLoader，以及类别数
    train_dataloader, val_dataloader, test_dataloader, num_classes = load_datasets(
        datafolder=config.datafolder,  # 数据集路径（如'data/ptbxl/'或'data/CPSC/'）
        experiment=config.experiment,  # 实验配置（仅PTB-XL数据集需要，指定任务类型）
    )

    # 3. 初始化模型（教师模型+学生模型）
    print('model_name:{}, num_classes={}'.format(config.model_name, num_classes))
    # 动态加载教师模型（从models模块根据配置文件的model_name获取）
    model_large = getattr(models, config.model_name)(num_classes=num_classes)
    # 动态加载学生模型（从models模块根据配置文件的model_name2获取）
    model_small = getattr(models, config.model_name2)(num_classes=num_classes)

    # 将模型移到指定设备（GPU/CPU）
    model_large = model_large.to(device)
    model_small = model_small.to(device)

    # 4. 初始化优化器和损失函数
    # 优化器：仅优化学生模型参数，使用Adam优化器
    optimizer = optim.Adam(model_small.parameters(), lr=config.lr)
    # 损失函数：知识蒸馏损失（KdLoss），参数alpha控制硬标签/软标签权重，temperature控制软标签平滑度
    criterion = utils.KdLoss(config.alpha, config.temperature)

    # 5. 加载教师模型预训练权重（如果配置文件指定了checkpoints路径）
    if config.checkpoints is not None:
        # 加载预训练权重文件（存储在'checkpoints'文件夹下）
        checkpoints = torch.load(os.path.join('checkpoints', config.checkpoints))
        # 获取教师模型的状态字典（参数名-参数值映射）
        model_dict = model_large.state_dict()
        # 过滤掉预训练权重中与教师模型不匹配的参数（避免维度错误）
        state_dict = {k: v for k, v in checkpoints['model_state_dict'].items() if k in model_dict.keys()}
        # 更新教师模型的参数
        model_dict.update(state_dict)
        model_large.load_state_dict(model_dict)
        # 打印预训练模型的最佳准确率（从权重文件中读取）
        print('best_auc of teacher model: ', checkpoints['best_auc'])

    # 初始化最佳测试集AUC（用于模型保存）
    best_test_auc = 0.0

    # =========> 开始迭代训练 <=========
    for epoch in range(1, config.max_epoch + 1):
        # 打印当前epoch信息、批次大小、学习率
        print('#epoch: {}  batch_size: {}  Current Learning Rate: {}'.format(
            epoch, config.batch_size, config.lr))

        # 记录当前epoch的开始时间（用于计算训练耗时）
        since = time.time()

        # 训练阶段：遍历训练集，更新学生模型
        train_loss, train_auc, train_TPR = train_epoch(
            model_large, model_small, optimizer, criterion, train_dataloader
        )

        # 验证阶段：遍历验证集，监控模型泛化能力
        val_loss, val_auc, val_TPR = test_epoch(
            model_large, model_small, criterion, val_dataloader
        )

        # 测试阶段：遍历测试集，评估最终性能
        test_loss, test_auc, test_TPR = test_epoch(
            model_large, model_small, criterion, test_dataloader
        )

        # 保存最佳模型（基于测试集AUC）
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            save_distill_checkpoint(best_test_auc, model_small, optimizer, epoch)

        # 保存当前epoch的训练结果到CSV文件
        # 构建结果列表（包含epoch、各集的损失、AUC、TPR）
        result_list = [
            [epoch, train_loss, train_auc, train_TPR,
             val_loss, val_auc, val_TPR,
             test_loss, test_auc, test_TPR]
        ]
        # 第一次保存时添加CSV表头，后续只添加数据
        if epoch == 1:
            columns = ['epoch', 'train_loss', 'train_auc', 'train_TPR',
                       'val_loss', 'val_auc', 'val_TPR',
                       'test_loss', 'test_auc', 'test_TPR']
        else:
            columns = ['', '', '', '', '', '', '', '', '', '']  # 后续行不重复表头
        # 创建DataFrame并以追加模式写入CSV
        dt = pd.DataFrame(result_list, columns=columns)
        dt.to_csv(
            f'{config.model_name}_{config.model_name2}_{config.experiment}_distill_result.csv',
            mode='a',  # 追加模式（避免覆盖之前的结果）
            index=False  # 不保存索引列
        )

        # 打印当前epoch的训练耗时
        print('time cost of this epoch:%s\n' % (utils.print_time_cost(since)))


# 主函数入口：当脚本直接运行时，启动训练流程
if __name__ == '__main__':
    train(config)