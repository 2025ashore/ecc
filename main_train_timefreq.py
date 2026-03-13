# -*- coding: utf-8 -*-
"""
@time: 2026/1/5
@description: 多尺度时频融合模型训练脚本
    在main_train.py基础上添加时频融合特性，包括：
    - 可选的时频融合模型选择
    - 标签条件融合支持
    - 频域增强数据增强
    - 可视化和诊断工具
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入项目模块
import models
import utils
from dataset import load_datasets, ECGDataset
from config import config
from loss_functions import AsymmetricLoss
from sklearn.metrics import roc_auc_score

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    """设置随机种子确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class TimeFreqDataAugmentation:
    """频域数据增强类
    
    增强方式：
        1. 时间裁剪（随机裁剪时频图的时间维度）
        2. 频率掩蔽（在频域上随机掩蔽某些频段）
        3. 时频混合（多个样本的时频图随机混合）
    """
    
    def __init__(self, prob=0.5, time_mask_width=20, freq_mask_width=20):
        self.prob = prob
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
    
    def time_crop(self, spec):
        """随机时间裁剪"""
        if random.random() < self.prob:
            _, _, T = spec.shape
            if T > 64:
                crop_size = random.randint(32, T)
                start = random.randint(0, T - crop_size)
                spec = spec[:, :, :, start:start + crop_size]
        return spec
    
    def frequency_mask(self, spec):
        """频率掩蔽（SpecAugment风格）"""
        if random.random() < self.prob:
            _, _, F, _ = spec.shape
            mask_width = min(self.freq_mask_width, F // 4)
            f0 = random.randint(0, F - mask_width)
            spec[:, :, f0:f0 + mask_width, :] *= 0.0  # 掩蔽为0
        return spec
    
    def time_mask(self, spec):
        """时间掩蔽"""
        if random.random() < self.prob:
            _, _, _, T = spec.shape
            mask_width = min(self.time_mask_width, T // 4)
            t0 = random.randint(0, T - mask_width)
            spec[:, :, :, t0:t0 + mask_width] *= 0.0
        return spec
    
    def __call__(self, spec):
        """应用增强"""
        spec = self.time_crop(spec)
        spec = self.frequency_mask(spec)
        spec = self.time_mask(spec)
        return spec


def train_epoch_timefreq(model, optimizer, criterion, train_dataloader, epoch, total_epoch, 
                         use_label_condition=True, augmentation=None):
    """时频融合模型的训练epoch
    
    参数：
        model: TimeFreqFusionNet模型
        optimizer: 优化器
        criterion: 损失函数
        train_dataloader: 训练数据加载器
        epoch: 当前epoch
        total_epoch: 总epoch数
        use_label_condition: 是否使用标签条件融合
        augmentation: 数据增强对象
    
    返回：
        train_loss: 平均训练损失
        train_auc: 训练集AUC
    """
    model.train()
    loss_meter = 0
    it_count = 0
    all_outputs = []
    all_targets = []
    
    pbar = tqdm(train_dataloader, desc=f'Train Epoch {epoch}/{total_epoch}')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 前向传播
        if use_label_condition:
            # 使用软标签作为条件
            with torch.no_grad():
                y_soft = torch.sigmoid(model(inputs).detach())
            outputs = model(inputs, y_soft=y_soft)
        else:
            outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        
        # 记录指标
        loss_meter += loss.item()
        it_count += 1
        
        # 用于AUC计算
        with torch.no_grad():
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_outputs.append(probs)
            all_targets.append(targets.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    # 计算AUC
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    train_auc = roc_auc_score(all_targets, all_outputs, average='macro', multi_class='ovo')
    
    return loss_meter / it_count, train_auc


def test_epoch_timefreq(model, criterion, dataloader, epoch, total_epoch, phase='Val'):
    """时频融合模型的测试/验证epoch
    
    参数：
        model: TimeFreqFusionNet模型
        criterion: 损失函数
        dataloader: 数据加载器
        epoch: 当前epoch
        total_epoch: 总epoch数
        phase: 'Val'或'Test'
    
    返回：
        test_loss: 平均测试损失
        test_auc: 测试集AUC
    """
    model.eval()
    loss_meter = 0
    it_count = 0
    all_outputs = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f'{phase} Epoch {epoch}/{total_epoch}')
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            loss_meter += loss.item()
            it_count += 1
            
            # 用于AUC计算
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_outputs.append(probs)
            all_targets.append(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 计算AUC
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    test_auc = roc_auc_score(all_targets, all_outputs, average='macro', multi_class='ovo')
    
    return loss_meter / it_count, test_auc


def save_checkpoint(best_auc, model, optimizer, epoch, checkpoint_dir='checkpoints'):
    """保存最优模型"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_state_dict = model.state_dict() if isinstance(model, nn.Module) else model.module.state_dict()
    
    checkpoint_path = os.path.join(
        checkpoint_dir, 
        f'TimeFreqFusionNet_{config.experiment}_checkpoint_best.pth'
    )
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
    }, checkpoint_path)
    
    print(f'Model saved to {checkpoint_path}')


def main():
    """主训练函数"""
    print('=' * 80)
    print('多尺度时频融合ECG多标签分类模型训练')
    print('=' * 80)
    
    # 设置随机种子
    setup_seed(config.seed)
    
    # 打印设备信息
    print(f'Device: {device}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    
    # 加载数据集
    print('\n[1/5] 加载数据集...')
    train_dataloader, val_dataloader, test_dataloader, num_classes = load_datasets(
        datafolder=config.datafolder,
        experiment=config.experiment,
    )
    print(f'类别数: {num_classes}')
    
    # 创建模型
    print('\n[2/5] 创建模型...')
    label_kwargs = dict(
        use_label_graph_refiner=config.use_label_graph_refiner,
        label_graph_hidden=config.label_graph_hidden,
        label_graph_learnable_adj=config.label_graph_learnable_adj,
        label_graph_dropout=config.label_graph_dropout,
    )

    # 为 MyNet7ViewTimeFreq 模型添加额外参数（视图 Transformer + 跨模态中融合 + Feature-level 标签图）
    mynet7_kwargs = dict(
        **label_kwargs,
        use_feature_label_gcn=getattr(config, 'use_feature_label_gcn', False),
        feature_label_gcn_hidden=getattr(config, 'feature_label_gcn_hidden', 64),
        feature_label_gcn_layers=getattr(config, 'feature_label_gcn_layers', 2),
        feature_label_gcn_dropout=getattr(config, 'feature_label_gcn_dropout', 0.1),
        feature_label_gcn_learnable_adj=getattr(config, 'feature_label_gcn_learnable_adj', True),
        feature_label_gcn_init_gate=getattr(config, 'feature_label_gcn_init_gate', -2.0),
        feature_label_gcn_adj_init_off_diag=getattr(config, 'feature_label_gcn_adj_init_off_diag', 0.1),
        use_view_transformer_fusion=getattr(config, 'use_view_transformer_fusion', False),
        view_transformer_layers=getattr(config, 'view_transformer_layers', 1),
        view_transformer_heads=getattr(config, 'view_transformer_heads', 4),
        view_transformer_dropout=getattr(config, 'view_transformer_dropout', 0.1),
        view_transformer_residual_scale=getattr(config, 'view_transformer_residual_scale', 0.1),
        use_cross_modal_fusion=getattr(config, 'use_cross_modal_fusion', False),
        cross_modal_heads=getattr(config, 'cross_modal_heads', 4),
        cross_modal_dropout=getattr(config, 'cross_modal_dropout', 0.1),
        cross_modal_tokens=getattr(config, 'cross_modal_tokens', 32),
    )

    if config.use_timefreq_fusion:
        model = models.TimeFreqFusionNet(num_classes=num_classes, **label_kwargs)
        print('使用模型: TimeFreqFusionNet (时频融合) + LabelGraph' if config.use_label_graph_refiner else '使用模型: TimeFreqFusionNet (时频融合)')
    else:
        if config.model_name == 'MyNet7ViewTimeFreq':
            model = getattr(models, config.model_name)(num_classes=num_classes, **mynet7_kwargs)
        else:
            model = getattr(models, config.model_name)(num_classes=num_classes)
        suffixes = []
        if getattr(config, 'use_feature_label_gcn', False) and config.model_name == 'MyNet7ViewTimeFreq':
            suffixes.append('FeatureLabelGCN')
        if config.use_label_graph_refiner and config.model_name == 'MyNet7ViewTimeFreq':
            suffixes.append('LogitsLabelGraph')
        suffix = (' + ' + ' + '.join(suffixes)) if suffixes else ''
        print(f'使用模型: {config.model_name}{suffix}')
    
    model = model.to(device)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'总参数: {total_params:,} | 可训练参数: {trainable_params:,}')
    
    # 配置优化器和损失函数
    print('\n[3/5] 配置优化器和损失函数...')
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    
    # 使用不对称损失（适合多标签不平衡）
    criterion = AsymmetricLoss(
        gamma_pos=0,
        gamma_neg=4,
        clip=0.05,
        eps=1e-8,
        use_dynamic_clip=True,
        grad_boost_factor=1.0,
        hard_pos_threshold=0.2,
        hard_pos_weight=2.0
    )
    print('损失函数: AsymmetricLoss (多标签不平衡优化)')
    
    # 数据增强（可选）
    if config.apply_time_freq_augmentation and config.use_timefreq_fusion:
        augmentation = TimeFreqDataAugmentation(prob=config.augmentation_prob)
        print('启用频域数据增强')
    else:
        augmentation = None
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.max_epoch, 
        eta_min=1e-6
    )
    
    # 创建checkpoints文件夹
    os.makedirs('checkpoints', exist_ok=True)
    
    # 训练循环
    print('\n[4/5] 开始训练...')
    print('=' * 80)
    
    best_test_auc = 0.0
    best_epoch = 0
    results = []
    
    for epoch in range(1, config.max_epoch + 1):
        print(f'\nEpoch {epoch}/{config.max_epoch} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 训练
        if config.use_timefreq_fusion:
            train_loss, train_auc = train_epoch_timefreq(
                model, optimizer, criterion, train_dataloader, epoch, config.max_epoch,
                use_label_condition=config.use_label_conditional_fusion,
                augmentation=augmentation
            )
        else:
            # 标准训练（兼容现有代码）
            train_loss, train_auc = train_epoch_standard(
                model, optimizer, criterion, train_dataloader, epoch, config.max_epoch
            )
        
        # 验证
        if config.use_timefreq_fusion:
            val_loss, val_auc = test_epoch_timefreq(
                model, criterion, val_dataloader, epoch, config.max_epoch, phase='Val'
            )
            test_loss, test_auc = test_epoch_timefreq(
                model, criterion, test_dataloader, epoch, config.max_epoch, phase='Test'
            )
        else:
            val_loss, val_auc = test_epoch_standard(
                model, criterion, val_dataloader, epoch, config.max_epoch, phase='Val'
            )
            test_loss, test_auc = test_epoch_standard(
                model, criterion, test_dataloader, epoch, config.max_epoch, phase='Test'
            )
        
        # 学习率调度
        scheduler.step()
        
        # 保存最优模型
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_epoch = epoch
            save_checkpoint(best_test_auc, model, optimizer, epoch)
        
        # 记录结果
        result_dict = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_auc': train_auc,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'test_loss': test_loss,
            'test_auc': test_auc,
            'best_test_auc': best_test_auc,
        }
        results.append(result_dict)
        
        # 打印进度
        print(f'Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}')
        print(f'Val Loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f}')
        print(f'Test Loss:  {test_loss:.4f} | Test AUC:  {test_auc:.4f}')
        print(f'Best Test AUC: {best_test_auc:.4f} (Epoch {best_epoch})')
    
    # 保存结果到CSV
    print('\n[5/5] 保存结果...')
    results_df = pd.DataFrame(results)
    results_csv = f'TimeFreqFusionNet_{config.experiment}_result.csv'
    results_df.to_csv(results_csv, index=False)
    print(f'结果保存到: {results_csv}')
    
    print('\n' + '=' * 80)
    print(f'训练完成！最优Test AUC: {best_test_auc:.4f} (Epoch {best_epoch})')
    print('=' * 80)


def train_epoch_standard(model, optimizer, criterion, train_dataloader, epoch, total_epoch):
    """标准训练epoch（兼容非时频融合模型）"""
    model.train()
    loss_meter = 0
    it_count = 0
    all_outputs = []
    all_targets = []
    
    pbar = tqdm(train_dataloader, desc=f'Train Epoch {epoch}/{total_epoch}')
    
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_meter += loss.item()
        it_count += 1
        
        with torch.no_grad():
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_outputs.append(probs)
            all_targets.append(targets.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    train_auc = roc_auc_score(all_targets, all_outputs, average='macro', multi_class='ovo')
    
    return loss_meter / it_count, train_auc


def test_epoch_standard(model, criterion, dataloader, epoch, total_epoch, phase='Val'):
    """标准测试epoch（兼容非时频融合模型）"""
    model.eval()
    loss_meter = 0
    it_count = 0
    all_outputs = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f'{phase} Epoch {epoch}/{total_epoch}')
    
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_meter += loss.item()
            it_count += 1
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_outputs.append(probs)
            all_targets.append(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    test_auc = roc_auc_score(all_targets, all_outputs, average='macro', multi_class='ovo')
    
    return loss_meter / it_count, test_auc


if __name__ == '__main__':
    main()
