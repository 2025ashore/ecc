# -*- coding: utf-8 -*-
"""
一键批量训练所有模型（排除 MyNet6View）
每个模型的结果保存到独立文件夹：result/<模型名>/
包含：
  - 每个实验的 CSV 训练日志
  - 模型权重 checkpoint
  - 最终汇总表 summary.csv（所有实验的最佳指标）
运行方式：python run_all_models.py
"""

import torch
import time
import os
import models
import utils
from torch import nn, optim
from dataset import load_datasets
from config import config
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 要训练的模型列表（排除 MyNet6View） ====================
ALL_MODELS = [
    'resnet1d_wang',
    'xresnet1d101',
    'xresnet1d50',
    'inceptiontime',
    'fcn_wang',
    'lstm',
    'lstm_bidir',
    'vit',
    'mobilenetv3_small',
    'mobilenetv3_large',
    'dccacb',
    'ATI_CNN',
    'MyNet',
    # 'MyNet6View',  # 已排除
]

# ==================== 实验配置（实验编号 -> 随机种子） ====================
EXPERIMENTS = {
    'exp0': 10,
    'exp1': 20,
    'exp1.1': 20,
    'exp1.1.1': 20,
    'exp2': 7,
    'exp3': 10,
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(best_auc, model, optimizer, epoch, save_dir, model_name, experiment):
    """保存模型权重到指定目录"""
    print('Model Saving...')
    if config.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
    }, os.path.join(save_dir, f'{model_name}_{experiment}_checkpoint_best.pth'))


def train_epoch(model, optimizer, criterion, train_dataloader, epoch, total_epoch):
    model.train()
    loss_meter, it_count = 0, 0
    outputs, targets = [], []

    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                desc=f'Epoch [{epoch}/{total_epoch}] Train',
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    for batch_idx, (inputs, target) in pbar:
        inputs = inputs + torch.randn_like(inputs) * 0.1
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_meter += loss.item()
        it_count += 1

        output = torch.sigmoid(output)
        for i in range(len(output)):
            outputs.append(output[i].cpu().detach().numpy())
            targets.append(target[i].cpu().detach().numpy())
        pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})

    pbar.close()
    train_auc = roc_auc_score(targets, outputs)
    train_TPR = utils.compute_TPR(targets, outputs)
    print('train_loss: %.4f,   macro_auc: %.4f,   TPR: %.4f' % (loss_meter / it_count, train_auc, train_TPR))
    return loss_meter / it_count, train_auc, train_TPR


def test_epoch(model, criterion, dataloader, epoch, total_epoch, phase='Val'):
    model.eval()
    loss_meter, it_count = 0, 0
    outputs, targets = [], []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f'Epoch [{epoch}/{total_epoch}] {phase}',
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    with torch.no_grad():
        for batch_idx, (inputs, target) in pbar:
            inputs = inputs + torch.randn_like(inputs) * 0.1
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)
            loss = criterion(output, target)

            loss_meter += loss.item()
            it_count += 1

            output = torch.sigmoid(output)
            for i in range(len(output)):
                outputs.append(output[i].cpu().detach().numpy())
                targets.append(target[i].cpu().detach().numpy())
            pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})

    pbar.close()
    test_auc = roc_auc_score(targets, outputs)
    test_TPR = utils.compute_TPR(targets, outputs)
    print(f'{phase}_loss: %.4f,   macro_auc: %.4f,   TPR: %.4f' % (loss_meter / it_count, test_auc, test_TPR))
    return loss_meter / it_count, test_auc, test_TPR


def train_single_model(model_name, experiments, datafolder='data/ptbxl/'):
    """
    训练单个模型的所有实验，结果保存到 result/<model_name>/ 目录
    """
    # 创建模型专属文件夹
    result_dir = os.path.join('result', model_name)
    ckpt_dir = os.path.join('result', model_name, 'checkpoints')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 用于汇总该模型所有实验的最佳指标
    summary_rows = []

    for exp, seed in experiments.items():
        print('\n' + '=' * 70)
        print(f'  模型: {model_name}  |  实验: {exp}  |  种子: {seed}  |  数据集: {datafolder}')
        print('=' * 70)

        # 设置种子
        setup_seed(seed)
        print('torch.cuda.is_available:', torch.cuda.is_available())

        # 加载数据
        train_dataloader, val_dataloader, test_dataloader, num_classes = load_datasets(
            datafolder=datafolder,
            experiment=exp,
        )

        # 初始化模型
        try:
            model = getattr(models, model_name)(num_classes=num_classes)
        except Exception as e:
            print(f'[ERROR] 模型 {model_name} 初始化失败: {e}')
            print(f'[SKIP] 跳过 {model_name} - {exp}')
            continue

        print(f'model_name: {model_name}, num_classes={num_classes}')
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.BCEWithLogitsLoss()

        # CSV 文件路径：result/<model_name>/<model_name>_<exp>_result.csv
        csv_path = os.path.join(result_dir, f'{model_name}_{exp}_result.csv')
        # 如果之前有残留文件，先删除（避免追加到旧数据上）
        if os.path.exists(csv_path):
            os.remove(csv_path)

        best_test_auc = 0.0
        best_epoch_info = {}

        total_epoch = config.max_epoch
        for epoch in range(1, total_epoch + 1):
            print(f'\n#epoch: {epoch}  batch_size: {config.batch_size}  lr: {config.lr}')
            since = time.time()

            train_loss, train_auc, train_TPR = train_epoch(
                model, optimizer, criterion, train_dataloader, epoch, total_epoch)
            val_loss, val_auc, val_TPR = test_epoch(
                model, criterion, val_dataloader, epoch, total_epoch, phase='Val')
            test_loss, test_auc, test_TPR = test_epoch(
                model, criterion, test_dataloader, epoch, total_epoch, phase='Test')

            # 保存最优模型（基于 test_auc）
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                save_checkpoint(test_auc, model, optimizer, epoch, ckpt_dir, model_name, exp)
                best_epoch_info = {
                    'epoch': epoch,
                    'train_loss': train_loss, 'train_auc': train_auc, 'train_TPR': train_TPR,
                    'val_loss': val_loss, 'val_auc': val_auc, 'val_TPR': val_TPR,
                    'test_loss': test_loss, 'test_auc': test_auc, 'test_TPR': test_TPR,
                }

            # 保存每个 epoch 的结果到 CSV
            result_list = [[epoch, train_loss, train_auc, train_TPR,
                            val_loss, val_auc, val_TPR,
                            test_loss, test_auc, test_TPR]]

            if epoch == 1:
                columns = ['epoch', 'train_loss', 'train_auc', 'train_TPR',
                           'val_loss', 'val_auc', 'val_TPR',
                           'test_loss', 'test_auc', 'test_TPR']
            else:
                columns = ['', '', '', '', '', '', '', '', '', '']

            dt = pd.DataFrame(result_list, columns=columns)
            dt.to_csv(csv_path, mode='a', index=False)

            print('time: %s' % utils.print_time_cost(since))

        # 记录该实验的最佳结果
        if best_epoch_info:
            summary_rows.append({
                'model': model_name,
                'experiment': exp,
                'seed': seed,
                'best_epoch': best_epoch_info['epoch'],
                'best_test_auc': best_epoch_info['test_auc'],
                'best_test_TPR': best_epoch_info['test_TPR'],
                'best_val_auc': best_epoch_info['val_auc'],
                'best_val_TPR': best_epoch_info['val_TPR'],
                'best_train_auc': best_epoch_info['train_auc'],
            })

        # 释放GPU显存
        del model
        torch.cuda.empty_cache()

    # 保存该模型的汇总表
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(result_dir, f'{model_name}_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f'\n[SUMMARY] {model_name} 汇总已保存到: {summary_path}')

    return summary_rows


if __name__ == '__main__':
    """
    一键批量训练所有模型（排除 MyNet6View），结果按模型名分目录保存
    """
    print('=' * 70)
    print('  批量训练开始')
    print(f'  待训练模型: {ALL_MODELS}')
    print(f'  实验列表: {list(EXPERIMENTS.keys())}')
    print(f'  数据集: {config.datafolder}')
    print(f'  设备: {device}')
    print('=' * 70)

    all_summary = []  # 所有模型所有实验的汇总

    for idx, model_name in enumerate(ALL_MODELS):
        print('\n' + '#' * 70)
        print(f'  [{idx + 1}/{len(ALL_MODELS)}] 开始训练模型: {model_name}')
        print('#' * 70)

        try:
            rows = train_single_model(model_name, EXPERIMENTS, datafolder=config.datafolder)
            all_summary.extend(rows)
        except Exception as e:
            print(f'\n[ERROR] 模型 {model_name} 训练失败: {e}')
            import traceback
            traceback.print_exc()
            continue

    # 保存所有模型的全局汇总表
    if all_summary:
        global_summary = pd.DataFrame(all_summary)
        global_summary_path = os.path.join('result', 'all_models_summary.csv')
        global_summary.to_csv(global_summary_path, index=False)
        print('\n' + '=' * 70)
        print(f'  全部训练完成！全局汇总表已保存到: {global_summary_path}')
        print('=' * 70)
        print('\n各模型最佳 Test AUC:')
        print(global_summary.groupby('model')['best_test_auc'].max().sort_values(ascending=False).to_string())
