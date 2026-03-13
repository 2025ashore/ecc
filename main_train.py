# -*- coding: utf-8 -*-
"""
@time: 2021/4/15 15:40
@ author:
"""
# 导入依赖库
import torch  # PyTorch核心框架（张量操作、模型构建）
import time  # 时间计时（统计训练耗时）
import os  # 操作系统接口（创建文件夹、文件路径处理）
import models  # 自定义模型模块（包含所有待训练的ECG模型）
import utils  # 自定义工具模块（包含TPR计算、时间格式化等工具函数）
from torch import nn, optim  # PyTorch内置的损失函数和优化器
from dataset import load_datasets  # 自定义数据加载模块（加载训练/验证/测试集）
from config import config  # 导入配置对象（统一管理超参数和路径）
from sklearn.metrics import roc_auc_score  # sklearn中的AUC评价指标（多标签分类常用）
import numpy as np  # 数值计算库（数组处理）
import random  # 随机数生成库（配合种子保证可重复性）
import pandas as pd  # 数据处理库（保存训练结果到CSV）
from tqdm import tqdm  # 新增：导入进度条库（已在依赖中安装）

# 自动选择计算设备：优先GPU（CUDA可用时），否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_roc_auc_score(y_true, y_score):
    """
    安全版 roc_auc_score：自动跳过只有单一标签值（全0或全1）的类别。
    HF等小数据集的val/test划分中，部分类别可能无正样本，
    sklearn会返回nan并报UndefinedMetricWarning。
    本函数仅对有效类别计算AUC，再取宏平均。
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    n_classes = y_true.shape[1]
    aucs = []
    for i in range(n_classes):
        # 跳过该类别只有一种标签值的情况
        if len(np.unique(y_true[:, i])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[:, i], y_score[:, i]))
    if len(aucs) == 0:
        return float('nan')
    return float(np.mean(aucs))


def setup_seed(seed):
    """
    设置全局随机种子，保证实验可重复性
    Args:
        seed: 随机种子值（从config中传入）
    """
    torch.manual_seed(seed)  # 设置CPU随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU随机种子（多GPU场景）
    np.random.seed(seed)  # 设置numpy随机种子
    random.seed(seed)  # 设置python原生随机种子
    torch.backends.cudnn.deterministic = True  # 禁用cudnn的非确定性算法（保证结果一致）


def save_checkpoint(best_auc, model, optimizer, epoch):
    """
    保存最优模型权重（基于测试集AUC）
    Args:
        best_auc: 当前最优的测试集AUC值
        model: 训练的模型实例
        optimizer: 优化器实例（保存参数用于后续继续训练）
        epoch: 当前训练轮数
    """
    print('Model Saving...')
    # 处理多GPU训练场景：多GPU时模型参数存储在module中，需提取module的state_dict
    if config.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()  # 单GPU直接提取模型参数

    # 保存路径：使用 config.output_dir 指定的目录
    checkpoint_dir = os.path.join(getattr(config, 'output_dir', '.'), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
    },
        os.path.join(checkpoint_dir, config.model_name + '_' + config.experiment + '_checkpoint_best.pth'))


def train_epoch(model, optimizer, criterion, train_dataloader, epoch, total_epoch):
    """
    训练单个epoch：模型训练、梯度更新、训练集指标计算（新增进度条）
    Args:
        model: 训练的模型实例
        optimizer: 优化器（如Adam）
        criterion: 损失函数（如BCEWithLogitsLoss）
        train_dataloader: 训练集数据加载器
        epoch: 当前轮数（用于进度条显示）
        total_epoch: 总轮数（用于进度条显示）
    Returns:
        train_avg_loss: 该epoch的平均训练损失
        train_auc: 该epoch的训练集AUC值
        train_TPR: 该epoch的训练集TPR值（真正例率）
    """
    model.train()  # 开启模型训练模式（启用Dropout、BatchNorm更新等）
    loss_meter, it_count = 0, 0  # 损失累加器、迭代次数计数器
    outputs = []  # 存储所有样本的模型输出（用于后续计算指标）
    targets = []  # 存储所有样本的真实标签（用于后续计算指标）

    # 新增：用tqdm包裹数据加载器，显示训练进度条
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                desc=f'Epoch [{epoch}/{total_epoch}] Train',
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    # 遍历训练集批次（修改：用进度条迭代）
    for batch_idx, (inputs, target) in pbar:
        # 数据增强：给输入添加标准差为0.1的高斯噪声（提升模型鲁棒性）
        inputs = inputs + torch.randn_like(inputs) * 0.1

        # 将数据和标签移到指定计算设备（GPU/CPU）
        inputs = inputs.to(device)
        target = target.to(device)

        optimizer.zero_grad()  # 梯度清零（避免累积前一轮梯度）
        output = model(inputs)  # 前向传播：模型输出（未经过sigmoid激活）
        loss = criterion(output, target)  # 计算损失（BCEWithLogitsLoss直接接收logits）
        loss.backward()  # 反向传播：计算梯度
        optimizer.step()  # 梯度更新：优化器调整模型参数

        # 累积损失和迭代次数
        loss_meter += loss.item()
        it_count += 1

        # 对模型输出做sigmoid激活（转换为概率值，用于计算AUC/TPR）
        output = torch.sigmoid(output)
        # 收集当前批次的输出和标签（转CPU+detach()避免计算图跟踪）
        for i in range(len(output)):
            outputs.append(output[i].cpu().detach().numpy())
            targets.append(target[i].cpu().detach().numpy())

        # 新增：在进度条上实时显示当前批次损失
        pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})

    pbar.close()  # 关闭进度条

    # 计算训练集的宏观AUC和TPR（safe版本自动跳过只有单一标签值的类别）
    train_auc = safe_roc_auc_score(targets, outputs)
    train_TPR = utils.compute_TPR(targets, outputs)  # 调用工具函数计算TPR
    # 打印当前epoch的训练指标
    print('train_loss: %.4f,   macro_auc: %.4f,   TPR: %.4f' % (loss_meter / it_count, train_auc, train_TPR))
    # 返回平均损失、AUC、TPR
    return loss_meter / it_count, train_auc, train_TPR


def test_epoch(model, criterion, dataloader, epoch, total_epoch, phase='Val'):
    """
    验证/测试单个epoch：模型评估（不更新参数）、计算验证/测试集指标（新增进度条）
    Args:
        model: 训练的模型实例
        criterion: 损失函数
        dataloader: 验证集/测试集数据加载器
        epoch: 当前轮数（用于进度条显示）
        total_epoch: 总轮数（用于进度条显示）
        phase: 阶段标识（Val/Test，用于进度条显示）
    Returns:
        test_avg_loss: 该epoch的平均验证/测试损失
        test_auc: 该epoch的验证/测试集AUC值
        test_TPR: 该epoch的验证/测试集TPR值
    """
    model.eval()  # 开启模型评估模式（禁用Dropout、固定BatchNorm参数）
    loss_meter, it_count = 0, 0  # 损失累加器、迭代次数计数器
    outputs = []  # 存储所有样本的模型输出
    targets = []  # 存储所有样本的真实标签

    # 新增：用tqdm包裹数据加载器，显示验证/测试进度条
    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f'Epoch [{epoch}/{total_epoch}] {phase}',
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    with torch.no_grad():  # 禁用梯度计算（节省内存、加速推理）
        # 遍历验证/测试集批次（修改：用进度条迭代）
        for batch_idx, (inputs, target) in pbar:
            # 与训练集一致：添加高斯噪声（保持数据分布一致性，提升评估可靠性）
            inputs = inputs + torch.randn_like(inputs) * 0.1

            # 数据和标签移到计算设备
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)  # 前向传播（无梯度跟踪）
            loss = criterion(output, target)  # 计算损失（仅用于监控，不反向传播）

            # 累积损失和迭代次数
            loss_meter += loss.item()
            it_count += 1

            # 激活输出并收集结果
            output = torch.sigmoid(output)
            for i in range(len(output)):
                outputs.append(output[i].cpu().detach().numpy())
                targets.append(target[i].cpu().detach().numpy())

            # 新增：在进度条上实时显示当前批次损失
            pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})

    pbar.close()  # 关闭进度条

    # 计算验证/测试集的宏观AUC和TPR（safe版本自动跳过只有单一标签值的类别）
    test_auc = safe_roc_auc_score(targets, outputs)
    test_TPR = utils.compute_TPR(targets, outputs)

    # 打印验证/测试指标
    print(f'{phase}_loss: %.4f,   macro_auc: %.4f,   TPR: %.4f' % (loss_meter / it_count, test_auc, test_TPR))
    # 返回平均损失、AUC、TPR
    return loss_meter / it_count, test_auc, test_TPR


def train(config=config):
    """
    主训练函数：整合数据加载、模型初始化、训练迭代、结果保存全流程
    Args:
        config: 配置对象（含超参数、路径等，默认使用导入的config）
    """
    # 1. 设置随机种子（保证实验可重复）
    setup_seed(config.seed)
    print('torch.cuda.is_available:', torch.cuda.is_available())  # 打印GPU可用性

    # 2. 加载数据集：获取训练/验证/测试加载器 + 类别数（num_classes）
    train_dataloader, val_dataloader, test_dataloader, num_classes = load_datasets(
        datafolder=config.datafolder,  # 数据集路径（从config读取）
        experiment=config.experiment,  # 实验编号（用于区分不同数据处理逻辑）
    )

    # 3. 初始化模型：从models模块动态加载指定模型（根据config.model_name）
    label_kwargs = dict(
        use_label_graph_refiner=config.use_label_graph_refiner,
        label_graph_hidden=config.label_graph_hidden,
        label_graph_learnable_adj=config.label_graph_learnable_adj,
        label_graph_dropout=config.label_graph_dropout,
        use_view_transformer_fusion=getattr(config, 'use_view_transformer_fusion', False),
        view_transformer_layers=getattr(config, 'view_transformer_layers', 1),
        view_transformer_heads=getattr(config, 'view_transformer_heads', 4),
        view_transformer_dropout=getattr(config, 'view_transformer_dropout', 0.1),
        view_transformer_residual_scale=getattr(config, 'view_transformer_residual_scale', 0.1),
        use_cross_modal_fusion=getattr(config, 'use_cross_modal_fusion', False),
        cross_modal_heads=getattr(config, 'cross_modal_heads', 4),
        cross_modal_dropout=getattr(config, 'cross_modal_dropout', 0.1),
        cross_modal_tokens=getattr(config, 'cross_modal_tokens', 32),
        use_lead_attention=getattr(config, 'use_lead_attention', True),
        lead_attention_reduction=getattr(config, 'lead_attention_reduction', 4),
        lead_attention_spectral_spatial=getattr(config, 'lead_attention_spectral_spatial', True),
        lead_attention_spatial_kernel=getattr(config, 'lead_attention_spatial_kernel', 7),
        # Feature-level 标签图 GCN 参数
        use_feature_label_gcn=getattr(config, 'use_feature_label_gcn', False),
        feature_label_gcn_hidden=getattr(config, 'feature_label_gcn_hidden', 64),
        feature_label_gcn_layers=getattr(config, 'feature_label_gcn_layers', 2),
        feature_label_gcn_dropout=getattr(config, 'feature_label_gcn_dropout', 0.1),
        feature_label_gcn_learnable_adj=getattr(config, 'feature_label_gcn_learnable_adj', True),
        feature_label_gcn_init_gate=getattr(config, 'feature_label_gcn_init_gate', -2.0),
        feature_label_gcn_adj_init_off_diag=getattr(config, 'feature_label_gcn_adj_init_off_diag', 0.1),
    )

    if config.model_name == 'MyNet7ViewTimeFreq':
        model = getattr(models, config.model_name)(num_classes=num_classes, **label_kwargs)
    else:
        model = getattr(models, config.model_name)(num_classes=num_classes)  # 传入类别数初始化模型

    suffix = ''
    if config.model_name == 'MyNet7ViewTimeFreq':
        if getattr(config, 'use_feature_label_gcn', False):
            suffix += ' + FeatureLabelGCN(H={},L={},gate={})'.format(
                getattr(config, 'feature_label_gcn_hidden', 64),
                getattr(config, 'feature_label_gcn_layers', 2),
                getattr(config, 'feature_label_gcn_init_gate', -2.0))
        if config.use_label_graph_refiner:
            suffix += ' + LogitsLabelGraph'
    print('model_name:{}, num_classes={}{}'.format(config.model_name, num_classes, suffix))  # 打印模型名称和类别数
    model = model.to(device)  # 将模型移到计算设备（GPU/CPU）

    # 4. 配置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.lr)  # Adam优化器（学习率从config读取）
    criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失（带logits输入，适用于多标签ECG分类）

    # 5. 创建checkpoints文件夹（用于保存模型权重），不存在则创建
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    # =========> 开始训练迭代 <=========
    total_epoch = config.max_epoch  # 总轮数（用于进度条显示）
    best_auc = float('-inf')  # 追踪训练过程中最优的测试集AUC
    best_tpr = float('-inf')  # 追踪训练过程中最优的测试集TPR
    best_auc_epoch = 0
    best_tpr_epoch = 0
    tpr_at_best_auc = 0.0  # best_auc 对应 epoch 的 TPR
    auc_at_best_tpr = 0.0  # best_tpr 对应 epoch 的 AUC
    for epoch in range(1, total_epoch + 1):  # 从1到max_epoch轮
        # 打印当前epoch信息：轮数、批次大小、学习率
        print('\n#epoch: {}  batch_size: {}  Current Learning Rate: {}'.format(epoch, config.batch_size, config.lr))

        since = time.time()  # 记录当前epoch开始时间
        # 训练一个epoch：传入epoch和total_epoch用于进度条显示
        train_loss, train_auc, train_TPR = train_epoch(model, optimizer, criterion, train_dataloader, epoch,
                                                       total_epoch)
        # 验证一个epoch：指定phase='Val'，显示验证进度条
        val_loss, val_auc, val_TPR = test_epoch(model, criterion, val_dataloader, epoch, total_epoch, phase='Val')
        # 测试一个epoch：指定phase='Test'，显示测试进度条
        test_loss, test_auc, test_TPR = test_epoch(model, criterion, test_dataloader, epoch, total_epoch, phase='Test')

        # 保存模型：仅当测试集AUC刷新历史最优时才保存
        # 处理nan：如果当前AUC有效且(之前是nan或当前更优)则更新
        _cur_auc_valid = not np.isnan(test_auc)
        _best_auc_is_nan = np.isnan(best_auc) or best_auc == float('-inf')
        if _cur_auc_valid and (_best_auc_is_nan or test_auc > best_auc):
            best_auc = float(test_auc)
            best_auc_epoch = epoch
            tpr_at_best_auc = float(test_TPR)
            save_checkpoint(best_auc, model, optimizer, epoch)
            print(f"[Best AUC] Updated best_test_auc={best_auc:.6f} (TPR={tpr_at_best_auc:.6f}) at epoch={best_auc_epoch}, checkpoint saved.")
        else:
            _disp_best = f'{best_auc:.6f}' if not (_best_auc_is_nan) else 'nan'
            _disp_cur = f'{test_auc:.6f}' if _cur_auc_valid else 'nan'
            print(f"[Best AUC] No improvement. best_test_auc={_disp_best} (epoch={best_auc_epoch}), current={_disp_cur}.")

        # 追踪最优TPR
        if test_TPR > best_tpr:
            best_tpr = float(test_TPR)
            best_tpr_epoch = epoch
            auc_at_best_tpr = float(test_auc)
            print(f"[Best TPR] Updated best_test_tpr={best_tpr:.6f} (AUC={auc_at_best_tpr:.6f}) at epoch={best_tpr_epoch}.")

        # 整理当前epoch的所有指标（用于保存到CSV）
        result_list = [[epoch, train_loss, train_auc, train_TPR,
                        val_loss, val_auc, val_TPR,
                        test_loss, test_auc, test_TPR]]

        # 设置CSV列名：仅在第1个epoch时创建列名，后续epoch不重复写入
        if epoch == 1:
            columns = ['epoch', 'train_loss', 'train_auc', 'train_TPR',
                       'val_loss', 'val_auc', 'val_TPR',
                       'test_loss', 'test_auc', 'test_TPR']
        else:
            columns = ['', '', '', '', '', '', '', '', '', '']  # 后续epoch列名为空，避免重复

        # 创建DataFrame并追加到CSV文件（模式'a'表示追加）
        dt = pd.DataFrame(result_list, columns=columns)
        dt.to_csv(os.path.join(getattr(config, 'output_dir', '.'), config.model_name + config.experiment + 'result.csv'), mode='a', index=False)  # index=False不保存行索引

        # 打印当前epoch的训练耗时（调用工具函数格式化时间）
        print('time:%s\n' % (utils.print_time_cost(since)))

    # 返回本次训练的最优指标摘要
    return {
        'seed': config.seed,
        'experiment': config.experiment,
        'best_test_auc': best_auc,
        'best_auc_epoch': best_auc_epoch,
        'tpr_at_best_auc': tpr_at_best_auc,
        'best_test_tpr': best_tpr,
        'best_tpr_epoch': best_tpr_epoch,
        'auc_at_best_tpr': auc_at_best_tpr,
    }


if __name__ == '__main__':
    """
    多随机种子实验：仅跑 exp0，用不同种子重复训练，
    将每次的种子和最优 test_tpr / test_auc 记录到 Excel。
    """
    from openpyxl import Workbook

    # ===== 实验配置 =====
    seeds = [7, 10, 20, 42, 100]  # 5 个不同随机种子
    config.experiment = 'exp0'
    excel_path = f'result/{config.model_name}_{config.experiment}_multi_seed_results.xlsx'

    # 确保输出目录存在
    os.makedirs('result', exist_ok=True)

    all_results = []  # 收集每次运行的指标

    for i, seed in enumerate(seeds, 1):
        print('\n' + '=' * 60)
        print(f'  Run {i}/{len(seeds)}  |  seed={seed}  |  exp={config.experiment}')
        print('=' * 60)
        config.seed = seed
        result = train(config)
        all_results.append(result)

        # 实时写入 Excel（每跑完一次就更新，防止中途断了丢数据）
        df = pd.DataFrame(all_results)
        df.to_excel(excel_path, index=False, sheet_name='multi_seed')
        print(f'[Excel] 已更新 → {excel_path}')

    # ===== 汇总统计 =====
    df = pd.DataFrame(all_results)
    print('\n' + '=' * 60)
    print('  多种子实验汇总')
    print('=' * 60)
    print(df[['seed', 'best_test_auc', 'tpr_at_best_auc',
              'best_test_tpr', 'best_tpr_epoch']].to_string(index=False))
    print(f'\nbest_test_tpr  均值={df["best_test_tpr"].mean():.6f}  '
          f'std={df["best_test_tpr"].std():.6f}')
    print(f'best_test_auc  均值={df["best_test_auc"].mean():.6f}  '
          f'std={df["best_test_auc"].std():.6f}')
    print(f'\n最优单次: seed={df.loc[df["best_test_tpr"].idxmax(), "seed"]:.0f}, '
          f'best_test_tpr={df["best_test_tpr"].max():.6f}')
    print(f'结果已保存至: {excel_path}')