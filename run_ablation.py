# -*- coding: utf-8 -*-
"""
消融实验一键运行脚本
=====================

基于 5 个创新点的三组递进式消融实验设计（A → A+B → A+B+C）

五个创新点:
    ① 多视图多尺度时频双分支架构 — 6 导联视图 + 可学习 STFT 频域视图
    ② 视图级 Transformer + 残差融合 — 8 视图 token 经 Transformer 交互后缩放注入
    ③ 跨模态中层双向交叉注意力 — 时域-频域特征图级信息交换 (Mid-Fusion)
    ④ Lead Attention — 频域导联通道注意力 + 频谱-时间空间注意力
    ⑤ Sobel 形态梯度专家视图 — 不可学习微分 + 门控渐进激活
    + Feature-level Label GCN — 标签共现从 logits 下放到特征空间
    + Logits-level LabelGraphRefiner — logits 层标签图平滑后处理

三模块分组:
┌──────────────┬────────────────────────────────────────────────────────────────────┐
│ 实验组       │ 启用的模块                                                        │
├──────────────┼────────────────────────────────────────────────────────────────────┤
│ A (Base)     │ 模块A: 专家先验驱动的多模态特征提取                               │
│              │   创新①: 时频双分支 (8视图架构, LearnableSTFT+SpectrogramCNN)      │
│              │   创新⑤: Sobel形态梯度专家视图 (一阶微分+门控+MyNet骨干)          │
│              │   ※ 融合方式: 纯自适应权重后融合 (Late Fusion Only)               │
├──────────────┼────────────────────────────────────────────────────────────────────┤
│ A + B        │ 模块A + 模块B: 全局-局部跨模态深度融合网络                        │
│              │   + 创新②: Transformer视图残差融合 (View-level Transformer)        │
│              │   + 创新③: 跨模态中融合 (Cross-Modal Bidirectional Attention)      │
│              │   + 创新④: Lead Attention (导联通道+频谱时间空间注意力)            │
├──────────────┼────────────────────────────────────────────────────────────────────┤
│ A + B + C    │ 完整模型: 全部创新点                                              │
│              │   + Feature-level Label GCN (特征空间标签共现图卷积)               │
│              │   + Logits-level LabelGraphRefiner (logits层标签依赖平滑)          │
└──────────────┴────────────────────────────────────────────────────────────────────┘

模块与 config 开关的精确映射:
    ┌────────┬──────────────────────────────────────────┬─────┬─────┬───────┐
    │ 模块   │ config 开关                              │  A  │ A+B │ A+B+C │
    ├────────┼──────────────────────────────────────────┼─────┼─────┼───────┤
    │ B      │ use_view_transformer_fusion              │ OFF │ ON  │  ON   │
    │ B      │ use_cross_modal_fusion                   │ OFF │ ON  │  ON   │
    │ B      │ use_lead_attention                       │ OFF │ ON  │  ON   │
    │ B      │ lead_attention_spectral_spatial           │ OFF │ ON  │  ON   │
    │ C      │ use_feature_label_gcn                    │ OFF │ OFF │  ON   │
    │ C      │ use_label_graph_refiner                  │ OFF │ OFF │  ON   │
    └────────┴──────────────────────────────────────────┴─────┴─────┴───────┘
    (模块A的8视图架构+Sobel视图始终启用，无需额外开关)

运行方式:
    python run_ablation.py                                  # 默认: 全部6个PTB-XL实验, seed=10
    python run_ablation.py --dataset hf                     # 一键跑HF数据集消融实验
    python run_ablation.py --dataset hf --seeds 9 10 42     # HF数据集 + 多种子
    python run_ablation.py --experiments exp0 exp1           # 只跑exp0和exp1
    python run_ablation.py --seeds 10 42                     # 多种子
    python run_ablation.py --groups A A+B+C                  # 只跑A和完整模型

输出文件 (result/ 文件夹):
    ablation_results_YYYYMMDD_HHMMSS.xlsx   双Sheet: 详细结果 + 消融汇总
    ablation_summary_YYYYMMDD_HHMMSS.csv    汇总表 (AUC/TPR 均值±标准差)
    ablation_config_YYYYMMDD_HHMMSS.json    完整配置快照 (论文复现用)

每组训练过程中的中间文件:
    {model_name}{experiment}_abl_{group}result.csv    逐epoch训练曲线
    checkpoints/{model_name}_{experiment}_abl_{group}_checkpoint_best.pth   最优权重
"""

import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime

# ============================================================
# 消融实验配置定义
# ============================================================

ABLATION_GROUPS = {
    # ─────────────────────────────────────────────────────────
    # 组A: 仅模块A（专家先验驱动的多模态特征提取）
    #   创新①: 8视图架构 (6导联分组 + LearnableSTFT频域视图)
    #   创新⑤: Sobel形态梯度专家视图 (不可学习微分 + 门控渐进激活)
    #   融合: 纯自适应权重后融合 (AdaptiveWeight + Softmax)
    #   关闭: Transformer、跨模态中融合、导联注意力、标签图GCN
    # ─────────────────────────────────────────────────────────
    'A': {
        'description': '模块A: 专家先验多模态特征提取 (创新①时频双分支 + 创新⑤Sobel)',
        'modules': '创新①(时频双分支架构) + 创新⑤(Sobel形态专家视图)',
        'overrides': {
            # ---- 模块B 全部关闭 ----
            'use_cross_modal_fusion': False,           # 关闭跨模态中融合 (Cross-Modal Mid-Fusion)
            'use_lead_attention': False,               # 关闭导联注意力 (Lead Attention)
            'lead_attention_spectral_spatial': False,   # 关闭频率-时间空间注意力
            'use_view_transformer_fusion': False,      # 关闭Transformer视图残差融合
            # ---- 模块C 全部关闭 ----
            'use_feature_label_gcn': False,            # 关闭Feature层标签图GCN
            'use_label_graph_refiner': False,          # 关闭Logits层标签依赖细化
        }
    },

    # ─────────────────────────────────────────────────────────
    # 组A+B: 模块A + 模块B（全局-局部跨模态深度融合网络）
    #   在A基础上启用:
    #   - 创新②: Transformer视图残差融合 (8视图token → Transformer → 残差delta)
    #   - 创新③: 跨模态中融合 (时域Q→频域K/V + 频域Q→时域K/V 双向交叉注意力)
    #   - 创新④: 导联注意力 (12导联通道注意力 + 频率-时间空间注意力)
    # ─────────────────────────────────────────────────────────
    'A+B': {
        'description': '模块A+B: 多模态提取 + 跨模态深度融合',
        'modules': '创新①②③④⑤ (A + Transformer残差 + 跨模态中融合 + Lead Attention)',
        'overrides': {
            # ---- 模块B 全部开启 ----
            'use_cross_modal_fusion': True,            # 启用跨模态中融合
            'use_lead_attention': True,                # 启用导联注意力
            'lead_attention_spectral_spatial': True,    # 启用频率-时间空间注意力
            'use_view_transformer_fusion': True,       # 启用Transformer视图残差融合
            # ---- 模块C 全部关闭 ----
            'use_feature_label_gcn': False,            # 关闭Feature层标签图GCN
            'use_label_graph_refiner': False,          # 关闭Logits层标签依赖细化
        }
    },

    # ─────────────────────────────────────────────────────────
    # 组A+B+C: 完整模型（全部创新点 + 标签图推理）
    #   在A+B基础上启用:
    #   - Feature-level Label GCN (fused_feat 上建模标签共现关系)
    #   - Logits-level LabelGraphRefiner (分类输出后标签图平滑)
    # ─────────────────────────────────────────────────────────
    'A+B+C': {
        'description': '完整模型: 多模态提取 + 跨模态融合 + 标签图推理',
        'modules': '创新①②③④⑤ + Feature-LabelGCN + Logits-LabelRefiner',
        'overrides': {
            # ---- 模块B 全部开启 ----
            'use_cross_modal_fusion': True,
            'use_lead_attention': True,
            'lead_attention_spectral_spatial': True,
            'use_view_transformer_fusion': True,
            # ---- 模块C 全部开启 ----
            'use_feature_label_gcn': True,             # 启用Feature层标签图GCN
            'use_label_graph_refiner': True,            # 启用Logits层标签图细化
        }
    },
}

# 所有模块开关的键名列表（用于打印状态和校验）
ALL_SWITCH_KEYS = [
    'use_cross_modal_fusion',
    'use_lead_attention',
    'lead_attention_spectral_spatial',
    'use_view_transformer_fusion',
    'use_feature_label_gcn',
    'use_label_graph_refiner',
]


# PTB-XL 全部实验编号
PTBXL_EXPERIMENTS = ['exp0', 'exp1', 'exp1.1', 'exp1.1.1', 'exp2', 'exp3']

# CPSC 数据集实验编号（CPSC 只有一个任务，实验标识为 'cpsc'）
CPSC_EXPERIMENTS = ['cpsc']

# HF 数据集实验编号（HF 只有一个任务，无需按 exp0~exp3 拆分）
HF_EXPERIMENTS = ['hf']

# 数据集路径映射
DATASET_FOLDERS = {
    'ptbxl': 'data/ptbxl/',
    'cpsc': 'data/CPSC/',
    'hf': 'data/hf/',
}

# 各数据集默认实验列表
DATASET_EXPERIMENTS = {
    'ptbxl': PTBXL_EXPERIMENTS,
    'cpsc': CPSC_EXPERIMENTS,
    'hf': HF_EXPERIMENTS,
}


# ============================================================
# 消融实验主函数
# ============================================================

def run_ablation(seeds=None, experiments=None, groups=None, dataset='ptbxl'):
    """运行三组递进式消融实验，遍历所有指定实验编号，每组使用多随机种子
    
    Args:
        seeds: 随机种子列表 (默认 [10])
        experiments: 实验编号列表 (默认根据 dataset 自动选择)
        groups: 要运行的消融组列表 (默认 ['A', 'A+B', 'A+B+C'])
        dataset: 数据集名称 ('ptbxl' 或 'hf', 默认 'ptbxl')
    """

    # 延迟导入，避免顶层导入时 config 就被固化
    from config import config
    from main_train import train

    # ===== 实验参数 =====
    if dataset not in DATASET_FOLDERS:
        raise ValueError(f"未知数据集 '{dataset}'，可选: {list(DATASET_FOLDERS.keys())}")
    # 各数据集默认种子：ptbxl=10, cpsc=10, hf=10
    DATASET_DEFAULT_SEEDS = {'ptbxl': [10], 'cpsc': [10], 'hf': [10]}
    if seeds is None:
        seeds = DATASET_DEFAULT_SEEDS.get(dataset, [10])
    if experiments is None:
        experiments = DATASET_EXPERIMENTS[dataset]
    if groups is None:
        groups = ['A', 'A+B', 'A+B+C']
    
    # 校验消融组名称
    for g in groups:
        if g not in ABLATION_GROUPS:
            raise ValueError(f"未知消融组 '{g}'，可选: {list(ABLATION_GROUPS.keys())}")

    # 设置数据集路径
    datafolder = DATASET_FOLDERS[dataset]

    # ===== 输出路径 =====
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f'result/ablation_{dataset}_{timestamp}'
    os.makedirs(base_dir, exist_ok=True)
    # 全局汇总文件
    global_excel_path = f'{base_dir}/ablation_all.xlsx'
    global_summary_csv = f'{base_dir}/ablation_summary_all.csv'
    ablation_config_path = f'{base_dir}/ablation_config.json'
    ablation_config_record = {}

    # 为每个实验创建单独的子文件夹
    exp_dirs = {}
    for exp in experiments:
        exp_dir = f'{base_dir}/{exp}'
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(f'{exp_dir}/checkpoints', exist_ok=True)
        exp_dirs[exp] = exp_dir

    all_group_results = []      # 收集所有实验×组×种子的结果
    group_summaries = []        # 收集每个(实验, 组)的汇总统计

    # 每个实验单独的结果收集器
    per_exp_results = {exp: [] for exp in experiments}
    per_exp_summaries = {exp: [] for exp in experiments}

    # 支持按实验指定种子：构建 per-experiment seeds 映射
    if isinstance(seeds, dict):
        seeds_map = {exp: list(seeds.get(exp, [])) or [10] for exp in experiments}
    else:
        seeds_map = {exp: list(seeds) for exp in experiments}

    # PTB-XL 特定：exp2 使用种子 7，exp3 使用种子 20（覆盖全局设置）
    if dataset == 'ptbxl':
        if 'exp2' in seeds_map:
            seeds_map['exp2'] = [7]
        if 'exp3' in seeds_map:
            seeds_map['exp3'] = [20]

    # 重新计算总运行次数（考虑每个实验可能的不同种子数）
    total_runs = sum(len(seeds_map[exp]) * len(groups) for exp in experiments)
    current_run = 0

    # 记录原始 config 中所有开关的初始值（运行结束后恢复）
    original_config_snapshot = {k: getattr(config, k) for k in ALL_SWITCH_KEYS}
    original_experiment = config.experiment
    original_output_dir = getattr(config, 'output_dir', '.')
    original_datafolder = config.datafolder

    # 设置数据集路径到全局 config
    config.datafolder = datafolder

    print('=' * 70)
    print('  消融实验 (Ablation Study)')
    print('=' * 70)
    print(f'  数据集:     {dataset} ({datafolder})')
    print(f'  实验:       {" / ".join(experiments)}')
    print(f'  消融组:     {" → ".join(groups)}')
    print(f'  随机种子(按实验):')
    for exp in experiments:
        print(f'    {exp}: {seeds_map[exp]}')
    print(f'  每组轮次:   {config.max_epoch} epochs')
    print(f'  总运行次数: {total_runs}  (按实验×组×种子数计算)')
    print(f'  模型:       {config.model_name}')
    print('=' * 70)

    try:
        for exp_idx, experiment in enumerate(experiments, 1):
            print('\n\n' + '╔' + '═' * 68 + '╗')
            print(f'  ▶ [{dataset.upper()}] 实验 [{experiment}]  ({exp_idx}/{len(experiments)})')
            print('╚' + '═' * 68 + '╝')

            for group_name in groups:
                group_cfg = ABLATION_GROUPS[group_name]
                overrides = group_cfg['overrides']

                # 安全的组标识（用于文件名，避免+号等特殊字符）
                group_tag = group_name.replace('+', '')  # 'A', 'AB', 'ABC'

                print('\n' + '█' * 70)
                print(f'  消融组 [{group_name}] | 实验 [{experiment}]')
                print(f'  {group_cfg["description"]}')
                print(f'  启用创新: {group_cfg["modules"]}')
                print('█' * 70)

                # ---- 将覆盖项应用到全局 config ----
                for key, value in overrides.items():
                    setattr(config, key, value)

                # 修改 experiment 名称以区分各消融组的输出文件
                config.experiment = f'{experiment}_abl_{group_tag}'

                # 设置输出目录为当前实验的子文件夹
                config.output_dir = exp_dirs[experiment]

                # 记录当前组的完整配置
                config_key = f'{experiment}/{group_name}'
                ablation_config_record[config_key] = {
                    'experiment': experiment,
                    'description': group_cfg['description'],
                    'modules': group_cfg['modules'],
                    'experiment_tag': config.experiment,
                    'overrides': {k: v for k, v in overrides.items()},
                    'all_switches': {k: getattr(config, k) for k in ALL_SWITCH_KEYS},
                }

                # 打印当前生效的关键开关状态
                print('\n  当前开关状态:')
                for k in ALL_SWITCH_KEYS:
                    v = getattr(config, k)
                    status = '✓ ON ' if v else '✗ OFF'
                    print(f'    {status}  {k}')
                print(f'  实验标识: {config.experiment}')

                group_results = []

                exp_seeds = seeds_map[experiment]
                for i, seed in enumerate(exp_seeds, 1):
                    current_run += 1
                    print('\n' + '-' * 60)
                    print(f'  [{experiment}/{group_name}] Run {i}/{len(exp_seeds)} | seed={seed} '
                          f'| 总进度 {current_run}/{total_runs}')
                    print('-' * 60)

                    config.seed = seed

                    # 调用主训练函数
                    result = train(config)

                    # 附加标识
                    result['ablation_group'] = group_name
                    result['ptbxl_experiment'] = experiment
                    result['group_description'] = group_cfg['description']
                    result['group_tag'] = group_tag
                    group_results.append(result)
                    all_group_results.append(result)
                    per_exp_results[experiment].append(result)

                    # 实时写入当前实验的单独 Excel（防中断丢数据）
                    exp_excel = f'{exp_dirs[experiment]}/ablation_{experiment}.xlsx'
                    _save_results_to_excel(per_exp_results[experiment], per_exp_summaries[experiment], exp_excel)
                    # 同时更新全局汇总
                    _save_results_to_excel(all_group_results, group_summaries, global_excel_path)

                # ---- 计算当前(实验, 组)的汇总统计 ----
                df_group = pd.DataFrame(group_results)
                summary = {
                    'ptbxl_experiment': experiment,
                    'ablation_group': group_name,
                    'description': group_cfg['description'],
                    'modules': group_cfg['modules'],
                    'num_seeds': len(seeds),
                    'auc_mean': df_group['best_test_auc'].mean(),
                    'auc_std': df_group['best_test_auc'].std(),
                    'tpr_mean': df_group['best_test_tpr'].mean(),
                    'tpr_std': df_group['best_test_tpr'].std(),
                    'tpr_at_best_auc_mean': df_group['tpr_at_best_auc'].mean(),
                    'tpr_at_best_auc_std': df_group['tpr_at_best_auc'].std(),
                    'best_single_auc': df_group['best_test_auc'].max(),
                    'best_single_tpr': df_group['best_test_tpr'].max(),
                }
                group_summaries.append(summary)
                per_exp_summaries[experiment].append(summary)

                # 打印当前组汇总
                print('\n' + '=' * 60)
                print(f'  [{experiment}/{group_name}] 组汇总')
                print(f'  AUC:  {summary["auc_mean"]:.6f} ± {summary["auc_std"]:.6f}')
                print(f'  TPR:  {summary["tpr_mean"]:.6f} ± {summary["tpr_std"]:.6f}')
                print(f'  TPR@BestAUC: {summary["tpr_at_best_auc_mean"]:.6f} ± {summary["tpr_at_best_auc_std"]:.6f}')
                print('=' * 60)

                # 每组结束后更新当前实验的 Excel 和全局 Excel
                exp_excel = f'{exp_dirs[experiment]}/ablation_{experiment}.xlsx'
                _save_results_to_excel(per_exp_results[experiment], per_exp_summaries[experiment], exp_excel)
                _save_results_to_excel(all_group_results, group_summaries, global_excel_path)

    finally:
        # ---- 无论是否出错，恢复 config 原始值 ----
        for key, value in original_config_snapshot.items():
            setattr(config, key, value)
        config.experiment = original_experiment
        config.output_dir = original_output_dir
        config.datafolder = original_datafolder

    # ============================================================
    # 最终汇总
    # ============================================================
    print('\n\n' + '█' * 70)
    print('  消融实验最终汇总 (Ablation Study Summary)')
    print('█' * 70)

    df_summary = pd.DataFrame(group_summaries)

    # 按实验分组打印
    for experiment in experiments:
        exp_summaries = [s for s in group_summaries if s['ptbxl_experiment'] == experiment]
        if not exp_summaries:
            continue
        print(f'\n  ── [{dataset.upper()}] [{experiment}] ──')
        print('  ┌────────────┬──────────────────────┬──────────────────────┬──────────────────────┐')
        print('  │ 实验组     │ AUC (mean ± std)     │ TPR (mean ± std)     │ TPR@BestAUC          │')
        print('  ├────────────┼──────────────────────┼──────────────────────┼──────────────────────┤')
        for s in exp_summaries:
            g = s['ablation_group'].ljust(10)
            auc_str = f'{s["auc_mean"]:.4f} ± {s["auc_std"]:.4f}'.ljust(20)
            tpr_str = f'{s["tpr_mean"]:.4f} ± {s["tpr_std"]:.4f}'.ljust(20)
            tba_str = f'{s["tpr_at_best_auc_mean"]:.4f} ± {s["tpr_at_best_auc_std"]:.4f}'.ljust(20)
            print(f'  │ {g} │ {auc_str} │ {tpr_str} │ {tba_str} │')
        print('  └────────────┴──────────────────────┴──────────────────────┴──────────────────────┘')

        # 增量提升分析
        if len(exp_summaries) >= 2:
            print('  增量提升:')
            for i in range(1, len(exp_summaries)):
                prev = exp_summaries[i - 1]
                curr = exp_summaries[i]
                auc_delta = curr['auc_mean'] - prev['auc_mean']
                tpr_delta = curr['tpr_mean'] - prev['tpr_mean']
                arrow = '↑' if auc_delta > 0 else '↓'
                print(f'    {prev["ablation_group"]:>5} → {curr["ablation_group"]:<5}: '
                      f'AUC {arrow}{abs(auc_delta):+.4f}  |  TPR {arrow}{abs(tpr_delta):+.4f}')

    # 保存全局汇总 CSV
    df_summary.to_csv(global_summary_csv, index=False, encoding='utf-8-sig')

    # 保存每个实验单独的汇总 CSV
    for experiment in experiments:
        if per_exp_summaries[experiment]:
            exp_csv = f'{exp_dirs[experiment]}/ablation_{experiment}.csv'
            pd.DataFrame(per_exp_summaries[experiment]).to_csv(exp_csv, index=False, encoding='utf-8-sig')

    # 保存消融配置 JSON
    with open(ablation_config_path, 'w', encoding='utf-8') as f:
        json.dump(ablation_config_record, f, ensure_ascii=False, indent=2)

    print(f'\n  输出文件:')
    print(f'    根目录:      {base_dir}/')
    print(f'    全局Excel:    {global_excel_path}')
    print(f'    全局CSV:     {global_summary_csv}')
    print(f'    配置快照:    {ablation_config_path}')
    print(f'    各实验子文件夹:')
    for experiment in experiments:
        print(f'      {exp_dirs[experiment]}/')
        print(f'        ablation_{experiment}.xlsx / .csv')
        print(f'        checkpoints/ → 各消融组的最优模型')
    
    # 列出各组的训练文件
    print(f'\n  各组训练文件:')
    for experiment in experiments:
        for g in groups:
            tag = g.replace('+', '')
            exp_tag = f'{experiment}_abl_{tag}'
            exp_d = exp_dirs[experiment]
            print(f'    [{experiment}/{g}] CSV:  {exp_d}/{config.model_name}{exp_tag}result.csv')
            print(f'    [{experiment}/{g}] Ckpt: {exp_d}/checkpoints/{config.model_name}_{exp_tag}_checkpoint_best.pth')

    print('\n  消融实验全部完成！')
    
    return group_summaries


def _save_results_to_excel(all_results, group_summaries, excel_path):
    """将当前所有结果实时写入 Excel（双Sheet: 详细结果 + 消融汇总）"""
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: 所有单次运行的详细结果
            df_detail = pd.DataFrame(all_results)
            col_order = ['ablation_group', 'seed', 'best_test_auc', 'tpr_at_best_auc',
                         'best_auc_epoch', 'best_test_tpr', 'auc_at_best_tpr',
                         'best_tpr_epoch', 'experiment', 'group_description']
            existing_cols = [c for c in col_order if c in df_detail.columns]
            df_detail = df_detail[existing_cols]
            df_detail.to_excel(writer, sheet_name='详细结果', index=False)

            # Sheet 2: 每组的汇总统计
            if group_summaries:
                df_summary = pd.DataFrame(group_summaries)
                df_summary.to_excel(writer, sheet_name='消融汇总', index=False)
    except Exception as e:
        print(f'  [Warning] Excel写入失败: {e}')


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='ECG多标签分类 — 消融实验一键运行脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_ablation.py                                    # 默认: 全部6个PTB-XL实验 × 3消融组
  python run_ablation.py --dataset hf                       # 一键跑HF数据集消融实验
  python run_ablation.py --dataset hf --seeds 9 10 42       # HF数据集 + 多种子
  python run_ablation.py --experiments exp0 exp1            # 只跑exp0和exp1
  python run_ablation.py --seeds 10 42                      # 多种子
  python run_ablation.py --groups A A+B+C                   # 只跑A和完整模型 (跳过A+B)
        """)
    parser.add_argument('--dataset', type=str, default='ptbxl',
                        choices=['ptbxl', 'cpsc', 'hf'],
                        help='数据集名称 (默认: ptbxl)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[10],
                        help='随机种子列表 (默认: 10)')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='实验编号列表 (默认: 根据数据集自动选择; ptbxl=exp0~exp3, hf=hf)')
    parser.add_argument('--groups', type=str, nargs='+', default=['A', 'A+B', 'A+B+C'],
                        choices=['A', 'A+B', 'A+B+C'],
                        help='要运行的消融组 (默认: A A+B A+B+C)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_ablation(
        seeds=args.seeds,
        experiments=args.experiments,
        groups=args.groups,
        dataset=args.dataset,
    )
