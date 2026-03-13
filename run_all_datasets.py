# -*- coding: utf-8 -*-
"""
一键跑完所有数据集（PTB-XL、CPSC、HF）消融实验脚本
================================================================

自动依次对三个数据集运行三组递进式消融实验（A → A+B → A+B+C）

数据集说明：
┌──────────┬────────────────────────────────────────────────────┐
│ 数据集   │ 说明                                               │
├──────────┼────────────────────────────────────────────────────┤
│ PTB-XL   │ 大规模公开ECG数据集，6个实验                       │
│          │  exp0(all) / exp1(diagnostic) / exp1.1(subdiag.)   │
│          │  exp1.1.1(superdiag.) / exp2(form) / exp3(rhythm)  │
├──────────┼────────────────────────────────────────────────────┤
│ CPSC     │ 中国心电学会数据集                                 │
│          │  使用 TrainingSet1 + TrainingSet2 + TrainingSet3    │
│          │  实验标识: cpsc                                     │
├──────────┼────────────────────────────────────────────────────┤
│ HF       │ 心力衰竭相关ECG数据集                              │
│          │  实验标识: hf                                       │
└──────────┴────────────────────────────────────────────────────┘

运行方式:
    python run_all_datasets.py                              # 默认: 三个数据集 × 三组消融 × 单种子
    python run_all_datasets.py --seeds 10 42                # 多种子运行
    python run_all_datasets.py --groups A A+B+C             # 只跑 A 和完整模型
    python run_all_datasets.py --datasets ptbxl cpsc        # 只跑 PTB-XL 和 CPSC
    python run_all_datasets.py --no-ablation                # 不做消融，直接用当前 config 跑全部数据集
    python run_all_datasets.py --skip-convert               # 跳过 CPSC 数据转换步骤

输出目录:
    result/all_datasets_YYYYMMDD_HHMMSS/
        ptbxl/          → PTB-XL 消融结果
        cpsc/           → CPSC 消融结果
        hf/             → HF 消融结果
        global_summary.xlsx   → 三个数据集汇总表
"""

import os
import sys
import json
import argparse
import time
import pandas as pd
import numpy as np
from datetime import datetime


# ============================================================
# 数据集配置
# ============================================================

# 各数据集默认种子（PTB-XL 的 exp2/exp3 会在 run_ablation 内部自动覆盖）
DATASET_DEFAULT_SEEDS = {
    'ptbxl': [10],
    'cpsc': [10],
    'hf': [10],
}

# 数据集顺序
ALL_DATASETS = ['ptbxl', 'cpsc', 'hf']

# 数据集路径
DATASET_PATHS = {
    'ptbxl': 'data/ptbxl/',
    'cpsc': 'data/CPSC/',
    'hf': 'data/hf/',
}

# 数据集中文名
DATASET_NAMES = {
    'ptbxl': 'PTB-XL',
    'cpsc': 'CPSC (TrainingSet1+2+3)',
    'hf': 'HF (心力衰竭)',
}


def convert_cpsc_if_needed():
    """
    检查 CPSC 数据是否已转换（TrainingSet1+2+3 → WFDB格式）。
    如果 cpsc_database.csv 不存在或 records100/ 为空则自动运行转换。
    """
    cpsc_path = 'data/CPSC/'
    csv_path = os.path.join(cpsc_path, 'cpsc_database.csv')
    records_path = os.path.join(cpsc_path, 'records100')

    needs_convert = False
    if not os.path.exists(csv_path):
        print('  [CPSC] cpsc_database.csv 不存在，需要转换数据...')
        needs_convert = True
    elif not os.path.exists(records_path) or len(os.listdir(records_path)) == 0:
        print('  [CPSC] records100/ 为空或不存在，需要转换数据...')
        needs_convert = True

    if needs_convert:
        print('  [CPSC] 正在运行 convert_cpsc.py (处理 TrainingSet1 + TrainingSet2 + TrainingSet3)...')
        # 清除旧的缓存文件（如果存在），因为数据可能发生变化
        for cache_file in ['raw100.npy', 'raw500.npy']:
            cache_path = os.path.join(cpsc_path, cache_file)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f'  [CPSC] 已删除旧缓存: {cache_path}')

        import convert_cpsc
        print('  [CPSC] 数据转换完成！')
    else:
        print('  [CPSC] 数据已存在，跳过转换步骤')


def run_with_ablation(datasets, seeds, groups, output_root):
    """
    使用消融实验框架，依次运行各数据集

    Args:
        datasets: 要运行的数据集列表
        seeds: 随机种子列表
        groups: 消融组列表
        output_root: 输出根目录
    """
    from run_ablation import run_ablation

    all_summaries = {}  # {dataset_name: [summary_dicts]}

    for ds_idx, dataset in enumerate(datasets, 1):
        print('\n\n' + '╔' + '═' * 68 + '╗')
        print(f'  ▶ 数据集 [{ds_idx}/{len(datasets)}]: {DATASET_NAMES.get(dataset, dataset)}')
        print(f'    路径: {DATASET_PATHS[dataset]}')
        print('╚' + '═' * 68 + '╝')

        ds_start = time.time()

        try:
            # 使用该数据集的默认种子（如果用户未指定）
            ds_seeds = seeds if seeds else DATASET_DEFAULT_SEEDS.get(dataset, [10])

            summaries = run_ablation(
                seeds=ds_seeds,
                experiments=None,  # 自动根据数据集选择实验列表
                groups=groups,
                dataset=dataset,
            )
            all_summaries[dataset] = summaries

        except Exception as e:
            print(f'\n  [ERROR] 数据集 {dataset} 运行失败: {e}')
            import traceback
            traceback.print_exc()
            all_summaries[dataset] = []
            continue

        ds_elapsed = time.time() - ds_start
        print(f'\n  [TIME] {DATASET_NAMES.get(dataset, dataset)} 总耗时: '
              f'{int(ds_elapsed // 3600)}h {int((ds_elapsed % 3600) // 60)}m {int(ds_elapsed % 60)}s')

    return all_summaries


def run_without_ablation(datasets, seeds, output_root):
    """
    不做消融，直接使用当前 config 配置，依次对各数据集训练

    Args:
        datasets: 要运行的数据集列表
        seeds: 随机种子列表
        output_root: 输出根目录
    """
    from config import config
    from main_train import train

    all_results = []
    original_datafolder = config.datafolder
    original_experiment = config.experiment
    original_seed = config.seed
    original_output_dir = getattr(config, 'output_dir', '.')

    try:
        for ds_idx, dataset in enumerate(datasets, 1):
            print('\n\n' + '╔' + '═' * 68 + '╗')
            print(f'  ▶ 数据集 [{ds_idx}/{len(datasets)}]: {DATASET_NAMES.get(dataset, dataset)}')
            print(f'    路径: {DATASET_PATHS[dataset]}')
            print(f'    模型: {config.model_name}')
            print('╚' + '═' * 68 + '╝')

            ds_start = time.time()

            # 设置数据集路径
            config.datafolder = DATASET_PATHS[dataset]

            # 根据数据集确定实验列表
            if dataset == 'ptbxl':
                experiments = ['exp0', 'exp1', 'exp1.1', 'exp1.1.1', 'exp2', 'exp3']
            elif dataset == 'cpsc':
                experiments = ['cpsc']
            elif dataset == 'hf':
                experiments = ['hf']
            else:
                experiments = [dataset]

            # 每个数据集的输出目录
            ds_output = os.path.join(output_root, dataset)
            os.makedirs(ds_output, exist_ok=True)
            os.makedirs(os.path.join(ds_output, 'checkpoints'), exist_ok=True)
            config.output_dir = ds_output

            ds_seeds = seeds if seeds else DATASET_DEFAULT_SEEDS.get(dataset, [10])

            for exp in experiments:
                for seed in ds_seeds:
                    print('\n' + '-' * 60)
                    print(f'  [{dataset}/{exp}] seed={seed}')
                    print('-' * 60)

                    config.experiment = exp
                    config.seed = seed

                    try:
                        result = train(config)
                        result['dataset'] = dataset
                        result['dataset_name'] = DATASET_NAMES.get(dataset, dataset)
                        all_results.append(result)
                    except Exception as e:
                        print(f'  [ERROR] {dataset}/{exp}/seed={seed} 失败: {e}')
                        import traceback
                        traceback.print_exc()
                        continue

            ds_elapsed = time.time() - ds_start
            print(f'\n  [TIME] {DATASET_NAMES.get(dataset, dataset)} 总耗时: '
                  f'{int(ds_elapsed // 3600)}h {int((ds_elapsed % 3600) // 60)}m {int(ds_elapsed % 60)}s')

    finally:
        # 恢复原始配置
        config.datafolder = original_datafolder
        config.experiment = original_experiment
        config.seed = original_seed
        config.output_dir = original_output_dir

    return all_results


def save_global_summary(all_summaries, output_root):
    """
    保存跨数据集的全局汇总表

    Args:
        all_summaries: {dataset: [summary_dicts]} 或 [result_dicts]
        output_root: 输出根目录
    """
    try:
        if isinstance(all_summaries, dict):
            # 消融模式：all_summaries = {dataset: [summaries]}
            rows = []
            for dataset, summaries in all_summaries.items():
                for s in summaries:
                    row = dict(s)
                    row['dataset'] = dataset
                    row['dataset_name'] = DATASET_NAMES.get(dataset, dataset)
                    rows.append(row)
            if rows:
                df = pd.DataFrame(rows)
                excel_path = os.path.join(output_root, 'global_summary.xlsx')
                csv_path = os.path.join(output_root, 'global_summary.csv')
                df.to_excel(excel_path, index=False, sheet_name='全部数据集汇总')
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f'\n  全局汇总已保存:')
                print(f'    Excel: {excel_path}')
                print(f'    CSV:   {csv_path}')
        else:
            # 非消融模式
            if all_summaries:
                df = pd.DataFrame(all_summaries)
                excel_path = os.path.join(output_root, 'global_summary.xlsx')
                csv_path = os.path.join(output_root, 'global_summary.csv')
                df.to_excel(excel_path, index=False, sheet_name='全部数据集汇总')
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f'\n  全局汇总已保存:')
                print(f'    Excel: {excel_path}')
                print(f'    CSV:   {csv_path}')
    except Exception as e:
        print(f'  [Warning] 全局汇总保存失败: {e}')


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='ECG多标签分类 — 一键跑完所有数据集（PTB-XL、CPSC、HF）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_all_datasets.py                              # 默认: 三个数据集 × 三组消融
  python run_all_datasets.py --seeds 10 42                # 多种子运行
  python run_all_datasets.py --groups A A+B+C             # 只跑 A 和完整模型
  python run_all_datasets.py --datasets ptbxl cpsc        # 只跑 PTB-XL 和 CPSC
  python run_all_datasets.py --no-ablation                # 不做消融，直接训练
  python run_all_datasets.py --skip-convert               # 跳过 CPSC 数据转换步骤
        """)
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['ptbxl', 'cpsc', 'hf'],
                        choices=['ptbxl', 'cpsc', 'hf'],
                        help='要运行的数据集列表 (默认: ptbxl cpsc hf)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='随机种子列表 (默认: 各数据集使用各自默认种子)')
    parser.add_argument('--groups', type=str, nargs='+',
                        default=['A', 'A+B', 'A+B+C'],
                        choices=['A', 'A+B', 'A+B+C'],
                        help='要运行的消融组 (默认: A A+B A+B+C)')
    parser.add_argument('--no-ablation', action='store_true',
                        help='不做消融实验，直接使用当前config配置训练')
    parser.add_argument('--skip-convert', action='store_true',
                        help='跳过 CPSC 数据转换步骤 (数据已转换好时使用)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    total_start = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('=' * 70)
    print('  ECG多标签分类 — 全数据集一键运行')
    print('=' * 70)
    print(f'  数据集:   {" / ".join(args.datasets)}')
    if not args.no_ablation:
        print(f'  消融组:   {" → ".join(args.groups)}')
    else:
        print(f'  模式:     直接训练 (不做消融)')
    print(f'  种子:     {args.seeds if args.seeds else "各数据集默认"}')
    print(f'  时间戳:   {timestamp}')
    print('=' * 70)

    # CPSC 数据转换检查
    if 'cpsc' in args.datasets and not args.skip_convert:
        print('\n[Step 0] 检查 CPSC 数据转换状态...')
        convert_cpsc_if_needed()

    # 创建输出根目录
    output_root = f'result/all_datasets_{timestamp}'
    os.makedirs(output_root, exist_ok=True)

    # 保存运行配置
    run_config = {
        'datasets': args.datasets,
        'seeds': args.seeds,
        'groups': args.groups if not args.no_ablation else None,
        'no_ablation': args.no_ablation,
        'timestamp': timestamp,
    }
    with open(os.path.join(output_root, 'run_config.json'), 'w', encoding='utf-8') as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    # 运行
    if args.no_ablation:
        results = run_without_ablation(args.datasets, args.seeds, output_root)
        save_global_summary(results, output_root)
    else:
        summaries = run_with_ablation(args.datasets, args.seeds, args.groups, output_root)
        save_global_summary(summaries, output_root)

    # 总耗时
    total_elapsed = time.time() - total_start
    print('\n\n' + '█' * 70)
    print('  全部数据集运行完成!')
    print('█' * 70)
    print(f'  总耗时: {int(total_elapsed // 3600)}h {int((total_elapsed % 3600) // 60)}m {int(total_elapsed % 60)}s')
    print(f'  输出目录: {output_root}/')
    for ds in args.datasets:
        print(f'    {ds}: result/ablation_{ds}_*/')
    print(f'  全局汇总: {output_root}/global_summary.xlsx')
    print('█' * 70)
