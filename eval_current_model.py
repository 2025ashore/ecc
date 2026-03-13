# -*- coding: utf-8 -*-
"""
独立评估脚本：加载当前模型和权重，对指定数据集进行测试

功能：
- 读取 `config.py` 的默认设置，可通过命令行覆盖 datafolder/experiment/checkpoint/batch-size
- 加载 `models` 与权重文件，使用 `dataset.load_datasets()` 获取数据
- 在选择的划分（Test/Val）上推理，计算并打印 TPR（samplewise/micro/macro）与 AUC（micro/macro）

用法示例（PowerShell）：
  python "eval_current_model.py" --phase test
  python "eval_current_model.py" --datafolder data/ptbxl/ --experiment exp0 --checkpoint checkpoints/MyNet7ViewTimeFreq_exp0_checkpoint_best.pth --phase val
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
rcParams['axes.unicode_minus'] = False

from config import config
from dataset import load_datasets
import models
import utils


def build_model(num_classes: int, device: torch.device):
    """根据 config 构建模型（含标签图细化设置）。"""
    if config.model_name == 'MyNet6View':
        label_kwargs = dict(
            use_label_graph_refiner=config.use_label_graph_refiner,
            label_graph_hidden=config.label_graph_hidden,
            label_graph_learnable_adj=config.label_graph_learnable_adj,
            label_graph_dropout=config.label_graph_dropout,
        )
        model = getattr(models, config.model_name)(num_classes=num_classes, **label_kwargs)
    else:
        model = getattr(models, config.model_name)(num_classes=num_classes)
    return model.to(device)


def resolve_checkpoint_path(arg_checkpoint: str | None) -> Path | None:
    """解析权重路径：优先命令行，其次 config.checkpoints，再次约定命名。"""
    ck_dir = Path('checkpoints')
    # 1) 命令行指定
    if arg_checkpoint:
        p = Path(arg_checkpoint)
        if p.exists():
            return p
        p_dir = ck_dir / arg_checkpoint
        if p_dir.exists():
            return p_dir
    # 2) config.checkpoints
    if hasattr(config, 'checkpoints') and isinstance(config.checkpoints, str):
        p = ck_dir / config.checkpoints
        if p.exists():
            return p
        p_root = Path(config.checkpoints)
        if p_root.exists():
            return p_root
    # 3) 约定命名
    candidate = ck_dir / f"{config.model_name}_{config.experiment}_checkpoint_best.pth"
    if candidate.exists():
        return candidate
    return None


def load_checkpoint(model: torch.nn.Module, ck_path: Path, device: torch.device) -> bool:
    try:
        ck = torch.load(str(ck_path), map_location=device)
        state = ck.get('model_state_dict', ck)
        model.load_state_dict(state, strict=False)
        print(f"✓ 已加载权重: {ck_path}")
        return True
    except Exception as e:
        print(f"⚠️ 加载权重失败: {e}")
        return False


def eval_loader(model: torch.nn.Module, loader, device: torch.device):
    """在给定 loader 上推理并返回 (y_true, y_score)。"""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            ys.append(y.numpy())
            ps.append(probs)
    y_true = np.vstack(ys)
    y_score = np.vstack(ps)
    return y_true, y_score


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5):
    """计算核心指标：TPR（samplewise/micro/macro）与 AUC（micro/macro）。"""
    metrics = {}
    # AUC
    metrics['roc_auc_macro'] = roc_auc_score(y_true, y_score, average='macro')
    metrics['roc_auc_micro'] = roc_auc_score(y_true, y_score, average='micro')
    # TPR
    try:
        metrics['tpr_samplewise'] = utils.compute_TPR(y_true, y_score)
    except Exception:
        metrics['tpr_samplewise'] = None
    y_pred = (y_score >= threshold).astype(int)
    metrics['tpr_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['tpr_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    return metrics


def per_class_metrics(y_true: np.ndarray, y_score: np.ndarray, prevalence: np.ndarray, threshold: float = 0.5,
                      sort_by: str | None = None):
    """计算逐类 AUC、TPR 与占比，返回 DataFrame，并按需排序。"""
    num_classes = y_true.shape[1]
    auc_list = []
    tpr_list = []
    prev_list = prevalence.astype(float)

    y_pred = (y_score >= threshold).astype(int)
    for c in range(num_classes):
        yt = y_true[:, c]
        ys = y_score[:, c]
        yp = y_pred[:, c]
        # AUC：仅在存在正负样本时计算
        if yt.sum() > 0 and (len(yt) - yt.sum()) > 0:
            try:
                auc_c = roc_auc_score(yt, ys)
            except Exception:
                auc_c = np.nan
        else:
            auc_c = np.nan
        # TPR：逐类召回
        try:
            tpr_c = recall_score(yt, yp, average='binary', zero_division=0)
        except Exception:
            tpr_c = np.nan
        auc_list.append(auc_c)
        tpr_list.append(tpr_c)

    df = pd.DataFrame({
        'class_index': np.arange(num_classes),
        'prevalence': prev_list,
        'auc': auc_list,
        'tpr': tpr_list,
    })

    if sort_by in {'prevalence', 'auc', 'tpr'}:
        df = df.sort_values(by=sort_by, ascending=False, na_position='last').reset_index(drop=True)
    return df


#############################
# 绘图辅助
#############################

def plot_prevalence_curve(df: pd.DataFrame, out_dir: Path, phase: str):
    prev_sorted = np.sort(df['prevalence'].values.astype(float))
    plt.figure(figsize=(8, 4))
    plt.plot(prev_sorted, marker='.', linewidth=1)
    plt.title(f'{phase.upper()} 类别占比曲线 (升序)')
    plt.xlabel('类别按占比升序')
    plt.ylabel('阳性占比')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f'prevalence_curve_{phase}.png', dpi=200)
    plt.close()


def plot_auc_vs_prevalence(df: pd.DataFrame, out_dir: Path, phase: str):
    valid = df.dropna(subset=['auc'])
    if valid.empty:
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(valid['prevalence'], valid['auc'], c=valid['auc'], cmap='viridis', s=24)
    plt.colorbar(label='AUC')
    plt.xlabel('类别阳性占比')
    plt.ylabel('单类AUC')
    plt.title(f'{phase.upper()} 单类AUC vs 占比')
    # 趋势线
    try:
        coeffs = np.polyfit(valid['prevalence'], valid['auc'], deg=1)
        xs = np.linspace(valid['prevalence'].min(), valid['prevalence'].max(), 100)
        ys = coeffs[0] * xs + coeffs[1]
        plt.plot(xs, ys, color='#D62728', linewidth=2, label=f'slope={coeffs[0]:.3f}')
        plt.legend()
    except Exception:
        pass
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f'auc_vs_prevalence_{phase}.png', dpi=200)
    plt.close()


def plot_topk_bars(df: pd.DataFrame, out_dir: Path, phase: str, key: str, k: int = 20):
    if key not in df.columns:
        return
    dfk = df.sort_values(by=key, ascending=False, na_position='last').head(k)
    plt.figure(figsize=(8, 6))
    plt.barh(dfk['class_index'].astype(str), dfk[key], color='#4C78A8')
    plt.gca().invert_yaxis()
    plt.xlabel(key)
    plt.title(f'{phase.upper()} Top{len(dfk)} {key}')
    plt.tight_layout()
    plt.savefig(out_dir / f'top{len(dfk)}_{key}_{phase}.png', dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='评估当前模型在指定数据划分上的表现')
    parser.add_argument('--datafolder', type=str, default=None, help='数据集根路径，默认使用config.datafolder')
    parser.add_argument('--experiment', type=str, default=None, help='PTB-XL实验编号（exp0/exp1/...），默认使用config.experiment')
    parser.add_argument('--checkpoint', type=str, default=None, help='权重文件路径或文件名，默认使用config.checkpoints')
    parser.add_argument('--batch-size', type=int, default=None, help='评估批大小，默认使用config.batch_size')
    parser.add_argument('--phase', type=str, default='test', choices=['test', 'val'], help='评估划分')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值（用于TPR等）')
    parser.add_argument('--sort-by', type=str, default='prevalence', choices=['prevalence', 'auc', 'tpr', 'none'],
                        help='逐类结果排序依据')
    parser.add_argument('--save-csv', action='store_true', help='将逐类结果保存为CSV')
    parser.add_argument('--save-plots', action='store_true', help='保存可视化图像')
    parser.add_argument('--out-dir', type=str, default=None, help='逐类CSV输出目录（默认当前目录）')
    args = parser.parse_args()

    # 覆盖配置
    if args.datafolder:
        config.datafolder = args.datafolder
    if args.experiment:
        config.experiment = args.experiment
    if args.batch_size:
        config.batch_size = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"数据集: {config.datafolder}, 实验: {config.experiment}, 批大小: {config.batch_size}")

    # 加载数据
    train_loader, val_loader, test_loader, num_classes = load_datasets(
        datafolder=config.datafolder,
        experiment=config.experiment,
    )
    loader = test_loader if args.phase == 'test' else val_loader

    # 构建并加载模型
    model = build_model(num_classes, device)
    ck_path = resolve_checkpoint_path(args.checkpoint)
    if ck_path is None:
        print('⚠️ 未找到权重文件，请通过 --checkpoint 指定或在 config.checkpoints 中设置。')
        return
    ok = load_checkpoint(model, ck_path, device)
    if not ok:
        return

    # 推理与指标
    print(f"\n在{args.phase.upper()}集上推理并评估...")
    y_true, y_score = eval_loader(model, loader, device)
    metrics = compute_metrics(y_true, y_score, threshold=args.threshold)

    print("\n指标（TPR 与 AUC）：")
    if metrics['tpr_samplewise'] is not None:
        print(f"  tpr_samplewise : {metrics['tpr_samplewise']:.6f}")
    print(f"  tpr_macro      : {metrics['tpr_macro']:.6f}")
    print(f"  tpr_micro      : {metrics['tpr_micro']:.6f}")
    print(f"  roc_auc_macro  : {metrics['roc_auc_macro']:.6f}")
    print(f"  roc_auc_micro  : {metrics['roc_auc_micro']:.6f}")

    # 逐类占比（该划分）
    ds = loader.dataset
    prevalence = ds.cls_num_list / float(len(ds))
    sort_key = None if args.sort_by == 'none' else args.sort_by
    df_cls = per_class_metrics(y_true, y_score, prevalence, threshold=args.threshold, sort_by=sort_key)

    # 打印前20行供快速查看
    print("\n逐类占比 + AUC + TPR（前20类，排序=", args.sort_by, ")：")
    print(df_cls.head(20).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # 可选保存CSV
    out_dir = Path(args.out_dir) if args.out_dir else Path('.')
    if args.save_csv:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"per_class_metrics_{args.phase}.csv"
        try:
            df_cls.to_csv(out_path, index=False)
            print(f"\n已保存逐类结果到 {out_path.resolve()}")
        except PermissionError:
            alt_path = out_dir / f"per_class_metrics_{args.phase}_tmp.csv"
            df_cls.to_csv(alt_path, index=False)
            print(f"\n⚠️ 目标文件可能被占用或无写权限，已改存为 {alt_path.resolve()}")

    if args.save_plots:
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_prevalence_curve(df_cls, out_dir, args.phase)
        plot_auc_vs_prevalence(df_cls, out_dir, args.phase)
        plot_topk_bars(df_cls, out_dir, args.phase, key='auc', k=20)
        plot_topk_bars(df_cls, out_dir, args.phase, key='prevalence', k=20)
        print(f"\n已保存可视化到 {out_dir.resolve()} (曲线/散点/Top20条形)")


if __name__ == '__main__':
    main()
