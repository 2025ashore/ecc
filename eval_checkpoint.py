# -*- coding: utf-8 -*-
"""eval_checkpoint.py

加载训练保存的 checkpoint，并在测试集上评估 multi-label 的 AUC 与 TPR。

指标口径默认与 main_train.py 保持一致：
- AUC: sklearn.metrics.roc_auc_score(y_true, y_score)（多标签宏平均）
- TPR: utils.compute_TPR（阈值0.5，逐样本计算混淆矩阵后平均）

用法示例：
  python eval_checkpoint.py --checkpoint checkpoints/MyNet7ViewTimeFreq_exp0_checkpoint_best.pth
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import models
import utils
from config import config
from dataset import load_datasets


def _resolve_checkpoint_path(path: str) -> str:
    """兼容 config 里只写文件名的情况。"""
    if os.path.isfile(path):
        return path

    candidate = os.path.join("checkpoints", path)
    if os.path.isfile(candidate):
        return candidate

    raise FileNotFoundError(
        f"找不到 checkpoint: '{path}'（也未在 '{candidate}' 找到）"
    )


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _load_checkpoint_flexible(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[Dict[str, Any], str]:
    """支持多种 checkpoint 字段命名。返回 (checkpoint_dict, used_key)。"""
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = _strip_module_prefix(ckpt[key])
                model.load_state_dict(state, strict=True)
                return ckpt, key

        # 也可能 ckpt 本身就是 state_dict
        if all(isinstance(k, str) for k in ckpt.keys()):
            state = _strip_module_prefix(ckpt)  # type: ignore[arg-type]
            model.load_state_dict(state, strict=True)
            return ckpt, "<root>"

    raise ValueError(
        "不支持的checkpoint格式：期望 dict，且包含 'model_state_dict' 或 'state_dict' 等字段"
    )


def _build_model(num_classes: int) -> torch.nn.Module:
    label_kwargs = dict(
        use_label_graph_refiner=getattr(config, "use_label_graph_refiner", False),
        label_graph_hidden=getattr(config, "label_graph_hidden", 64),
        label_graph_learnable_adj=getattr(config, "label_graph_learnable_adj", True),
        label_graph_dropout=getattr(config, "label_graph_dropout", 0.1),
    )

    if config.model_name == "MyNet7ViewTimeFreq":
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
        return getattr(models, config.model_name)(num_classes=num_classes, **mynet7_kwargs)

    return getattr(models, config.model_name)(num_classes=num_classes)


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader, device: torch.device) -> Dict[str, Any]:
    model.eval()

    probs_list = []
    targets_list = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        probs = torch.sigmoid(logits)

        probs_list.append(probs.detach().cpu().numpy())
        targets_list.append(targets.detach().cpu().numpy())

    y_score = np.concatenate(probs_list, axis=0)
    y_true = np.concatenate(targets_list, axis=0)

    # AUC：macro & micro（若某些类只有单一标签，macro可能报错；做一个更稳健的退化处理）
    auc_macro = None
    auc_micro = None
    per_class_auc = []

    try:
        auc_macro = float(roc_auc_score(y_true, y_score, average="macro"))
    except Exception:
        auc_macro = None

    try:
        auc_micro = float(roc_auc_score(y_true, y_score, average="micro"))
    except Exception:
        auc_micro = None

    for c in range(y_true.shape[1]):
        try:
            per_class_auc.append(float(roc_auc_score(y_true[:, c], y_score[:, c])))
        except Exception:
            per_class_auc.append(float("nan"))

    # TPR：复用工程内实现（阈值0.5，逐样本）
    tpr = float(utils.compute_TPR(y_true, y_score))

    return {
        "num_samples": int(y_true.shape[0]),
        "num_classes": int(y_true.shape[1]),
        "auc_macro": auc_macro,
        "auc_micro": auc_micro,
        "tpr": tpr,
        "per_class_auc": per_class_auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on test set (AUC/TPR).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=getattr(config, "checkpoints", ""),
        help="checkpoint路径（可传文件名，会自动去checkpoints/下寻找）",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=getattr(config, "model_name", ""),
        help="覆盖 config.model_name（可选）",
    )
    parser.add_argument(
        "--datafolder",
        type=str,
        default=getattr(config, "datafolder", "data/ptbxl/"),
        help="数据集根目录（默认读取config.datafolder）",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=getattr(config, "experiment", "exp0"),
        help="实验编号（PTB-XL用，如exp0/exp1...；默认config.experiment）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=getattr(config, "batch_size", 128),
        help="覆盖 config.batch_size（可选）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="运行设备",
    )
    args = parser.parse_args()

    if not args.checkpoint:
        raise ValueError("未提供 --checkpoint，且 config.checkpoints 为空")

    # 运行时覆盖配置（尽量复用现有训练/数据加载代码）
    config.model_name = args.model_name
    config.datafolder = args.datafolder
    config.experiment = args.experiment
    config.batch_size = args.batch_size

    device = torch.device(args.device)

    ckpt_path = _resolve_checkpoint_path(args.checkpoint)

    # 加载数据
    _, _, test_loader, num_classes = load_datasets(datafolder=config.datafolder, experiment=config.experiment)

    # 构建模型并加载权重
    model = _build_model(num_classes=num_classes).to(device)
    ckpt, used_key = _load_checkpoint_flexible(model, ckpt_path, device)

    metrics = evaluate(model, test_loader, device)

    # 打印结果
    print("=" * 80)
    print("Checkpoint evaluation")
    print("=" * 80)
    print(f"model_name      : {config.model_name}")
    print(f"datafolder      : {config.datafolder}")
    print(f"experiment      : {config.experiment}")
    print(f"checkpoint      : {ckpt_path}")
    print(f"state_dict_key  : {used_key}")

    if isinstance(ckpt, dict) and "global_epoch" in ckpt:
        print(f"global_epoch    : {ckpt['global_epoch']}")
    if isinstance(ckpt, dict) and "best_auc" in ckpt:
        print(f"ckpt_best_auc   : {ckpt['best_auc']}")

    print("-" * 80)
    print(f"num_samples     : {metrics['num_samples']}")
    print(f"num_classes     : {metrics['num_classes']}")
    print(f"AUC(macro)      : {metrics['auc_macro']}")
    print(f"AUC(micro)      : {metrics['auc_micro']}")
    print(f"TPR(th=0.5)     : {metrics['tpr']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
