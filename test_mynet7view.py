# -*- coding: utf-8 -*-
"""
@time: 2026/1/5
@description: MyNet7ViewTimeFreq 模型验证脚本
    对比6视图 vs 7视图的性能和计算开销
"""

import torch
import torch.nn as nn
import time
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models import MyNet6View, MyNet7ViewTimeFreq


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """计算可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_comparison():
    """对比6视图和7视图模型"""
    print("=" * 80)
    print("[Test 1] 模型规模对比")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")
    
    # 创建两个模型
    model_6view = MyNet6View(num_classes=10)
    model_7view = MyNet7ViewTimeFreq(num_classes=10)
    
    model_6view = model_6view.to(device)
    model_7view = model_7view.to(device)
    
    # 比较参数量
    params_6view = count_parameters(model_6view)
    params_7view = count_parameters(model_7view)
    param_increase = (params_7view - params_6view) / params_6view * 100
    
    print(f"MyNet6View:")
    print(f"  总参数数: {params_6view:,}")
    print(f"  可训练参数: {count_trainable_parameters(model_6view):,}")
    
    print(f"\nMyNet7ViewTimeFreq:")
    print(f"  总参数数: {params_7view:,}")
    print(f"  可训练参数: {count_trainable_parameters(model_7view):,}")
    print(f"  参数增加: {param_increase:.1f}%\n")
    
    return params_6view, params_7view


def test_inference_speed():
    """测试推理速度"""
    print("=" * 80)
    print("[Test 2] 推理速度对比")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_6view = MyNet6View(num_classes=10).to(device)
    model_7view = MyNet7ViewTimeFreq(num_classes=10).to(device)
    
    # 预热
    x_dummy = torch.randn(1, 12, 5000).to(device)
    with torch.no_grad():
        _ = model_6view(x_dummy)
        _ = model_7view(x_dummy)
    
    # 测试不同批量大小
    batch_sizes = [1, 4, 8, 16]
    
    for bs in batch_sizes:
        x = torch.randn(bs, 12, 5000).to(device)
        
        # 测试6视图
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model_6view(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_6view = (time.time() - start) / 10 * 1000  # ms
        
        # 测试7视图
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model_7view(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_7view = (time.time() - start) / 10 * 1000  # ms
        
        overhead = (time_7view / time_6view - 1) * 100
        
        print(f"批量大小 {bs:2d}:")
        print(f"  6视图: {time_6view:6.2f}ms")
        print(f"  7视图: {time_7view:6.2f}ms")
        print(f"  开销: {overhead:+.1f}%")


def test_memory_usage():
    """测试显存占用"""
    print("\n" + "=" * 80)
    print("[Test 3] 显存占用对比")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("GPU不可用，跳过此测试\n")
        return
    
    device = torch.device("cuda")
    
    model_6view = MyNet6View(num_classes=10).to(device)
    model_7view = MyNet7ViewTimeFreq(num_classes=10).to(device)
    
    batch_sizes = [1, 4, 8]
    
    for bs in batch_sizes:
        # 测试6视图
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        x = torch.randn(bs, 12, 5000, device=device)
        with torch.no_grad():
            _ = model_6view(x)
        
        peak_6view = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
        current_6view = torch.cuda.memory_allocated(device) / 1024**2
        
        # 测试7视图
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        x = torch.randn(bs, 12, 5000, device=device)
        with torch.no_grad():
            _ = model_7view(x)
        
        peak_7view = torch.cuda.max_memory_allocated(device) / 1024**2
        current_7view = torch.cuda.memory_allocated(device) / 1024**2
        
        print(f"批量大小 {bs}:")
        print(f"  6视图 - 当前: {current_6view:6.1f}MB, 峰值: {peak_6view:6.1f}MB")
        print(f"  7视图 - 当前: {current_7view:6.1f}MB, 峰值: {peak_7view:6.1f}MB")
        print(f"  增幅 - 当前: {(current_7view/current_6view-1)*100:+.1f}%, 峰值: {(peak_7view/peak_6view-1)*100:+.1f}%")


def test_forward_pass():
    """测试前向传播"""
    print("\n" + "=" * 80)
    print("[Test 4] 前向传播测试")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_6view = MyNet6View(num_classes=10).to(device)
    model_7view = MyNet7ViewTimeFreq(num_classes=10).to(device)
    
    x = torch.randn(2, 12, 5000).to(device)
    
    try:
        with torch.no_grad():
            out_6view = model_6view(x)
        print(f"✓ MyNet6View 前向传播成功，输出形状: {out_6view.shape}")
    except Exception as e:
        print(f"✗ MyNet6View 前向传播失败: {e}")
        return False
    
    try:
        with torch.no_grad():
            out_7view = model_7view(x)
        print(f"✓ MyNet7ViewTimeFreq 前向传播成功，输出形状: {out_7view.shape}")
    except Exception as e:
        print(f"✗ MyNet7ViewTimeFreq 前向传播失败: {e}")
        return False
    
    print()
    return True


def test_intermediate_features():
    """测试中间特征提取"""
    print("=" * 80)
    print("[Test 5] 中间特征提取（仅7视图）")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_7view = MyNet7ViewTimeFreq(num_classes=10).to(device)
    x = torch.randn(2, 12, 5000).to(device)
    
    try:
        with torch.no_grad():
            logits, intermediate = model_7view(x, return_intermediate=True)
        
        print(f"✓ 中间特征提取成功")
        print(f"  视图数: {len(intermediate['view_features'])}")
        print(f"  视图名称: {intermediate['view_names']}")
        
        # 计算视图权重
        fuse_weights = torch.cat(intermediate['fuse_weights'], dim=1)  # (2, 7)
        weights_softmax = torch.softmax(fuse_weights, dim=1)  # (2, 7)
        weights_mean = weights_softmax.mean(dim=0)  # (7,)
        
        print(f"\n  平均视图权重:")
        for i, (name, weight) in enumerate(zip(intermediate['view_names'], weights_mean)):
            marker = " ← 时频视图" if i == 6 else ""
            print(f"    {name:20s}: {weight:.4f}{marker}")
        
        print(f"  总权重: {weights_mean.sum():.4f} (应为1.0)")
        print()
        
    except Exception as e:
        print(f"✗ 中间特征提取失败: {e}\n")
        return False
    
    return True


def test_view_weights():
    """测试get_view_weights方法"""
    print("=" * 80)
    print("[Test 6] 视图权重获取")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_7view = MyNet7ViewTimeFreq(num_classes=10).to(device)
    x = torch.randn(8, 12, 5000).to(device)
    
    try:
        weights = model_7view.get_view_weights(x)  # (8, 7)
        
        print(f"✓ 视图权重获取成功，形状: {weights.shape}")
        print(f"\n  每个样本的权重 (行和应为1.0):")
        
        for i in range(min(3, weights.shape[0])):
            row_sum = weights[i].sum()
            print(f"    样本{i}: {weights[i]} (和={row_sum:.4f})")
        
        # 统计视图权重分布
        weights_mean = weights.mean(axis=0)
        weights_std = weights.std(axis=0)
        
        print(f"\n  各视图权重统计:")
        view_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7_TimeFreq']
        for name, mean, std in zip(view_names, weights_mean, weights_std):
            marker = " ← 时频视图" if name == 'V7_TimeFreq' else ""
            print(f"    {name:12s}: {mean:.4f} ± {std:.4f}{marker}")
        
        print()
        
    except Exception as e:
        print(f"✗ 视图权重获取失败: {e}\n")
        return False
    
    return True


def test_backward_pass():
    """测试反向传播"""
    print("=" * 80)
    print("[Test 7] 反向传播测试")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_6view = MyNet6View(num_classes=10).to(device)
    model_7view = MyNet7ViewTimeFreq(num_classes=10).to(device)
    
    x = torch.randn(4, 12, 5000).to(device)
    y = torch.randint(0, 2, (4, 10)).float().to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # 测试6视图
    try:
        out_6view = model_6view(x)
        loss_6view = criterion(out_6view, y)
        loss_6view.backward()
        print(f"✓ MyNet6View 反向传播成功，损失: {loss_6view.item():.4f}")
    except Exception as e:
        print(f"✗ MyNet6View 反向传播失败: {e}")
        return False
    
    # 测试7视图
    try:
        out_7view = model_7view(x)
        loss_7view = criterion(out_7view, y)
        loss_7view.backward()
        print(f"✓ MyNet7ViewTimeFreq 反向传播成功，损失: {loss_7view.item():.4f}")
    except Exception as e:
        print(f"✗ MyNet7ViewTimeFreq 反向传播失败: {e}")
        return False
    
    print()
    return True


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  MyNet7ViewTimeFreq - 验证测试".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝\n")
    
    tests = [
        test_model_comparison,
        test_inference_speed,
        test_memory_usage,
        test_forward_pass,
        test_intermediate_features,
        test_view_weights,
        test_backward_pass,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            if result is None:
                result = True
            results.append(("✓", test_func.__doc__))
        except Exception as e:
            print(f"✗ 异常: {str(e)[:50]}\n")
            results.append(("✗", test_func.__doc__))
    
    # 汇总
    print("=" * 80)
    print("测试汇总")
    print("=" * 80)
    
    for status, test_name in results:
        print(f"{status} {test_name}")
    
    print("=" * 80)
    
    passed = sum(1 for s, _ in results if s == "✓")
    total = len(results)
    
    if passed == total:
        print(f"\n🎉 所有测试通过！({passed}/{total})")
        print("MyNet7ViewTimeFreq 模型已准备好用于训练。\n")
        return True
    else:
        print(f"\n⚠️  有{total - passed}个测试未通过。({passed}/{total})\n")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
