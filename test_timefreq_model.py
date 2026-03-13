# -*- coding: utf-8 -*-
"""
@time: 2026/1/5
@description: 时频融合模型验证脚本
    检查模型是否能正常前向传播、计算梯度、保存加载等
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# 导入模型
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models import TimeFreqFusionNet
from timefreq_visualizer import TimeFreqVisualizer


def test_model_forward():
    """测试模型前向传播"""
    print("=" * 80)
    print("[Test 1] 模型前向传播")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")
    
    # 创建模型
    model = TimeFreqFusionNet(num_classes=10, input_channels=12)
    model = model.to(device)
    
    # 生成随机输入
    batch_size = 4
    x = torch.randn(batch_size, 12, 5000).to(device)
    y = torch.randint(0, 2, (batch_size, 10)).float().to(device)
    
    print(f"输入形状: {x.shape}")
    print(f"标签形状: {y.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 前向传播
    try:
        with torch.no_grad():
            logits = model(x)
        print(f"✓ 前向传播成功")
        print(f"  输出形状: {logits.shape}")
        print(f"  输出范围: [{logits.min():.3f}, {logits.max():.3f}]")
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False
    
    # 梯度计算
    try:
        logits = model(x)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        loss.backward()
        print(f"✓ 梯度计算成功")
        print(f"  损失值: {loss.item():.4f}")
        
        # 检查梯度
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        print(f"  梯度范数: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}")
    except Exception as e:
        print(f"✗ 梯度计算失败: {e}")
        return False
    
    print()
    return True


def test_model_with_label_condition():
    """测试标签条件融合"""
    print("=" * 80)
    print("[Test 2] 标签条件融合")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TimeFreqFusionNet(num_classes=10, input_channels=12)
    model = model.to(device)
    
    x = torch.randn(4, 12, 5000).to(device)
    y_soft = torch.randn(4, 10).to(device)
    
    try:
        with torch.no_grad():
            logits = model(x, y_soft=y_soft)
        print(f"✓ 标签条件融合成功")
        print(f"  输出形状: {logits.shape}\n")
    except Exception as e:
        print(f"✗ 标签条件融合失败: {e}\n")
        return False
    
    return True


def test_get_intermediate_features():
    """测试中间特征提取"""
    print("=" * 80)
    print("[Test 3] 中间特征提取")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TimeFreqFusionNet(num_classes=10, input_channels=12)
    model = model.to(device)
    
    x = torch.randn(2, 12, 5000).to(device)
    
    try:
        with torch.no_grad():
            feats = model.get_intermediate_features(x)
        
        print(f"✓ 中间特征提取成功")
        print(f"  时域特征: {feats['time_feat'].shape}")
        print(f"  频域特征: {feats['freq_feat'].shape}")
        print(f"  时域预测: {feats['time_logits'].shape}")
        print(f"  频域预测: {feats['freq_logits'].shape}")
        print(f"  谱图数量: {len(feats['spectrograms'])} (多尺度)")
        if len(feats['spectrograms']) > 0:
            print(f"  谱图形状: {feats['spectrograms'][0].shape}\n")
    except Exception as e:
        print(f"✗ 中间特征提取失败: {e}\n")
        return False
    
    return True


def test_visualizer():
    """测试可视化工具"""
    print("=" * 80)
    print("[Test 4] 可视化工具")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TimeFreqFusionNet(num_classes=10, input_channels=12)
    model = model.to(device)
    
    try:
        viz = TimeFreqVisualizer(model, device=device)
        print(f"✓ 可视化工具初始化成功")
        
        # 测试单个样本可视化
        x = torch.randn(1, 12, 5000).to(device)
        viz.visualize_single_sample(
            x, 
            save_path='test_viz',
            label_name='TestSample'
        )
        
        # 测试批量对比
        x_batch = torch.randn(8, 12, 5000).to(device)
        y_batch = torch.randint(0, 2, (8, 10)).float().to(device)
        viz.compare_predictions_batch(x_batch, y_batch, save_path='test_viz')
        
        print(f"✓ 所有可视化生成成功\n")
    except Exception as e:
        print(f"✗ 可视化失败: {e}\n")
        return False
    
    return True


def test_model_save_load():
    """测试模型保存和加载"""
    print("=" * 80)
    print("[Test 5] 模型保存/加载")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建并保存模型
    model1 = TimeFreqFusionNet(num_classes=10, input_channels=12)
    model1 = model1.to(device)
    
    checkpoint_path = 'test_checkpoint.pth'
    
    try:
        torch.save({
            'model_state_dict': model1.state_dict(),
            'num_classes': 10,
        }, checkpoint_path)
        print(f"✓ 模型保存成功: {checkpoint_path}")
        
        # 加载模型
        model2 = TimeFreqFusionNet(num_classes=10, input_channels=12)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model2.load_state_dict(checkpoint['model_state_dict'])
        model2 = model2.to(device)
        print(f"✓ 模型加载成功")
        
        # 验证两个模型的输出一致
        x = torch.randn(2, 12, 5000).to(device)
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)
        
        diff = (out1 - out2).abs().max().item()
        if diff < 1e-5:
            print(f"✓ 模型输出一致 (差异={diff:.2e})\n")
        else:
            print(f"⚠ 模型输出差异较大 (差异={diff:.2e})\n")
        
        # 清理
        import os
        os.remove(checkpoint_path)
        
    except Exception as e:
        print(f"✗ 模型保存/加载失败: {e}\n")
        return False
    
    return True


def test_different_input_sizes():
    """测试不同输入尺寸的兼容性"""
    print("=" * 80)
    print("[Test 6] 不同输入尺寸兼容性")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeFreqFusionNet(num_classes=10, input_channels=12)
    model = model.to(device)
    
    input_sizes = [2500, 5000, 7500, 10000]
    
    try:
        for size in input_sizes:
            x = torch.randn(2, 12, size).to(device)
            with torch.no_grad():
                logits = model(x)
            print(f"✓ 输入尺寸 {size:5d} → 输出 {logits.shape}")
        print()
    except Exception as e:
        print(f"✗ 输入尺寸测试失败: {e}\n")
        return False
    
    return True


def test_batch_sizes():
    """测试不同批量大小"""
    print("=" * 80)
    print("[Test 7] 不同批量大小")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeFreqFusionNet(num_classes=10, input_channels=12)
    model = model.to(device)
    
    batch_sizes = [1, 4, 8, 16]
    
    try:
        for bs in batch_sizes:
            x = torch.randn(bs, 12, 5000).to(device)
            with torch.no_grad():
                logits = model(x)
            print(f"✓ 批量大小 {bs:2d} → 输出 {logits.shape}")
        print()
    except Exception as e:
        print(f"✗ 批量大小测试失败: {e}\n")
        return False
    
    return True


def test_memory_usage():
    """测试显存占用"""
    print("=" * 80)
    print("[Test 8] 显存占用")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("GPU不可用，跳过此测试\n")
        return True
    
    device = torch.device("cuda")
    model = TimeFreqFusionNet(num_classes=10, input_channels=12)
    model = model.to(device)
    
    # 清除GPU缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        x = torch.randn(8, 12, 5000).to(device)
        with torch.no_grad():
            logits = model(x)
        
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
        current_memory = torch.cuda.memory_allocated(device) / 1024**2
        
        print(f"✓ 显存占用统计:")
        print(f"  当前占用: {current_memory:.2f} MB")
        print(f"  峰值占用: {peak_memory:.2f} MB")
        print()
        
    except Exception as e:
        print(f"✗ 显存占用测试失败: {e}\n")
        return False
    
    return True


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  多尺度时频融合ECG模型 - 验证测试".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝\n")
    
    tests = [
        ("模型前向传播", test_model_forward),
        ("标签条件融合", test_model_with_label_condition),
        ("中间特征提取", test_get_intermediate_features),
        ("可视化工具", test_visualizer),
        ("模型保存/加载", test_model_save_load),
        ("不同输入尺寸", test_different_input_sizes),
        ("不同批量大小", test_batch_sizes),
        ("显存占用", test_memory_usage),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "✓ 通过" if result else "✗ 失败"
        except Exception as e:
            results[test_name] = f"✗ 异常: {str(e)[:30]}"
    
    # 汇总报告
    print("=" * 80)
    print("测试汇总")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if "通过" in v)
    total = len(results)
    
    for test_name, result in results.items():
        status_icon = "✓" if "通过" in result else "✗"
        print(f"{status_icon} {test_name:20s} ... {result}")
    
    print("-" * 80)
    print(f"通过: {passed}/{total}")
    print("=" * 80)
    
    if passed == total:
        print("\n🎉 所有测试通过！模型可以正常使用。\n")
        return True
    else:
        print(f"\n⚠️  有{total - passed}个测试未通过，请检查错误信息。\n")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
