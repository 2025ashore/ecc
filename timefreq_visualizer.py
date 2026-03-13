# -*- coding: utf-8 -*-
"""
@time: 2026/1/5
@description: 时频融合模型的可视化和诊断工具
    用于理解模型如何利用时频特征，检验改进效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TimeFreqVisualizer:
    """时频融合模型的可视化工具"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def visualize_single_sample(self, x, save_path='timefreq_viz', 
                               sample_idx=0, label_name='Sample'):
        """可视化单个样本的时频处理过程
        
        参数：
            x: 输入信号，形状 (1, 12, 5000) 或 (12, 5000)
            save_path: 保存图像的路径
            sample_idx: 样本索引
            label_name: 标签名称
        """
        os.makedirs(save_path, exist_ok=True)
        
        # 确保输入维度正确
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        
        # 获取中间特征
        with torch.no_grad():
            feats = self.model.get_intermediate_features(x)
        
        # 绘图
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        fig.suptitle(f'时频融合模型分析 - {label_name}', fontsize=16, fontweight='bold')
        
        # 1. 原始时域信号
        ax = axes[0, 0]
        signal = x[0, 0, :].cpu().numpy()  # 取第一个通道
        ax.plot(signal, linewidth=0.5)
        ax.set_title('原始ECG信号 (通道0)', fontsize=12)
        ax.set_xlabel('时间样本')
        ax.set_ylabel('幅度')
        ax.grid(True, alpha=0.3)
        
        # 2. 多导联信号叠加
        ax = axes[0, 1]
        for ch in range(min(3, x.shape[1])):
            signal = x[0, ch, :].cpu().numpy()
            ax.plot(signal, label=f'Lead {ch}', linewidth=0.5, alpha=0.7)
        ax.set_title('多导联信号 (前3个)', fontsize=12)
        ax.set_xlabel('时间样本')
        ax.set_ylabel('幅度')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 3. 时域特征分布
        ax = axes[0, 2]
        time_feat = feats['time_feat'][0].cpu().numpy()
        ax.hist(time_feat, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax.set_title(f'时域特征分布 (dim={time_feat.shape[0]})', fontsize=12)
        ax.set_xlabel('特征值')
        ax.set_ylabel('频数')
        ax.grid(True, alpha=0.3)
        
        # 4. 多尺度谱图 - 短窗
        ax = axes[1, 0]
        spec_short = feats['spectrograms'][0][0, 0].cpu().numpy()  # (n_freqs, T)
        im = ax.imshow(spec_short, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title('多尺度谱图 - 短窗口', fontsize=12)
        ax.set_xlabel('时间帧')
        ax.set_ylabel('频率 bin')
        plt.colorbar(im, ax=ax, label='dB')
        
        # 5. 多尺度谱图 - 中窗
        ax = axes[1, 1]
        spec_mid = feats['spectrograms'][1][0, 0].cpu().numpy() if len(feats['spectrograms']) > 1 else spec_short
        im = ax.imshow(spec_mid, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title('多尺度谱图 - 中窗口', fontsize=12)
        ax.set_xlabel('时间帧')
        ax.set_ylabel('频率 bin')
        plt.colorbar(im, ax=ax, label='dB')
        
        # 6. 多尺度谱图 - 长窗
        ax = axes[1, 2]
        spec_long = feats['spectrograms'][2][0, 0].cpu().numpy() if len(feats['spectrograms']) > 2 else spec_short
        im = ax.imshow(spec_long, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title('多尺度谱图 - 长窗口', fontsize=12)
        ax.set_xlabel('时间帧')
        ax.set_ylabel('频率 bin')
        plt.colorbar(im, ax=ax, label='dB')
        
        # 7. 频域特征分布
        ax = axes[2, 0]
        freq_feat = feats['freq_feat'][0].cpu().numpy()
        ax.hist(freq_feat, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax.set_title(f'频域特征分布 (dim={freq_feat.shape[0]})', fontsize=12)
        ax.set_xlabel('特征值')
        ax.set_ylabel('频数')
        ax.grid(True, alpha=0.3)
        
        # 8. 时频特征对比（2D）
        ax = axes[2, 1]
        # 随机选择2个特征维度进行散点图
        feat_indices = np.random.choice(min(time_feat.shape[0], freq_feat.shape[0]), 
                                        size=min(2, min(time_feat.shape[0], freq_feat.shape[0])), 
                                        replace=False)
        if len(feat_indices) >= 2:
            ax.scatter(time_feat[feat_indices[0]:feat_indices[0]+1], 
                      time_feat[feat_indices[1]:feat_indices[1]+1], 
                      label='Time', s=50, alpha=0.7)
            ax.scatter(freq_feat[feat_indices[0]:feat_indices[0]+1], 
                      freq_feat[feat_indices[1]:feat_indices[1]+1], 
                      label='Freq', s=50, alpha=0.7)
            ax.set_xlabel(f'Feature {feat_indices[0]}')
            ax.set_ylabel(f'Feature {feat_indices[1]}')
            ax.set_title('时频特征对比', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 9. 预测 logits 对比
        ax = axes[2, 2]
        time_logits = feats['time_logits'][0].cpu().numpy()
        freq_logits = feats['freq_logits'][0].cpu().numpy()
        
        x_pos = np.arange(min(5, len(time_logits)))  # 只显示前5个类
        width = 0.35
        ax.bar(x_pos - width/2, time_logits[x_pos], width, label='Time Branch', alpha=0.8)
        ax.bar(x_pos + width/2, freq_logits[x_pos], width, label='Freq Branch', alpha=0.8)
        ax.set_xlabel('类别')
        ax.set_ylabel('Logits值')
        ax.set_title('时频分支预测对比 (前5类)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_file = os.path.join(save_path, f'{label_name}_timefreq_analysis.png')
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f'✓ 可视化图保存到: {save_file}')
        plt.close()
    
    def compare_predictions_batch(self, x_batch, y_batch, save_path='timefreq_viz'):
        """批量对比时频分支的预测结果
        
        参数：
            x_batch: 批量输入，形状 (B, 12, 5000)
            y_batch: 批量标签，形状 (B, num_classes)
        """
        os.makedirs(save_path, exist_ok=True)
        
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        
        with torch.no_grad():
            feats = self.model.get_intermediate_features(x_batch)
        
        time_logits = feats['time_logits'].cpu().numpy()  # (B, num_classes)
        freq_logits = feats['freq_logits'].cpu().numpy()
        final_logits = self.model(x_batch).cpu().numpy()
        targets = y_batch.cpu().numpy()
        
        # 计算分支间的相关性
        batch_size, num_classes = time_logits.shape
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 1. 时域分支 vs 频域分支的logits相关性
        ax = axes[0]
        correlation_matrix = np.zeros((batch_size, num_classes))
        for b in range(batch_size):
            for c in range(num_classes):
                correlation_matrix[b, c] = np.corrcoef(
                    time_logits[b, :], freq_logits[b, :]
                )[0, 1]
        
        im = ax.imshow(correlation_matrix, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xlabel('类别')
        ax.set_ylabel('样本')
        ax.set_title('时频分支Logits相关性', fontsize=12)
        plt.colorbar(im, ax=ax, label='相关系数')
        
        # 2. 分支权重分布
        ax = axes[1]
        time_weight = []
        freq_weight = []
        for b in range(batch_size):
            time_contrib = np.abs(time_logits[b]).sum()
            freq_contrib = np.abs(freq_logits[b]).sum()
            total = time_contrib + freq_contrib + 1e-8
            time_weight.append(time_contrib / total)
            freq_weight.append(freq_contrib / total)
        
        ax.hist([time_weight, freq_weight], bins=20, label=['Time', 'Freq'], 
               color=['blue', 'green'], alpha=0.7, edgecolor='black')
        ax.set_xlabel('分支权重')
        ax.set_ylabel('样本数')
        ax.set_title('分支权重分布', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. 分支预测差异
        ax = axes[2]
        time_pred = (torch.sigmoid(torch.tensor(time_logits)) > 0.5).cpu().numpy().astype(int)
        freq_pred = (torch.sigmoid(torch.tensor(freq_logits)) > 0.5).cpu().numpy().astype(int)
        
        disagreement_rate = (time_pred != freq_pred).mean(axis=1)  # (B,)
        ax.hist(disagreement_rate, bins=20, color='red', alpha=0.7, edgecolor='black')
        ax.set_xlabel('分支不同意的标签比例')
        ax.set_ylabel('样本数')
        ax.set_title(f'时频分支不同意率 (平均={disagreement_rate.mean():.2%})', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_file = os.path.join(save_path, 'batch_branch_comparison.png')
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f'✓ 批量对比图保存到: {save_file}')
        plt.close()
    
    def analyze_robustness(self, x_clean, y, noise_levels=[0.01, 0.05, 0.1, 0.2], 
                          save_path='timefreq_viz'):
        """分析模型对噪声的鲁棒性
        
        参数：
            x_clean: 干净信号，形状 (B, 12, 5000)
            y: 标签，形状 (B, num_classes)
            noise_levels: 噪声水平列表
        """
        os.makedirs(save_path, exist_ok=True)
        
        x_clean = x_clean.to(self.device)
        y = y.to(self.device)
        
        results = {'noise_level': [], 'time_branch_auc': [], 'freq_branch_auc': [], 
                  'fusion_auc': []}
        
        from sklearn.metrics import roc_auc_score
        
        # 干净数据的性能
        with torch.no_grad():
            clean_logits = self.model(x_clean).cpu().numpy()
            feats = self.model.get_intermediate_features(x_clean)
            time_logits_clean = feats['time_logits'].cpu().numpy()
            freq_logits_clean = feats['freq_logits'].cpu().numpy()
        
        y_np = y.cpu().numpy()
        
        # 计算干净数据的AUC
        clean_auc = roc_auc_score(y_np, clean_logits, average='macro', multi_class='ovo')
        time_auc_clean = roc_auc_score(y_np, time_logits_clean, average='macro', multi_class='ovo')
        freq_auc_clean = roc_auc_score(y_np, freq_logits_clean, average='macro', multi_class='ovo')
        
        results['noise_level'].append(0.0)
        results['time_branch_auc'].append(time_auc_clean)
        results['freq_branch_auc'].append(freq_auc_clean)
        results['fusion_auc'].append(clean_auc)
        
        # 加噪声
        for noise_level in noise_levels:
            x_noisy = x_clean + torch.randn_like(x_clean) * noise_level
            
            with torch.no_grad():
                noisy_logits = self.model(x_noisy).cpu().numpy()
                feats = self.model.get_intermediate_features(x_noisy)
                time_logits_noisy = feats['time_logits'].cpu().numpy()
                freq_logits_noisy = feats['freq_logits'].cpu().numpy()
            
            noisy_auc = roc_auc_score(y_np, noisy_logits, average='macro', multi_class='ovo')
            time_auc_noisy = roc_auc_score(y_np, time_logits_noisy, average='macro', multi_class='ovo')
            freq_auc_noisy = roc_auc_score(y_np, freq_logits_noisy, average='macro', multi_class='ovo')
            
            results['noise_level'].append(noise_level)
            results['time_branch_auc'].append(time_auc_noisy)
            results['freq_branch_auc'].append(freq_auc_noisy)
            results['fusion_auc'].append(noisy_auc)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(results['noise_level'], results['time_branch_auc'], 
               marker='o', label='时域分支', linewidth=2, markersize=8)
        ax.plot(results['noise_level'], results['freq_branch_auc'], 
               marker='s', label='频域分支', linewidth=2, markersize=8)
        ax.plot(results['noise_level'], results['fusion_auc'], 
               marker='^', label='融合模型', linewidth=2, markersize=8, color='red')
        
        ax.set_xlabel('噪声水平 (std)', fontsize=12)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title('模型鲁棒性分析 - 对高斯噪声', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.0])
        
        plt.tight_layout()
        save_file = os.path.join(save_path, 'robustness_analysis.png')
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f'✓ 鲁棒性分析图保存到: {save_file}')
        plt.close()


# 使用示例
if __name__ == '__main__':
    print('该模块作为库使用，请在测试脚本中导入使用')
    print('示例：')
    print('  from visualizer import TimeFreqVisualizer')
    print('  viz = TimeFreqVisualizer(model, device="cuda")')
    print('  viz.visualize_single_sample(x_sample, save_path="viz_output")')
