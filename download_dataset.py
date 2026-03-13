from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter  # 用于统计标签数量（兼容旧版本）

# 加载数据集
ds = load_dataset("kushalps/cpsc2018")

# ---------------------- 1. 确认字段和标签映射 ----------------------
print("字段名称：", ds["train"].column_names)
print("标签映射（数字→类别名称）：")
label_names = ds["train"].features["label"].names  # 9类标签名称
for idx, name in enumerate(label_names):
    print(f"  数字 {idx} → 类别 {name}")

# ---------------------- 2. 查看单条样本（图片+标签）----------------------
print("\n=== 训练集第1条样本 ===")
sample = ds["train"][0]
label_idx = sample["label"]
label_name = label_names[label_idx]
print(f"标签：数字 {label_idx} → 类别 {label_name}")
print(f"图片尺寸：{sample['image'].size}，图片模式：{sample['image'].mode}")

# 显示图片（直观查看心电图）
plt.figure(figsize=(8, 8))
plt.imshow(sample["image"])
plt.title(f"ECG Image (Label: {label_name})")
plt.axis("off")  # 隐藏坐标轴
plt.show()

# ---------------------- 3. 查看图片像素值（用于模型训练参考）----------------------
image_array = np.array(sample["image"])
print(f"\n图片数组形状：{image_array.shape}")  # (518, 518, 3) → 高×宽×RGB通道
print(f"像素值范围：{image_array.min()} ~ {image_array.max()}")  # 3~255（RGB图片）

# ---------------------- 4. 批量查看前5条样本标签 ----------------------
print("\n=== 训练集前5条样本的标签 ===")
samples = ds["train"][:5]
for i, (img, lbl_idx) in enumerate(zip(samples["image"], samples["label"])):
    lbl_name = label_names[lbl_idx]
    print(f"第{i+1}条：数字 {lbl_idx} → 类别 {lbl_name}，图片尺寸 {img.size}")

# ---------------------- 5. 兼容版：统计标签分布（核心修正！）----------------------
print("\n=== 训练集标签分布（数量+比例）===")
# 步骤1：提取所有训练集标签（变成列表，兼容旧版本）
all_labels = [sample["label"] for sample in ds["train"]]  # 循环提取每个样本的标签
total_samples = len(all_labels)  # 训练集总样本数（44327）

# 步骤2：用 Counter 统计每个标签的数量（Python内置，无需额外依赖）
label_counter = Counter(all_labels)  # 结果：{标签索引: 数量}

# 步骤3：按标签索引排序（保证和标签映射顺序一致）
sorted_labels = sorted(label_counter.items(), key=lambda x: x[0])  # 按标签数字排序

# 步骤4：输出数量和比例，并存入列表用于可视化
labels = []  # 标签名称（如1AVB、Normal）
counts = []  # 每个标签的样本数
ratios = []  # 每个标签的占比（百分比）
for lbl_idx, count in sorted_labels:
    lbl_name = label_names[lbl_idx]
    ratio = (count / total_samples) * 100  # 计算占比
    labels.append(lbl_name)
    counts.append(count)
    ratios.append(ratio)
    print(f"{lbl_name}：{count} 条（{ratio:.2f}%）")

# ---------------------- 6. 可视化标签分布（柱状图，直观查看平衡性）----------------------
plt.figure(figsize=(12, 6))
bars = plt.bar(labels, counts, color='#2E86AB')  # 蓝色柱状图
plt.title("CPSC2018 Training Set Label Distribution", fontsize=14, fontweight='bold')
plt.xlabel("ECG Diagnosis Label", fontsize=12)
plt.ylabel("Sample Count", fontsize=12)
plt.xticks(rotation=45, ha="right")  # 标签名旋转45度，避免重叠

# 在柱子上标注「数量+比例」（更清晰）
for bar, count, ratio in zip(bars, counts, ratios):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2.,  # 水平居中
        height + 100,  # 位置在柱子顶部上方100处
        f"{count}\n({ratio:.1f}%)",  # 显示内容
        ha='center', va='bottom', fontsize=10
    )

plt.tight_layout()  # 自动调整布局，防止标签被截断
plt.show()