import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 Seaborn 风格
sns.set(style="whitegrid")

# 1. 加载保存的 DCT 系数
dct_data = torch.load("high_low_dct.pth")
high_dct = dct_data['high_dct'].detach().cpu().numpy()  # (B, C, H, W)
low_dct = dct_data['low_dct'].detach().cpu().numpy()

# 2. 读取原始输入图像 (x) 和 YCbCr 变换后的 DCT 频谱
input_data = torch.load("input_images.pth")
x_img = input_data['x'].detach().cpu().numpy()[:, 0, :, :]  # 取所有 batch 的 Y 通道 (空间域)
x_ycbcr_img = input_data['x_yxbcr'].detach().cpu().numpy()[:, 0, :, :]  # 取 Y 通道 (DCT 频域)


# 3. 计算 DCT 频谱的能量强度
def compute_energy(dct_coeffs):
    return np.sum(np.abs(dct_coeffs), axis=(1, 2))  # 计算每张图的总能量


high_energy = compute_energy(high_dct[:, 0, :, :])
low_energy = compute_energy(low_dct[:, 0, :, :])

# 4. 处理每个 batch 并保存
for i in range(high_dct.shape[0]):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    # 进行 log 变换以增强对比度
    high_dct_img = high_dct[i, 0, :, :]
    low_dct_img = low_dct[i, 0, :, :]
    high_dct_log = np.log1p(np.abs(high_dct_img))
    low_dct_log = np.log1p(np.abs(low_dct_img))
    x_ycbcr_log = np.log1p(np.abs(x_ycbcr_img[i]))

    # (第一行) 原始空间域 vs. 频域
    sns.heatmap(x_img[i], cmap="gray", ax=axes[0, 0])
    axes[0, 0].set_title("Original Image (Spatial)")

    sns.heatmap(x_ycbcr_img[i], cmap="coolwarm", ax=axes[0, 1])
    axes[0, 1].set_title("YCbCr DCT Transformed Image")

    sns.heatmap(x_ycbcr_log, cmap="magma", ax=axes[0, 2])
    axes[0, 2].set_title("YCbCr DCT Transformed (Log)")

    # (第二行) 高频部分
    sns.heatmap(high_dct_img, cmap="coolwarm", ax=axes[1, 0])
    axes[1, 0].set_title("High Frequency DCT")

    sns.heatmap(high_dct_log, cmap="magma", ax=axes[1, 1])
    axes[1, 1].set_title("High Frequency DCT (Log)")

    sns.heatmap(np.abs(high_dct_img), cmap="inferno", ax=axes[1, 2])
    axes[1, 2].set_title("High Frequency (Magnitude)")

    # (第三行) 低频部分
    sns.heatmap(low_dct_img, cmap="coolwarm", ax=axes[2, 0])
    axes[2, 0].set_title("Low Frequency DCT")

    sns.heatmap(low_dct_log, cmap="magma", ax=axes[2, 1])
    axes[2, 1].set_title("Low Frequency DCT (Log)")

    sns.heatmap(np.abs(low_dct_img), cmap="inferno", ax=axes[2, 2])
    axes[2, 2].set_title("Low Frequency (Magnitude)")

    # 5. 调整布局并保存
    plt.tight_layout()
    plt.savefig(f"dct_comparison_{i}.png", dpi=300)
    plt.close()

# 6. 绘制能量强度直方图
plt.figure(figsize=(8, 5))
plt.hist(high_energy, bins=20, alpha=0.7, label='High Frequency Energy', color='r')
plt.hist(low_energy, bins=20, alpha=0.7, label='Low Frequency Energy', color='b')
plt.xlabel("Energy")
plt.ylabel("Frequency")
plt.legend()
plt.title("High vs Low Frequency Energy Distribution")
plt.savefig("dct_energy_distribution.png", dpi=300)
plt.show()
