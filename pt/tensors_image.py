import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import platform

# 1. 创建一个简单的 4x4 RGB图片张量
# 使用具体的数值便于理解
simple_image = torch.tensor([
    # Red Channel (第0通道)
    [[100, 120, 150, 180],
     [110, 130, 160, 190],
     [120, 140, 170, 200],
     [130, 150, 180, 210]],

    # Green Channel (第1通道)
    [[50, 60, 70, 80],
     [55, 65, 75, 85],
     [60, 70, 80, 90],
     [65, 75, 85, 95]],


    # Blue Channel (第2通道)
    [[25, 30, 35, 40],
     [30, 35, 40, 45],
     [35, 40, 45, 50],
     [40, 45, 50, 55]]
], dtype=torch.float32)

print("图片张量形状:", simple_image.shape)  # [3, 4, 4]

# 2. 分析每个通道
print("\n各通道数据:")
print("红色通道:\n", simple_image[0])
print("\n绿色通道:\n", simple_image[1])
print("\n蓝色通道:\n", simple_image[2])

# 3. 查看特定位置的RGB值
row, col = 1, 2  # 选择第1行第2列的像素
pixel_rgb = simple_image[:, row, col]
print(f"\n所有层上位置({row},{col})的RGB值:", pixel_rgb)  # [160, 75, 40]


# 1. 检测系统环境
def check_system():
    system = platform.system()
    print(f"操作系统: {system}")
    print(f"Python版本: {sys.version}")
    # print(f"Matplotlib版本: {plt.__version__}")
    return system


# 2. 设置中文字体的几种方法
def setup_chinese_font():
    system = check_system()

    if system == 'Windows':
        # Windows系统，使用微软雅黑
        plt.rcParams['font.family'] = ['Microsoft YaHei']
    elif system == 'Darwin':
        # macOS系统，使用苹方字体
        plt.rcParams['font.family'] = ['PingFang HK']
    else:
        # Linux系统，需要提前安装字体
        plt.rcParams['font.family'] = ['Noto Sans CJK JP']

    # 用于显示负号
    plt.rcParams['axes.unicode_minus'] = False


# 4. 显示图片
def display_channels(image_tensor):
    # 设置中文字体
    setup_chinese_font()

    # 创建一个2x2的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # 显示完整RGB图片
    rgb_img = image_tensor.permute(1, 2, 0).numpy() / 255.0
    axs[0, 0].imshow(rgb_img)
    axs[0, 0].set_title('RGB完整图片')

    # 显示红色通道
    axs[0, 1].imshow(image_tensor[0], cmap='Reds')
    axs[0, 1].set_title('红色通道')

    # 显示绿色通道
    axs[1, 0].imshow(image_tensor[1], cmap='Greens')
    axs[1, 0].set_title('绿色通道')

    # 显示蓝色通道
    axs[1, 1].imshow(image_tensor[2], cmap='Blues')
    axs[1, 1].set_title('蓝色通道')

    plt.tight_layout()
    plt.show()


display_channels(simple_image)

# 5. 基本统计信息
print("\n基本统计信息:")
print("红色通道均值:", simple_image[0].mean().item())
print("绿色通道均值:", simple_image[1].mean().item())
print("蓝色通道均值:", simple_image[2].mean().item())
print("整体像素最大值:", simple_image.max().item())
print("整体像素最小值:", simple_image.min().item())

# 6. 图片变换示例
# 调整亮度（所有值翻倍）
brighter_image = simple_image * 2
# 水平翻转
flipped_image = torch.flip(simple_image, [2])
# 旋转90度（调换行列）
rotated_image = torch.rot90(simple_image, 1, [1, 2])

print("\n变换后的形状:")
print("调亮后形状:", brighter_image.shape)
print("翻转后形状:", flipped_image.shape)
print("旋转后形状:", rotated_image.shape)
