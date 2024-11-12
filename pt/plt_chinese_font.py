import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import sys
import platform


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


# 3. 使用特定字体文件
def use_specific_font(font_path):
    # 添加自定义字体文件
    font = font_manager.FontProperties(fname=font_path)
    return font


# 4. 创建示例图表
def create_sample_plot():
    # 设置中文字体
    setup_chinese_font()

    # 创建数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # 创建图表
    plt.figure(figsize=(12, 6))

    # 绘制两条线
    plt.plot(x, y1, label='正弦曲线')
    plt.plot(x, y2, label='余弦曲线')

    # 添加标题和标签
    plt.title('三角函数图像示例', fontsize=14)
    plt.xlabel('X轴（角度）', fontsize=12)
    plt.ylabel('Y轴（值）', fontsize=12)
    plt.legend(loc='upper right')

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()


# 5. 使用不同方法显示中文
def show_different_methods():
    setup_chinese_font()

    # 创建示例数据
    categories = ['第一季度', '第二季度', '第三季度', '第四季度']
    values = [23, 45, 56, 78]

    # 创建多个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 第一个图：柱状图
    ax1.bar(categories, values)
    ax1.set_title('季度销售额')
    ax1.set_ylabel('销售额（万元）')

    # 第二个图：饼图
    ax2.pie(values, labels=categories, autopct='%1.1f%%')
    ax2.set_title('季度销售占比')

    plt.tight_layout()
    plt.show()


# 6. 完整示例
def main():
    print("系统环境信息：")
    check_system()
    print("\n正在设置中文字体...")
    setup_chinese_font()

    print("\n创建示例图表...")
    create_sample_plot()

    print("\n展示不同类型的图表...")
    show_different_methods()

    # 如果要使用特定字体文件
    # font = use_specific_font('path/to/your/font.ttf')
    # plt.title('使用特定字体的标题', fontproperties=font)


if __name__ == "__main__":
    main()
