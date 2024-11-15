import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.family'] = ['Heiti TC']  # 或者使用其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)


def tanh(x):
    """tanh激活函数"""
    return np.tanh(x)



# 创建输入值
x = np.linspace(-5, 5, 200)

# 创建图形
plt.figure(figsize=(12, 6))

# 绘制ReLU
plt.subplot(1, 2, 1)
plt.plot(x, relu(x), 'b-', label='ReLU')
plt.grid(True)
plt.title('ReLU激活函数')
plt.xlabel('输入 x')
plt.ylabel('输出 y')
plt.legend()
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# 绘制tanh
plt.subplot(1, 2, 2)
plt.plot(x, tanh(x), 'r-', label='tanh')
plt.grid(True)
plt.title('tanh激活函数')
plt.xlabel('输入 x')
plt.ylabel('输出 y')
plt.legend()
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()

plt.show()