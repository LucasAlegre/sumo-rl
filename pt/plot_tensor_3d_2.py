import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_tensor_3d(tensor):
    """
    绘制 3D 张量的示意图。

    参数:
    tensor (np.ndarray): 输入的 3D 张量,维度为 (z, x, y)。
    """
    # 获取张量的尺寸
    z_dim, x_dim, y_dim = tensor.shape

    # 创建 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制立方体
    for z in range(z_dim):
        for x in range(x_dim):
            for y in range(y_dim):
                ax.bar3d(x, y, z, 1, 1, 1, color='lightgray', edgecolor='black')
                ax.text(x, y, z, str(tensor[z, x, y]), ha='center', va='center', color='black')

    # 设置坐标轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Tensor Visualization')

    # 设置坐标轴范围和刻度
    ax.set_xlim(0, x_dim)
    ax.set_ylim(0, y_dim)
    ax.set_zlim(0, z_dim)
    ax.set_xticks(np.arange(0, x_dim, 1))
    ax.set_yticks(np.arange(0, y_dim, 1))
    ax.set_zticks(np.arange(0, z_dim, 1))

    # 调整视角
    ax.view_init(elev=-90, azim=-90)

    # 显示图形
    plt.show()

# 示例 3D 张量
tensor_3d = np.array([
    [[0, 1, 2, 3],
     [4, 5, 6, 7],
     [8, 9, 10, 11]],

    [[12, 13, 14, 15],
     [16, 17, 18, 19],
     [20, 21, 22, 23]]
])

plot_tensor_3d(tensor_3d)
