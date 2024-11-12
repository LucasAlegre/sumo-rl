import argparse

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_cube(ax: Axes, origin=(0, 0, 0), size=1, label=None):
    """
    绘制一个单独的立方体
    :param ax: 3D坐标系
    :param origin: 立方体的起始位置（左下角）
    :param size: 立方体的边长
    """
    x, y, z = origin
    vertices = [
        [x, y, z],  # 0
        [x + size, y, z],  # 1
        [x + size, y + size, z],  # 2
        [x, y + size, z],  # 3
        [x, y, z + size],  # 4
        [x + size, y, z + size],  # 5
        [x + size, y + size, z + size],  # 6
        [x, y + size, z + size]  # 7
    ]

    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
        [vertices[3], vertices[2], vertices[6], vertices[7]],  # 后面
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
        [vertices[1], vertices[2], vertices[6], vertices[5]]  # 右面
    ]

    colors = ['green', 'blue', 'red', 'orange', 'white', 'yellow']

    for i, face in enumerate(faces):
        poly3d = Poly3DCollection([face], facecolors=colors[i], linewidths=1, edgecolors='k', alpha=0.9)
        ax.add_collection3d(poly3d)

    if label is not None:
        # 计算小立方体的中心位置
        # center = (x + 0.1, y + 0.1, z + 0.1)
        center = (x, y, z)
        # 在中心位置添加标签
        ax.text(center[0], center[1], center[2], str(label), color='black', fontsize=10, ha='center', va='center', zorder=10)


def draw_rubik_cube(ax: Axes, x=3, y=3, z=3):
    """
    绘制一个3阶魔方，沿x, y, z轴方向堆叠小立方体
    :param ax: 3D坐标系
    :param x: x轴方向的小立方体数
    :param y: y轴方向的小立方体数
    :param z: z轴方向的小立方体数
    """
    counter = 0  # 用于标记序号
    for i in range(x):
        for j in range(y):
            for k in range(z):
                # 计算每个小立方体的原点坐标
                origin = (i, j, k)
                draw_cube(ax, origin=origin, size=1, label=counter)
                counter += 1


def draw_axes_ticks(ax: Axes, x=3, y=3, z=3):
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置坐标轴范围
    ax.set_xlim([0, x])
    ax.set_ylim([0, y])
    ax.set_zlim([0, z])

    # 设置整数刻度
    ax.set_xticks(range(0, x + 1))  # 在x轴上标0到3的整数
    ax.set_yticks(range(0, y + 1))  # 在y轴上标0到3的整数
    ax.set_zticks(range(0, z + 1))  # 在z轴上标0到3的整数


def draw_rubik(ax: Axes, cube=(3, 3, 3)):
    x1, y1, z1 = cube
    draw_rubik_cube(ax, x=x1, y=y1, z=z1)
    draw_axes_ticks(ax, x=x1, y=y1, z=z1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--x", type=int, default=3, help="X axis")
    parser.add_argument("-y", "--y", type=int, default=3, help="Y axis")
    parser.add_argument("-z", "--z", type=int, default=3, help="Z axis")

    args = parser.parse_args()

    # 创建一个3D坐标系
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    draw_rubik(ax, (args.x, args.y, args.z))

    # 显示图形
    plt.show()
