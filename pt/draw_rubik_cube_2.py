import argparse

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_cube(ax: Axes, origin=(0, 0, 0), size=1, label=None, max_coords=(2, 2, 2)):
    """
    绘制一个单独的立方体
    :param ax: 3D坐标系
    :param origin: 立方体的起始位置（左下角）
    :param size: 立方体的边长
    :param max_coords: 魔方的最大坐标值(x_max, y_max, z_max)
    """
    x, y, z = origin
    x_max, y_max, z_max = max_coords
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
        poly3d = Poly3DCollection([face], facecolors=colors[i], linewidths=1, edgecolors='k', alpha=0.8)
        ax.add_collection3d(poly3d)

    if label is not None:
        # 计算立方体的中心位置
        center_x = x + size / 2
        center_y = y + size / 2
        center_z = z + size / 2
        label_offset = 0.1

        # 前面和后面
        if x == 0:  # 前面
            ax.text(center_x, y - label_offset, center_z, str(label), color='black', fontsize=8, ha='center', va='center')
        if x == x_max:  # 后面
            ax.text(center_x, y + size + label_offset, center_z, str(label), color='black', fontsize=8, ha='center', va='center')

        # 左面和右面
        if y == 0:  # 左面
            ax.text(x - label_offset, center_y, center_z, str(label), color='black', fontsize=8, ha='center', va='center')
        if y == y_max:  # 右面
            ax.text(x + size + label_offset, center_y, center_z, str(label), color='black', fontsize=8, ha='center', va='center')

        # 底面和顶面
        if z == 0:  # 底面
            ax.text(center_x, center_y, z - label_offset, str(label), color='black', fontsize=8, ha='center', va='center')
        if z == z_max:  # 顶面
            ax.text(center_x, center_y, z + size + label_offset, str(label), color='black', fontsize=8, ha='center', va='center')


def draw_rubik_cube(ax: Axes, x=3, y=3, z=3):
    """
    绘制一个3阶魔方，沿x, y, z轴方向堆叠小立方体
    :param ax: 3D坐标系
    :param x: x轴方向的小立方体数
    :param y: y轴方向的小立方体数
    :param z: z轴方向的小立方体数
    """
    counter = 0  # 用于标记序号
    # 计算最大坐标值（减1是因为坐标从0开始）
    max_coords = (x - 1, y - 1, z - 1)

    for i in range(x):
        for j in range(y):
            for k in range(z):
                origin = (i, j, k)
                draw_cube(ax, origin=origin, size=1, label=counter, max_coords=max_coords)
                counter += 1


def draw_axes_ticks(ax: Axes, x=3, y=3, z=3):
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置坐标轴范围，稍微扩大范围以显示标签
    ax.set_xlim([-0.5, x + 0.5])
    ax.set_ylim([-0.5, y + 0.5])
    ax.set_zlim([-0.5, z + 0.5])

    # 设置整数刻度
    ax.set_xticks(range(0, x + 1))
    ax.set_yticks(range(0, y + 1))
    ax.set_zticks(range(0, z + 1))


def draw_rubik(ax: Axes, cube=(3, 3, 3)):
    x1, y1, z1 = cube
    draw_rubik_cube(ax, x=x1, y=y1, z=z1)
    draw_axes_ticks(ax, x=x1, y=y1, z=z1)

    # 设置合适的视角
    # ax.view_init(elev=20, azim=45)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--x", type=int, default=3, help="X axis")
    parser.add_argument("-y", "--y", type=int, default=3, help="Y axis")
    parser.add_argument("-z", "--z", type=int, default=3, help="Z axis")

    args = parser.parse_args()

    # 创建一个3D坐标系
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    draw_rubik(ax, (args.x, args.y, args.z))

    # 显示图形
    plt.show()