import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# 创建一个绘图窗口
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 定义每个小方块的边长
block_size = 1

# 魔方6个面对应的颜色
colors = {
    'top': 'blue',  # 上面是蓝色
    'bottom': 'green',  # 下面是绿色
    'left': 'white',  # 左面是白色
    'right': 'yellow',  # 右面是黄色
    'front': 'red',  # 前面是红色
    'back': 'orange'  # 后面是橙色
}


# 绘制单个小方块
def draw_single_block(ax, x, y, z, face_colors):
    # 小方块的顶点坐标
    vertices = np.array([[x, y, z], [x + block_size, y, z], [x + block_size, y + block_size, z], [x, y + block_size, z],  # top
                         [x, y, z - block_size], [x + block_size, y, z - block_size], [x + block_size, y + block_size, z - block_size],
                         [x, y + block_size, z - block_size]])  # bottom
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # top
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # bottom
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
    ]

    # 绘制每个面的颜色
    for i, face in enumerate(faces):
        poly3d = Poly3DCollection([face], color=face_colors[i], linewidths=1, edgecolors='black')
        ax.add_collection3d(poly3d)


# 绘制一个 3x3x3 的立方体
def draw_cube():
    # 只渲染外部的小方块
    for x in range(3):
        for y in range(3):
            for z in range(3):
                # 判断是否是外部的小方块，外部方块坐标：x, y, z 在 0 或 2
                if x == 0 or x == 2 or y == 0 or y == 2 or z == 0 or z == 2:
                    face_colors = []

                    # 判断每个小方块是否在外部面上，并分配颜色
                    if z == 0:  # 前面
                        face_colors.append(colors['front'])
                    else:
                        face_colors.append('gray')  # 内部不显示颜色

                    if z == 2:  # 后面
                        face_colors.append(colors['back'])
                    else:
                        face_colors.append('gray')  # 内部不显示颜色

                    if x == 0:  # 左面
                        face_colors.append(colors['left'])
                    else:
                        face_colors.append('gray')  # 内部不显示颜色

                    if x == 2:  # 右面
                        face_colors.append(colors['right'])
                    else:
                        face_colors.append('gray')  # 内部不显示颜色

                    if y == 0:  # 下面
                        face_colors.append(colors['bottom'])
                    else:
                        face_colors.append('gray')  # 内部不显示颜色

                    if y == 2:  # 上面
                        face_colors.append(colors['top'])
                    else:
                        face_colors.append('gray')  # 内部不显示颜色

                    # 绘制每个外部小方块
                    draw_single_block(ax, x, y, z, face_colors)


# 绘制立方体
draw_cube()

# 设置坐标轴范围
ax.set_xlim([0, 3])
ax.set_ylim([0, 3])
ax.set_zlim([0, 3])

# 去掉坐标轴刻度
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()
