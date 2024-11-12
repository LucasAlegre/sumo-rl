import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 创建一个3D坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 定义立方体的8个顶点
vertices = [
    [0, 0, 0],  # 0
    [3, 0, 0],  # 1
    [3, 3, 0],  # 2
    [0, 3, 0],  # 3
    [0, 0, 3],  # 4
    [3, 0, 3],  # 5
    [3, 3, 3],  # 6
    [0, 3, 3],  # 7
]

# 定义立方体的6个面，每个面由4个顶点组成
faces = [
    [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
    [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
    [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
    [vertices[3], vertices[2], vertices[6], vertices[7]],  # 后面
    [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
    [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
]

# 设置各个面颜色
colors = ['green', 'blue', 'red', 'orange', 'white', 'yellow']

# 绘制并填充每个面
for i, face in enumerate(faces):
    poly3d = Poly3DCollection([face], facecolors=colors[i], linewidths=1, edgecolors='k', alpha=1)
    ax.add_collection3d(poly3d)

# 添加标签
ax.text(0.5, 0.5, 1.05, "Top", color="blue", fontsize=12)
ax.text(0.5, 0.5, -0.05, "Bottom", color="green", fontsize=12)
ax.text(-0.05, 0.5, 0.5, "Left", color="white", fontsize=12)
ax.text(1.05, 0.5, 0.5, "Right", color="yellow", fontsize=12)
ax.text(0.5, -0.05, 0.5, "Front", color="red", fontsize=12)
ax.text(0.5, 1.05, 0.5, "Back", color="orange", fontsize=12)

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置坐标轴范围
ax.set_xlim([0, 3])
ax.set_ylim([0, 3])
ax.set_zlim([0, 3])

# 设置整数刻度
ax.set_xticks([0, 1, 2, 3])
ax.set_yticks([0, 1, 2, 3])
ax.set_zticks([0, 1, 2, 3])

# 显示图形
plt.show()
