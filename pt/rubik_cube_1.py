import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建一个3D坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 定义立方体的8个顶点
vertices = [
    [0, 0, 0],  # 0
    [1, 0, 0],  # 1
    [1, 1, 0],  # 2
    [0, 1, 0],  # 3
    [0, 0, 1],  # 4
    [1, 0, 1],  # 5
    [1, 1, 1],  # 6
    [0, 1, 1],  # 7
]

# 定义立方体的12条边
edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # 底面四条边
    [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面四条边
    [0, 4], [1, 5], [2, 6], [3, 7],  # 竖直边
]

# 绘制边
for edge in edges:
    ax.plot3D(*zip(*[vertices[edge[0]], vertices[edge[1]]]), color="b")

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
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

# 显示图形
plt.show()
