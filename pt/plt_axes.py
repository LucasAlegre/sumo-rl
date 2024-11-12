import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制立方体的八个顶点
x = np.array([1, -1, -1, 1, 1, -1, -1, 1])
y = np.array([1, 1, -1, -1, 1, 1, -1, -1])
z = np.array([1, 1, 1, 1, -1, -1, -1, -1])

# 连接顶点绘制立方体的边
ax.plot([x[0], x[1]], [y[0], y[1]], [z[0], z[1]], color="b")
ax.plot([x[1], x[2]], [y[1], y[2]], [z[1], z[2]], color="b")
ax.plot([x[2], x[3]], [y[2], y[3]], [z[2], z[3]], color="b")
ax.plot([x[3], x[0]], [y[3], y[0]], [z[3], z[0]], color="b")

ax.plot([x[4], x[5]], [y[4], y[5]], [z[4], z[5]], color="b")
ax.plot([x[5], x[6]], [y[5], y[6]], [z[5], z[6]], color="b")
ax.plot([x[6], x[7]], [y[6], y[7]], [z[6], z[7]], color="b")
ax.plot([x[7], x[4]], [y[7], y[4]], [z[7], z[4]], color="b")

ax.plot([x[0], x[4]], [y[0], y[4]], [z[0], z[4]], color="b")
ax.plot([x[1], x[5]], [y[1], y[5]], [z[1], z[5]], color="b")
ax.plot([x[2], x[6]], [y[2], y[6]], [z[2], z[6]], color="b")
ax.plot([x[3], x[7]], [y[3], y[7]], [z[3], z[7]], color="b")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
