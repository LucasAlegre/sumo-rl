from mayavi import mlab
import numpy as np

# 创建网格
x, y = np.mgrid[-3:3:100j, -3:3:100j]
z = np.sin(x ** 2 + y ** 2)

# 绘制三维表面
mlab.mesh(x, y, z)

mlab.show()

"""
不能运行。
"""