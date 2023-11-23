import numpy as np

data = []

for i in range(100):
    x = np.random.uniform(-10., 10.)
    eps = np.random.normal(0., 0.01)
    y = 1.477 * x + 0.089 + eps
    data.append([x, y])

data = np.array(data)
print(data)


def mse(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))

