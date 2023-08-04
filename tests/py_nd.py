import numpy as np

temp = np.zeros([2, 3, 4])
print(type(temp))
print("shape", temp.shape)  # 数组的维度
print("size", temp.size)  # 元素的个数
print("ndim", temp.ndim)  # 有多少个维度
print("len", temp.__len__())  # 数组第一维度的大小
print("len", len(temp))

print("temp", temp)

data = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]], [[17, 18, 19, 20], [21, 22, 23, 24]]])
print("data", data)
print("data.ndim", data.ndim)
print("data.shape", data.shape)
print("data.shape[0]", data.shape[0])
print("data.shape[data.ndim-1]", data.shape[data.ndim-1])
