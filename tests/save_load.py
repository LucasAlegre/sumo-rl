"""
np.save("filename.npy",a)
b = np.load("filename.npy")
"""

import numpy as np

coeff = np.loadtxt("tests/coeff.txt", delimiter=" ")

dict1 = np.load("tests/data/2023-08-02 19:48:12-47.npy", allow_pickle=True)

print("dict1=", dict1)
print("type(dict1)=", type(dict1).__name__)

nn1 = np.cos(np.dot(np.pi * coeff, dict1))
print("nn1=", nn1)



dict2 = np.load("tests/data/2023-08-02 19:48:12-48.npy", allow_pickle=True)

print("dict2=", dict2)
print("type(dict2)=", type(dict2).__name__)

# nn2 = np.cos(np.dot(np.pi * coeff, dict2))
# print("nn2=", nn2)


a1 = np.array([1, 2, 3, 4], dtype=np.complex128)
print(a1)
print("数据类型", type(a1))  # 打印数组数据类型
print("数组元素数据类型：", a1.dtype)  # 打印数组元素数据类型
print("数组元素总数：", a1.size)  # 打印数组尺寸，即数组元素总数
print("数组形状：", a1.shape)  # 打印数组形状
print("数组的维度数目", a1.ndim)  # 打印数组的维度数目
print("数组的实部：", a1.real)  # 打印数组的实部