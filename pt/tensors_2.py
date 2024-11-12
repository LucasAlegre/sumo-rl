import torch
import numpy as np

# 创建一个形状为(2,3,4)的3维张量
tensor3d = torch.tensor([
    # 第一个2D切片 (第0个batch)
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]],

    # 第二个2D切片 (第1个batch)
    [[13, 14, 15, 16],
     [17, 18, 19, 20],
     [21, 22, 23, 24]]
])

print("\n张量:\n",tensor3d)
print("\n张量形状:", tensor3d.shape)  # torch.Size([2, 3, 4])
print("\n张量维度:", tensor3d.dim())

# 访问不同维度的元素
print("\n维度访问示例:")
print("第0个2D切片:\n", tensor3d[0])  # 第一个2D矩阵
print("\n第0个2D切片的第1行:\n", tensor3d[0][1])  # 第一个矩阵的第二行
print("\n位置(0,1,2)的元素:", tensor3d[0, 1, 2])  # 值为7

# 维度切片操作
print("\n维度切片示例:")
print("第一维的切片:\n", tensor3d[0, :, :])  # 等同于tensor3d[0]
print("\n第二维的切片:\n", tensor3d[:, 0, :])  # 所有batch的第一行
print("\n第三维的切片:\n", tensor3d[:, :, 0])  # 所有位置的第一个元素

# 展示维度变换
print("\n维度变换示例:")
# 交换维度
permuted = tensor3d.permute(2, 0, 1)
print("维度交换后的形状:", permuted.shape)  # torch.Size([4, 2, 3])
print("\n张量permuted:\n",permuted)

# 重新排列维度
reshaped = tensor3d.reshape(6, 4)
print("重排后的形状:", reshaped.shape)  # torch.Size([6, 4])
print("\nreshaped:\n",reshaped)

# 增加维度
expanded = tensor3d.unsqueeze(0)
print("增加维度后的形状:", expanded.shape)  # torch.Size([1, 2, 3, 4])
print("\nexpanded:\n",expanded)
