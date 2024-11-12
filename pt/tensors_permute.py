import torch
import numpy as np

# 1. 创建一个3维示例张量 (2,3,4)
# 第一维(2): 批次
# 第二维(3): 行数
# 第三维(4): 列数
tensor = torch.tensor([
    # 第一个批次
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]],

    # 第二个批次
    [[13, 14, 15, 16],
     [17, 18, 19, 20],
     [21, 22, 23, 24]]
])

print("原始张量形状:", tensor.shape)  # torch.Size([2, 3, 4])
print("原始张量内容:\n", tensor)

# 2. 基本维度重排示例
# permute(1,0,2): 交换第一维和第二维 : 前后翻转再交换上前后顺序
permuted_1 = tensor.permute(1, 0, 2)
print("\n交换批次维和行维后 (1,0,2):")
print("新形状:", permuted_1.shape)  # torch.Size([3, 2, 4])
print("内容:\n", permuted_1)

# 3. 循环移动维度示例
# permute(2,0,1): 将最后一维移到最前面：左右翻转再水平翻转
permuted_2 = tensor.permute(2, 0, 1)
print("\n将列维度移到最前面 (2,0,1):")
print("新形状:", permuted_2.shape)  # torch.Size([4, 2, 3])
print("内容:\n", permuted_2)

# 4. 实际应用场景示例：图像数据格式转换
# 创建一个模拟的图像批次张量 (batch_size, channels, height, width)
image_batch = torch.randn(2, 3, 64, 64)  # 2张RGB图片，每张64x64
print("\n原始图像批次形状:", image_batch.shape)  # [2, 3, 64, 64]

# 转换为 PyTorch 到 Numpy 格式 (batch_size, height, width, channels)
converted = image_batch.permute(0, 2, 3, 1)
print("转换后的图像形状:", converted.shape)  # [2, 64, 64, 3]

# 5. 多维张量的复杂重排
# 创建一个4维张量
tensor_4d = torch.arange(48).reshape(2, 3, 4, 2)
print("\n4维张量原始形状:", tensor_4d.shape)  # [2, 3, 4, 2]

# 复杂的维度重排
permuted_4d = tensor_4d.permute(3, 0, 2, 1)
print("复杂重排后形状:", permuted_4d.shape)  # [2, 2, 4, 3]

# 6. 验证数据完整性
# 检查permute操作前后的元素值是否保持不变
print("\n数据完整性验证:")
original_sum = tensor.sum().item()
permuted_sum = permuted_1.sum().item()
print("原始张量元素和:", original_sum)
print("重排后元素和:", permuted_sum)
print("元素和是否相等:", original_sum == permuted_sum)
