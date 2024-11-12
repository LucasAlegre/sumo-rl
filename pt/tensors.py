from torchvision import transforms
from PIL import Image

# 读取图像
img = Image.open('./pt/red-girl.png')

# 创建 ToTensor 转换对象
transform = transforms.ToTensor()

# 转换图像为 Tensor
tensor_img = transform(img)

# 查看 Tensor 的形状和数值范围
print(tensor_img.shape)  # 输出形状: torch.Size([C, H, W])
print(tensor_img.min(), tensor_img.max())  # 输出范围: torch.Size([0.0, 1.0])
print(tensor_img)