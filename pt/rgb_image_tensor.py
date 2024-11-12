import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F


def create_tensor_grid(tensor):
    """
    创建表示张量维度的3D网格
    """
    shape = tensor.shape
    grid = pv.ImageData(
        dimensions=(shape[0] + 1, shape[1] + 1, shape[2] + 1),
        spacing=(1, 1, 1),
        origin=(0, 0, 0)
    )
    return grid


def normalize_tensor(tensor):
    """
    将张量值归一化到0-1范围
    """
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)


def rgb_to_scalar(rgb_values):
    """
    将RGB值转换为单一的标量值用于颜色映射
    使用亮度作为标量值
    """
    # RGB到亮度的转换系数
    weights = np.array([0.299, 0.587, 0.114])
    return np.dot(rgb_values, weights)


def visualize_rgb_tensor(tensor, show_points=True, show_edges=True):
    """
    可视化RGB图片张量

    参数:
        tensor: shape为(H, W, 3)的PyTorch张量，值在0-1之间
    """
    assert len(tensor.shape) == 3 and tensor.shape[2] == 3, "输入必须是HxWx3的RGB张量"
    shape = tensor.shape

    # 如果张量在GPU上，将其移到CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # 创建plotter
    plotter = pv.Plotter()

    # 创建网格
    grid = create_tensor_grid(tensor)

    # 添加网格到场景
    plotter.add_mesh(
        grid,
        show_edges=show_edges,
        opacity=0.3,
        color='lightgray',
        edge_color='gray'
    )

    if show_points:
        # 创建所有点的坐标和对应的RGB值
        points = []
        colors = []
        scalar_values = []

        for i in range(shape[0]):
            for j in range(shape[1]):
                points.append([i, j, 0])  # 所有点在z=0平面上
                rgb = tensor[i, j].numpy()
                colors.append(rgb)
                scalar_values.append(rgb_to_scalar(rgb))

        # 转换为numpy数组
        points = np.array(points, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        scalar_values = np.array(scalar_values, dtype=np.float32)

        # 创建点云
        point_cloud = pv.PolyData(points)
        point_cloud['scalars'] = scalar_values
        point_cloud['colors'] = colors

        # 创建球体
        sphere = pv.Sphere(radius=0.2)

        # 使用glyph创建球体表示
        spheres = point_cloud.glyph(
            scale='scalars',
            geom=sphere,
            factor=0.5,
            orient=False,
        )

        # 添加到场景中，使用实际的RGB颜色
        plotter.add_mesh(
            spheres,
            rgb=True,
            scalars='colors',
            show_scalar_bar=False
        )

    # 添加坐标轴和标签
    plotter.add_axes()
    plotter.add_title(f"RGB Image Tensor Shape: {shape[0]}x{shape[1]}x{shape[2]}")

    # 添加维度标签
    plotter.add_text(
        f"Height (size: {shape[0]})",
        position=(shape[0] / 2, -0.5, -0.5),
        font_size=12,
        color='black'
    )
    plotter.add_text(
        f"Width (size: {shape[1]})",
        position=(-0.5, shape[1] / 2, -0.5),
        font_size=12,
        color='black'
    )
    plotter.add_text(
        "RGB Channels",
        position=(-0.5, -0.5, shape[2] / 2),
        font_size=12,
        color='black'
    )

    plotter.view_xy()  # 设置为俯视图
    plotter.set_background('white')

    return plotter


if __name__ == "__main__":
    # 创建4x4 RGB测试图像张量
    # 方法1：创建彩色格子图案
    tensor = torch.zeros(4, 4, 3)

    # 创建一些有趣的颜色模式
    # 红色渐变
    tensor[:, :, 0] = torch.linspace(0, 1, 4).repeat(4, 1)
    # 绿色棋盘格
    tensor[::2, ::2, 1] = 1.0
    tensor[1::2, 1::2, 1] = 1.0
    # 蓝色圆形图案
    x, y = torch.meshgrid(torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 4), indexing='ij')
    radius = torch.sqrt(x ** 2 + y ** 2)
    tensor[:, :, 2] = torch.exp(-2 * radius)

    print("Visualizing 4x4 RGB tensor:")
    plotter = visualize_rgb_tensor(tensor, show_points=True, show_edges=True)
    plotter.show()

    # 为了更好地查看实际的图像效果，也可以使用matplotlib显示
    import matplotlib.pyplot as plt

    plt.imshow(tensor.numpy())
    plt.title("RGB Image")
    plt.axis('off')
    plt.show()
