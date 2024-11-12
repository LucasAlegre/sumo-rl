import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F


def create_tensor_grid(tensor):
    """
    创建表示张量维度的3D网格

    参数:
        tensor: PyTorch张量
    """
    shape = tensor.shape
    # 创建网格
    grid = pv.ImageData(
        dimensions=(shape[0] + 1, shape[1] + 1, shape[2] + 1),
        spacing=(1, 1, 1),
        origin=(0, 0, 0)
    )

    return grid


def normalize_tensor(tensor):
    """
    将张量值归一化到0-1范围，用于颜色映射
    """
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)


def visualize_tensor_3d(tensor, show_points=True, show_edges=True):
    """
    可视化3D PyTorch张量的几何结构和数值

    参数:
        tensor: PyTorch 3D张量
        show_points: 是否显示网格点
        show_edges: 是否显示边框
    """
    # 确保输入是3D张量
    assert len(tensor.shape) == 3, "输入必须是3D张量"
    shape = tensor.shape

    # 如果张量在GPU上，将其移到CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # 将张量值归一化到0-1范围
    normalized_values = normalize_tensor(tensor)

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

    # 显示网格点和张量值
    if show_points:
        # 创建所有点的坐标和对应的值
        points = []
        values = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    points.append([i, j, k])
                    values.append(normalized_values[i, j, k].item())

        # 将所有点转换为PolyData
        points = np.array(points, dtype=np.float32)  # 确保使用float32类型
        values = np.array(values, dtype=np.float32)  # 确保使用float32类型
        point_cloud = pv.PolyData(points)
        point_cloud['values'] = values

        # 使用值来调整球体大小和颜色
        min_radius, max_radius = 0.1, 0.2
        sphere = pv.Sphere(radius=1)

        # 修改glyph参数
        spheres = point_cloud.glyph(
            scale='values',  # 使用values进行缩放
            geom=sphere,  # 使用球体作为基础几何体
            factor=max_radius,  # 缩放因子
            orient=False,  # 不进行方向调整
        )

        # 使用值来设置颜色
        plotter.add_mesh(
            spheres,
            scalars='values',
            cmap='plasma',
            show_scalar_bar=True,
            scalar_bar_args={'title': 'Normalized Values'}
        )

    # 添加坐标轴和标签
    plotter.add_axes()
    plotter.add_title(f"Tensor Shape: {shape[0]}x{shape[1]}x{shape[2]}")

    # 添加维度标签
    plotter.add_text(
        f"Dim 0 (size: {shape[0]})",
        position=(shape[0] / 2, -0.5, -0.5),
        font_size=12,
        color='black'
    )
    plotter.add_text(
        f"Dim 1 (size: {shape[1]})",
        position=(-0.5, shape[1] / 2, -0.5),
        font_size=12,
        color='black'
    )
    plotter.add_text(
        f"Dim 2 (size: {shape[2]})",
        position=(-0.5, -0.5, shape[2] / 2),
        font_size=12,
        color='black'
    )

    # 设置相机位置为等轴测视图
    plotter.view_isometric()

    # 设置背景颜色为白色
    plotter.set_background('white')

    return plotter

def random_tensor():
    # 方法1：创建一个形状为(2,3,4)的随机张量
    tensor1 = torch.rand(2, 3, 4)
    print("tensor1: ", tensor1)



    # 可视化tensor1
    print("Visualizing random tensor:")
    plotter1 = visualize_tensor_3d(
        tensor1,
        show_points=True,
        show_edges=True
    )
    plotter1.show()

def wave_tensor():
    # 方法2：创建一个有规律的张量
    x, y, z = torch.meshgrid(
        torch.arange(2),
        torch.arange(3),
        torch.arange(4),
        indexing='ij'
    )
    tensor2 = torch.sin(x.float()) * torch.cos(y.float()) * torch.exp(-z.float() / 4)
    print("tensor2: ", tensor2)

    # 可视化tensor2
    print("\nVisualizing patterned tensor:")
    plotter2 = visualize_tensor_3d(
        tensor2,
        show_points=True,
        show_edges=True
    )
    plotter2.show()

if __name__ == "__main__":
    # 创建示例PyTorch张量
    random_tensor()
    wave_tensor()
