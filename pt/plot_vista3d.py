import numpy as np
import pyvista as pv
import torch


def visualize_3d_tensor(tensor, title="3D Tensor Visualization"):
    """
    使用PyVista可视化3D张量数据

    参数:
        tensor: 3维numpy数组或PyTorch张量
        title: 可视化窗口标题
    """
    # 如果输入是PyTorch张量，转换为numpy数组
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # 创建网格点坐标
    x, y, z = np.meshgrid(
        np.arange(tensor.shape[0]),
        np.arange(tensor.shape[1]),
        np.arange(tensor.shape[2]),
        indexing='ij'
    )

    # 创建PyVista结构化网格
    grid = pv.StructuredGrid(x, y, z)

    # 将张量值添加为标量场
    grid.point_data["values"] = tensor.flatten(order='F')

    # 创建等值面 - 使用更新的API
    contours = grid.contour(
        isosurfaces=10,  # 指定等值面的数量
        scalars="values"
    )

    # 创建3D绘图
    plotter = pv.Plotter()
    plotter.add_mesh(
        contours,
        opacity=0.5,
        cmap="viridis",
        show_scalar_bar=True,
        scalar_bar_args={"title": "Tensor Values"}
    )

    # 添加坐标轴
    plotter.add_axes()
    plotter.add_title(title)

    # 设置视角
    plotter.view_isometric()

    return plotter


# 示例使用
if __name__ == "__main__":
    # 创建示例3D张量数据
    tensor_size = (20, 20, 20)
    center = np.array(tensor_size) / 2

    # 生成高斯分布的数据
    x, y, z = np.meshgrid(
        np.arange(tensor_size[0]),
        np.arange(tensor_size[1]),
        np.arange(tensor_size[2]),
        indexing='ij'
    )

    # 创建三维高斯分布
    sigma = 5.0
    tensor_data = np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2) / (2 * sigma ** 2))

    # 可视化张量
    plotter = visualize_3d_tensor(tensor_data, "3D Gaussian Distribution")
    plotter.show()