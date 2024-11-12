import numpy as np
import pyvista as pv


def create_tensor_grid(shape=(2, 3, 4)):
    """
    创建表示张量维度的3D网格

    参数:
        shape: 张量的形状，默认为(2, 3, 4)
    """
    # 创建网格 - 使用ImageData替代UniformGrid
    grid = pv.ImageData(
        dimensions=(shape[0] + 1, shape[1] + 1, shape[2] + 1),
        spacing=(1, 1, 1),
        origin=(0, 0, 0)
    )

    return grid


def visualize_tensor_3d(shape=(2, 3, 4), show_points=True, show_edges=True):
    """
    可视化3D张量的几何结构

    参数:
        shape: 张量的形状
        show_points: 是否显示网格点
        show_edges: 是否显示边框
    """
    # 创建plotter
    plotter = pv.Plotter()

    # 创建网格
    grid = create_tensor_grid(shape)

    # 添加网格到场景
    plotter.add_mesh(
        grid,
        show_edges=show_edges,
        opacity=0.5,
        color='lightblue',
        edge_color='blue'
    )

    # 可选显示网格点
    if show_points:
        # 创建表示张量元素的球体
        points = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    points.append([i, j, k])

        # 将所有点转换为PolyData
        point_cloud = pv.PolyData(np.array(points))
        # 使用glyph将球体放置在每个点上
        spheres = point_cloud.glyph(
            geom=pv.Sphere(radius=0.1),
            scale=False
        )
        plotter.add_mesh(spheres, color='red', opacity=1.0)

    # 添加坐标轴和标签
    plotter.add_axes()
    plotter.add_title(f"Tensor Shape: {shape[0]}x{shape[1]}x{shape[2]}")

    # 添加维度标签
    plotter.add_text(
        "Dim 0",
        position=(shape[0] / 2, -0.5, -0.5),
        font_size=12,
        color='black'
    )
    plotter.add_text(
        "Dim 1",
        position=(-0.5, shape[1] / 2, -0.5),
        font_size=12,
        color='black'
    )
    plotter.add_text(
        "Dim 2",
        position=(-0.5, -0.5, shape[2] / 2),
        font_size=12,
        color='black'
    )

    # 设置相机位置为等轴测视图
    plotter.view_isometric()

    # 设置背景颜色为白色
    plotter.set_background('white')

    return plotter


if __name__ == "__main__":
    # 创建并显示2x3x4的张量可视化
    tensor_shape = (2, 3, 4)
    plotter = visualize_tensor_3d(
        shape=tensor_shape,
        show_points=True,
        show_edges=True
    )
    plotter.show()

"""
可以运行。但是，numpy 与 pytorch 表示图象的张量有区别，分别是：(height,width,channel),(channel,height,width)，看起来不习惯。

"""