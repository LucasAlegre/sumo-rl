import vtk

# 创建立方体
cube = vtk.vtkCubeSource()

# 创建映射器
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(cube.GetOutputPort())

# 创建演员
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# 创建渲染器、渲染窗口和交互器
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# 添加演员到渲染器
renderer.AddActor(actor)

# 启动渲染
render_window.Render()
render_window_interactor.Start()

"""
可以运行。可以交互。

"""