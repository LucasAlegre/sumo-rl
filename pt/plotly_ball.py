import plotly.graph_objs as go
import numpy as np

# 生成球体的参数方程
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])

fig.update_layout(title='3D Sphere', autosize=True, scene=dict(
    xaxis=dict(showbackground=False),
    yaxis=dict(showbackground=False),
    zaxis=dict(showbackground=False)
))

fig.show()
