import numpy as np
import plotly.graph_objs as go

x1k = 1 
x2k = -1
a1 = 1 
b1 = 0.5
a2 = -1 
b2 = 0.5

x1_values = np.linspace(-50, 50, 1000)
x2_values = np.linspace(-50, 50, 1000)
x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)

f = lambda x1, x2: a1 * (x1 - x1k) + b1 * (x1 - x1k)**2 + a2 * (x2 - x2k) + b2 * (x2 - x2k)**2
f_values = f(x1_grid, x2_grid)

fig = go.Figure(data=[go.Surface(
    x=x1_grid,
    y=x2_grid,
    z=f_values,
    colorscale='Viridis'
)])

fig.update_layout(
    title=f"3D Plot of f(x1, x2) with a1={a1}, b1={b1}, a2={a2}, b2={b2}",
    scene=dict(
        xaxis_title="x1",
        yaxis_title="x2",
        zaxis_title="f(x1, x2)"
    )
)

fig.show()


