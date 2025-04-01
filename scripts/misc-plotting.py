'''
Misc. plotting script
'''

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from bayesian_wq_calibration.constants import TIMESERIES_DIR
import plotly.colors
default_colors = plotly.colors.qualitative.Plotly


mean_temp = []
N = 20
for data_period in range(1, N+1):
    wq_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(data_period).zfill(2)}-wq.csv", low_memory=False)
    temp = wq_df[wq_df['data_type'] == 'temperature']['mean'].mean()
    mean_temp.append(temp)


fig = go.Figure(data=[go.Bar(x=list(range(1, N+1)), y=mean_temp, marker_color=default_colors[0])])
fig.update_layout(
    xaxis_title="Data period",
    yaxis_title="Mean temperature [°C]",
    template="simple_white",
    width=800,
    height=500,
    xaxis=dict(
        tickfont=dict(size=32),
        titlefont=dict(size=32)
    ),
    yaxis=dict(
        tickfont=dict(size=32),
        titlefont=dict(size=32)
    )
)
fig.show()


