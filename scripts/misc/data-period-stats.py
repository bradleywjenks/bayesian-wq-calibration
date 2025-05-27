'''
Misc. plotting script
'''

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from bayesian_wq_calibration.constants import TIMESERIES_DIR
import plotly.colors
# default_colors = plotly.colors.qualitative.Plotly
wong_colors = [
    "rgb(89, 89, 89)",       # wong-black/grey
    "rgb(230, 159, 0)",      # wong-orange
    "rgb(86, 180, 233)",     # wong-skyblue
    "rgb(0, 158, 115)",      # wong-green
    "rgb(240, 228, 66)",     # wong-yellow
    "rgb(0, 114, 178)",      # wong-blue
    "rgb(213, 94, 0)",       # wong-vermillion
    "rgb(204, 121, 167)"     # wong-purple
]

# number of data periods
N = 22

# temperature statistics
mean_temp = []
for data_period in range(1, N+1):
    wq_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(data_period).zfill(2)}-wq.csv", low_memory=False)
    temp = wq_df[wq_df['data_type'] == 'temperature']['mean'].mean()
    mean_temp.append(temp)


fig = go.Figure(data=[go.Bar(x=list(range(1, N+1)), y=mean_temp, marker_color=wong_colors[2])])
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


# flow statistics
mean_flow = []
for data_period in range(1, N+1):
    flow_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(data_period).zfill(2)}-flow.csv", low_memory=False)
    pivot_df = flow_df.pivot_table(index='datetime', columns='bwfl_id', values='mean')
    mean_flow.append(pivot_df[['inlet_2005', 'inlet_2296']].sum(axis=1).mean())

# Create bar plot of mean flows
fig = go.Figure(data=[go.Bar(x=list(range(1, N+1)), y=mean_flow, marker_color=wong_colors[1])])
fig.update_layout(
    xaxis_title="Data period",
    yaxis_title="Mean total flow [L/s]",
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


