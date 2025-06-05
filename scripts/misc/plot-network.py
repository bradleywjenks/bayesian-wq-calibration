import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots
import plotly.io as pio
from bayesian_wq_calibration.data import load_network_data
from bayesian_wq_calibration.constants import NETWORK_DIR, INP_FILE
import os

pio.kaleido.scope.mathjax = None

wong_colors = [
    "rgb(0, 114, 178)",      # wong-blue
    "rgb(230, 159, 0)",      # wong-orange
    "rgb(0, 158, 115)",      # wong-green
    "rgb(86, 180, 233)",     # wong-skyblue
    "rgb(240, 228, 66)",     # wong-yellow
    "rgb(213, 94, 0)",       # wong-vermillion
    "rgb(204, 121, 167)",     # wong-purple
    "rgb(128, 128, 128)"     # wong-grey
]


# get network data
wdn = load_network_data(NETWORK_DIR / INP_FILE)
link_df = wdn.link_df
link_df.rename(columns={'link_ID': 'link_id'}, inplace=True)
node_df = wdn.node_df
net_info = wdn.net_info

model_df = pd.read_excel(NETWORK_DIR / 'InfoWorks_to_EPANET_mapping.xlsx', sheet_name='Links')
model_df.rename(columns={
    'EPANET Link ID': 'link_id',
    'DMA ID': 'dma_id',
    'WWMD ID': 'wwmd_id'
}, inplace=True)
model_df = model_df[['link_id', 'dma_id', 'wwmd_id']]

link_df = link_df.merge(model_df[['link_id', 'dma_id', 'wwmd_id']], on='link_id', how='left')



""""
    Network plotting function
"""
def plot_network(link_df, node_df, net_info, colors=wong_colors, nodes=False, reservoirs=False, fig_size=(600, 600), save_plot=False, feature=None):

    # networkx data
    uG = nx.from_pandas_edgelist(link_df, source='node_out', target='node_in')
    pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}

    # get coordinates
    x_coords = [pos[node][0] for node in uG.nodes()]
    y_coords = [pos[node][1] for node in uG.nodes()]

    # junction nodes
    node_trace = go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        marker=dict(
            size=6,
            color=colors[0],
        ),
        text=list(uG.nodes),
        hoverinfo='none',
        name='junction'
    )

    # reservoir nodes
    reservoir_nodes = net_info['reservoir_names']
    reservoir_x = [pos[node][0] for node in reservoir_nodes]
    reservoir_y = [pos[node][1] for node in reservoir_nodes]

    reservoir_trace = go.Scatter(
        x=reservoir_x,
        y=reservoir_y,
        mode='markers',
        marker=dict(
            size=15,
            color=colors[1],
            symbol='square'
        ),
        text=['inlet_2296', 'inlet_2005'],
        hoverinfo='none',
        name='DMA inlet'
    )

    fig = go.Figure()

    if feature is not None and feature in link_df.columns:
        feature_values = link_df[feature].unique()
        color_map = {val: colors[i % len(colors)] for i, val in enumerate(feature_values)}

        # plot links by feature group
        for val in feature_values:
            sub_links = link_df[link_df[feature] == val]
            edge_x = []
            edge_y = []
            for _, row in sub_links.iterrows():
                x0, y0 = pos[row['node_out']]
                x1, y1 = pos[row['node_in']]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=1.0, color=color_map[val]),
                hoverinfo='text',
                mode='lines',
                name=str(val),
            )
            fig.add_trace(edge_trace)

    else:
        edge_x = []
        edge_y = []
        for edge in uG.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1.0, color=colors[3]),
            hoverinfo='none',
            mode='lines',
            name='link',
        )
        fig.add_trace(edge_trace)

    if nodes:
        fig.add_trace(node_trace)
    if reservoirs:
        fig.add_trace(reservoir_trace)

    # layout
    fig.update_layout(
        showlegend=False,
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=fig_size[0],
        height=fig_size[1],
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(size=16),
    )

    if save_plot:
        output_path = os.path.join(os.path.dirname(__file__), "network_plot.pdf")
        pio.write_image(fig, output_path, format='pdf')
        print(f"Figure saved to: {output_path}")

    else:
        fig.show()




# plot network
plot_network(link_df, node_df, net_info, feature='wwmd_id', nodes=False, reservoirs=False, fig_size=(600, 600), save_plot=True)