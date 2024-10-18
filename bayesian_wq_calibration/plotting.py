import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from bayesian_wq_calibration.simulation import sensor_model_id
from bayesian_wq_calibration.data import load_network_data
from bayesian_wq_calibration.constants import NETWORK_DIR, DEVICE_DIR, INP_FILE



""""
    Main network plotting function
"""
def plot_network(wq_sensors=True, flow_meters=True, prvs=False, dbvs=False):

    # unload data
    wdn = load_network_data(NETWORK_DIR / INP_FILE)
    link_df = wdn.link_df
    node_df = wdn.node_df
    net_info = wdn.net_info

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
            size=7,
            color='grey',
            opacity=1
        ),
        text=list(uG.nodes),
        hoverinfo='text',
        name='Junction'
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
            size=18,
            color='black',
            symbol='square'
        ),
        text=['inlet_2296', 'inlet_2005'],
        hoverinfo='text',
        name='Reservoir'
    )

    # water quality sensors
    if wq_sensors:
        sensor_data = sensor_model_id('wq')
        sensor_names = sensor_data['model_id'].values
        sensor_x = [pos[node][0] for node in sensor_names]
        sensor_y = [pos[node][1] for node in sensor_names]
        
        sensor_trace = go.Scatter(
            x=sensor_x,
            y=sensor_y,
            mode='markers',
            marker=dict(
                size=14,
                color='red',
                line=dict(color='white', width=2)
            ),
            text=[str(sensor_data['bwfl_id'][idx]) for idx in range(len(sensor_names))],
            hoverinfo='text',
            name='Water quality sensor'
        )


    # flow meters
    if flow_meters:
        flow_data = sensor_model_id('flow')
        bwfl_ids = ['inlet_2296', 'inlet_2005', 'Snowden Road DBV', 'New Station Way DBV']
        flow_names = [flow_data[flow_data['bwfl_id'] == name]['model_id'].values[0] for name in bwfl_ids]
        flow_position = link_df.loc[link_df['link_ID'].isin(flow_names), 'node_in'].tolist()
        flow_x = [pos[node][0] for node in flow_position]
        flow_y = [pos[node][1] for node in flow_position]

        flow_trace = go.Scatter(
            x=flow_x,
            y=flow_y,
            mode='markers',
            marker=dict(
                size=14,
                color='blue',
                line=dict(color='white', width=2),
                symbol='diamond'
            ),
            text=['inlet_2296', 'Snowden Road DBV', 'New Station Way DBV', 'inlet_2005'],
            hoverinfo='text',
            name='DMA flow meter'
        )

    # plot links
    edge_x = []
    edge_y = []
    for edge in uG.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])  # None creates breaks between line segments
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.0, color='black'),
        hoverinfo='none',
        mode='lines',
        name='Link'
    )

    fig = go.Figure(data=[edge_trace, node_trace, reservoir_trace])

    if wq_sensors:
        fig.add_trace(sensor_trace)

    if flow_meters:
        fig.add_trace(flow_trace)

    fig.update_layout(
        showlegend=True,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=650,
        height=750,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    fig.show()

    # fig, ax = plt.subplots(figsize=(5.5, 8))
    # ax.margins(0.025, 0.025)

    # # draw network
    # uG = nx.from_pandas_edgelist(link_df, source='node_out', target='node_in')
    # pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}
    # nx.draw(uG, pos, node_size=20, node_shape='o', alpha=0.5, node_color='grey', ax=ax)

    # # draw reservoir
    # nx.draw_networkx_nodes(uG, pos, nodelist=net_info['reservoir_names'], node_size=150, node_shape='s', node_color='black', ax=ax)
    # reservoir_labels = {'node_2859': 'inlet_2296', 'node_2860': 'inlet_2005'}
    # labels_res = nx.draw_networkx_labels(uG, pos, reservoir_labels, font_size=12, verticalalignment='top')
    # for _, label in labels_res.items():
    #     label.set_y(label.get_position()[1] - 100)

    # # draw sensor nodes
    # if wq_sensors:
    #     sensor_data = sensor_model_id('wq')
    #     sensor_names = sensor_data['model_id'].values
    #     nx.draw_networkx_nodes(uG, pos, sensor_names, node_size=100, node_shape='o', node_color='orange', ax=ax)

    #     sensor_labels = {node: str(sensor_data['bwfl_id'][idx]) for (idx, node) in enumerate(sensor_names)}
    #     labels_sen = nx.draw_networkx_labels(uG, pos, sensor_labels, font_size=12, verticalalignment='bottom', ax=ax)
    #     for _, label in labels_sen.items():
    #         label.set_y(label.get_position()[1] + 80)