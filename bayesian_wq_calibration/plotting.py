import networkx as nx
import matplotlib.pyplot as plt
from bayesian_wq_calibration.simulation import sensor_model_id
from bayesian_wq_calibration.data import load_network_data
from bayesian_wq_calibration.constants import NETWORK_DIR, DEVICE_DIR, INP_FILE



""""
    Main network plotting function
"""
def plot_network(wq_sensors=True, flow_sensors=True, prvs=False, dbvs=False):

    # unload data
    wdn = load_network_data(NETWORK_DIR / INP_FILE)
    link_df = wdn.link_df
    node_df = wdn.node_df
    net_info = wdn.net_info

    fig, ax = plt.subplots(figsize=(5.5, 8))
    ax.margins(0.025, 0.025)

    # draw network
    uG = nx.from_pandas_edgelist(link_df, source='node_out', target='node_in')
    pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}
    nx.draw(uG, pos, node_size=20, node_shape='o', alpha=0.5, node_color='grey', ax=ax)

    # draw reservoir
    nx.draw_networkx_nodes(uG, pos, nodelist=net_info['reservoir_names'], node_size=150, node_shape='s', node_color='black', ax=ax)
    reservoir_labels = {'node_2859': 'inlet_2296', 'node_2860': 'inlet_2005'}
    labels_res = nx.draw_networkx_labels(uG, pos, reservoir_labels, font_size=12, verticalalignment='top')
    for _, label in labels_res.items():
        label.set_y(label.get_position()[1] - 100)

    # draw sensor nodes
    if wq_sensors:
        sensor_data = sensor_model_id('wq')
        sensor_names = sensor_data['model_id'].values
        nx.draw_networkx_nodes(uG, pos, sensor_names, node_size=100, node_shape='o', node_color='orange', ax=ax)

        sensor_labels = {node: str(sensor_data['bwfl_id'][idx]) for (idx, node) in enumerate(sensor_names)}
        labels_sen = nx.draw_networkx_labels(uG, pos, sensor_labels, font_size=12, verticalalignment='bottom', ax=ax)
        for _, label in labels_sen.items():
            label.set_y(label.get_position()[1] + 80)