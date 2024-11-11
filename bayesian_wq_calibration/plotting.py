import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.colors
default_colors = plotly.colors.qualitative.Plotly
from bayesian_wq_calibration.epanet import sensor_model_id
from bayesian_wq_calibration.data import load_network_data
from bayesian_wq_calibration.constants import NETWORK_DIR, DEVICE_DIR, INP_FILE, SPLIT_INP_FILE



""""
    Main network plotting function
"""
def plot_network(reservoir=False, wq_sensors=False, flow_meters=False, pressure_sensors=False, prvs=False, dbvs=False, grouping=None, vals=None, val_type='pressure', t=0, inp_file='full'):

    # unload data
    if inp_file == 'full':
        wdn = load_network_data(NETWORK_DIR / INP_FILE)
    elif inp_file == 'split':
        wdn = load_network_data(NETWORK_DIR / SPLIT_INP_FILE)
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
            size=6,
            color='grey',
            opacity=0.8
        ),
        text=list(uG.nodes),
        hoverinfo='text',
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
            size=16,
            color='black',
            symbol='square'
        ),
        text=['inlet_2296', 'inlet_2005'],
        hoverinfo='text',
        name='reservoir'
    )

    # water quality sensors
    if wq_sensors:
        sensor_data = sensor_model_id('wq')
        sensor_names = sensor_data['model_id'].values
        sensor_x = [pos[node][0] for node in sensor_names]
        sensor_y = [pos[node][1] for node in sensor_names]
        
        wq_trace = go.Scatter(
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
            name='water quality sensor'
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

    # pressure loggers
    if pressure_sensors:
        sensor_data = sensor_model_id('pressure')
        sensor_names = sensor_data['model_id'].values
        sensor_x = [pos[node][0] for node in sensor_names]
        sensor_y = [pos[node][1] for node in sensor_names]
        
        pressure_trace = go.Scatter(
            x=sensor_x,
            y=sensor_y,
            mode='markers',
            marker=dict(
                size=14,
                color='purple',
                line=dict(color='white', width=2)
            ),
            text=[str(sensor_data['bwfl_id'][idx]) for idx in range(len(sensor_names))],
            hoverinfo='text',
            name='pressure sensor'
        )


    # plot links
    if grouping == 'single':
        link_df['grouping'] = link_df['link_type'].apply(lambda x: 'valve' if x == 'valve' else 'single')
        group_colors = {
            'single': default_colors[1],
            'valve': 'black'
        }
    elif grouping == 'material':
        material_df = pd.read_excel(NETWORK_DIR / 'gis_data.xlsx')
        link_df = link_df.merge(material_df[['model_id', 'material']], left_on='link_ID', right_on='model_id', how='left')
        link_df = link_df.drop(columns=['model_id'])
        link_df = link_df.rename(columns={'material': 'grouping'})
        metallic = ['CI', 'SI', 'Pb', 'DI', 'ST']
        cement = ['AC']
        plastic_unknown = ['HPPE', 'HPPE+FOIL', 'LDPE', 'MDPE', 'MDPE+FOIL', 'PE100+Skin', 'PVC', 'Unknown']
        link_df['grouping'] = link_df['grouping'].apply(
            lambda x: 'metallic' if x in metallic 
            else 'cement' if x in cement 
            else 'plastic_unknown' if x in plastic_unknown 
            else 'valve'
        )
        group_colors = {
            'metallic': default_colors[1],
            'cement': default_colors[0],
            'plastic_unknown': default_colors[2],
            'valve': 'black'
        }
    elif grouping == 'material-diameter':
        material_df = pd.read_excel(NETWORK_DIR / 'gis_data.xlsx')
        link_df = link_df.merge(material_df[['model_id', 'material']], left_on='link_ID', right_on='model_id', how='left')
        link_df = link_df.drop(columns=['model_id'])
        link_df = link_df.rename(columns={'material': 'grouping'})
        metallic = ['CI', 'SI', 'Pb', 'DI', 'ST']
        cement = ['AC']
        plastic_unknown = ['HPPE', 'HPPE+FOIL', 'LDPE', 'MDPE', 'MDPE+FOIL', 'PE100+Skin', 'PVC', 'Unknown']
        link_df['grouping'] = link_df.apply(
        lambda row: (
            'metallic_less_than_150' if row['grouping'] in metallic and row['diameter'] * 1000 <= 150 else
            'metallic_greater_than_150' if row['grouping'] in metallic and row['diameter'] * 1000 > 150 else
            'cement' if row['grouping'] in cement else
            'plastic_unknown' if row['grouping'] in plastic_unknown else
            'valve'
        ), axis=1
        )
        group_colors = {
            'metallic_less_than_150': default_colors[1],
            'metallic_greater_than_150': default_colors[3],
            'cement': default_colors[0],
            'plastic_unknown': default_colors[2],
            'valve': 'black'
        }
    elif grouping == 'roughness':
        link_df['grouping'] = link_df.apply(
        lambda row: (
            'less_than_50' if row['link_type'] == 'pipe' and row['C'] < 50 else
            'between_50_and_65' if row['link_type'] == 'pipe' and row['C'] >= 50 and row['C'] < 65 else
            'between_65_and_80' if row['link_type'] == 'pipe' and row['C'] >= 65 and row['C'] < 80 else
            'between_80_and_100' if row['link_type'] == 'pipe' and row['C'] >= 80 and row['C'] < 100 else
            'between_100_and_120' if row['link_type'] == 'pipe' and row['C'] >= 100 and row['C'] < 120 else
            'greater_than_120' if row['link_type'] == 'pipe' and row['C'] >= 120 else
            'valve'
        ), axis=1
        )
        group_colors = {
            'less_than_50': default_colors[0],
            'between_50_and_65': default_colors[1],
            'between_65_and_80': default_colors[2],
            'between_80_and_100': default_colors[3],
            'between_100_and_120': default_colors[4],
            'greater_than_120': default_colors[5],
            'valve': 'black'
        }

    if vals is not None:
        if val_type == 'pressure':
            vals_df = vals.set_index('node_id')[f'p_{t}']
            junction_vals = [vals_df[node] for node in net_info['junction_names']]
            reservoir_vals = [0 for node in net_info['reservoir_names']]

            junction_nodes = net_info['junction_names']
            reservoir_nodes = net_info['reservoir_names']

            junction_x = [pos[node][0] for node in junction_nodes]
            junction_y = [pos[node][1] for node in junction_nodes]
            reservoir_x = [pos[node][0] for node in reservoir_nodes]
            reservoir_y = [pos[node][1] for node in reservoir_nodes]

            min_val = min(junction_vals + reservoir_vals)
            max_val = max(junction_vals + reservoir_vals)

            junction_val_trace = go.Scatter(
                x=junction_x,
                y=junction_y,
                mode='markers',
                marker=dict(
                    size=6,
                    color=junction_vals,
                    colorscale='RdYlBu',
                    cmin=min_val,
                    cmax=max_val,
                    colorbar=dict(
                        title="Pressure head [m]",
                        titleside="right"
                    ),
                    symbol='circle'
                ),
                text=[f"{node}<br>{val:.1f} m" for node, val in zip(junction_nodes, junction_vals)],
                hoverinfo='text',
                name="junctions"
            )

            reservoir_val_trace = go.Scatter(
                x=reservoir_x,
                y=reservoir_y,
                mode='markers',
                marker=dict(
                    size=16,
                    color=reservoir_vals,
                    colorscale='RdYlBu',
                    cmin=min_val,
                    cmax=max_val,
                    symbol='square'
                ),
                text=[f"{node}<br>{val:.1f} m" for node, val in zip(reservoir_nodes, reservoir_vals)],
                hoverinfo='text',
                name="reservoirs"
            )


    fig = go.Figure()

    if grouping is not None:
        for group, color in group_colors.items():
            uG = []
            uG = nx.from_pandas_edgelist(link_df[link_df['grouping'] == group], source='node_out', target='node_in')
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
                line=dict(width=1.0, color=color),
                hoverinfo='text',
                mode='lines',
                name=group
            )
            fig.add_trace(edge_trace)
        
        fig.update_layout(
            legend_title_text=f'{grouping} grouping'
        )
    
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
            line=dict(width=1.0, color='black'),
            hoverinfo='none',
            mode='lines',
            name='link'
        )
        fig.add_trace(edge_trace)

    if reservoir:
        fig.add_trace(reservoir_trace)

    if grouping is None and vals is None:
        fig.add_trace(node_trace)

    if wq_sensors:
        fig.add_trace(wq_trace)

    if flow_meters:
        fig.add_trace(flow_trace)

    if pressure_sensors:
        fig.add_trace(pressure_trace)

    if vals is not None:
        if val_type == 'pressure':
            fig.add_trace(junction_val_trace)
            fig.add_trace(reservoir_val_trace)
        show_legend = False
    else:
        show_legend = True

    fig.update_layout(
        showlegend=show_legend,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=600,
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    fig.show()