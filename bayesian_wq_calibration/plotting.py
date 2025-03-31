import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm, triang, uniform
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.colors
default_colors = plotly.colors.qualitative.Plotly
from bayesian_wq_calibration.epanet import sensor_model_id
from bayesian_wq_calibration.data import load_network_data
from bayesian_wq_calibration.calibration import get_observable_paths
from bayesian_wq_calibration.constants import NETWORK_DIR, DEVICE_DIR, INP_FILE, SPLIT_INP_FILE, RESULTS_DIR


wong_colors = [
    "rgb(0, 114, 178)",      # wong-blue
    "rgb(230, 159, 0)",      # wong-orange
    "rgb(0, 158, 115)",      # wong-green
    "rgb(0, 0, 0)",          # wong-black
    "rgb(86, 180, 233)",     # wong-skyblue
    "rgb(240, 228, 66)",     # wong-yellow
    "rgb(213, 94, 0)",       # wong-vermillion
    "rgb(204, 121, 167)"     # wong-purple
]



""""
    Main network plotting function
"""
def plot_network(reservoir=False, wq_sensors=False, flow_meters=False, pressure_sensors=False, prvs=False, dbvs=False, vals=None, val_type='pressure', t=0, inp_file='full', show_legend=False, fig_size=(600, 600)):

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
            size=15,
            color='blue',
            symbol='square'
        ),
        text=['inlet_2296', 'inlet_2005'],
        hoverinfo='text',
        name='DMA inlet'
    )

    if wq_sensors:
        sensor_data = sensor_model_id('wq')
        sensor_names = sensor_data['model_id'].values
        sensor_x = [pos[node][0] for node in sensor_names]
        sensor_y = [pos[node][1] for node in sensor_names]
        sensor_hydrant = [2, 5, 6]
        sensor_kiosk = [i for i in list(range(len(sensor_names))) if i not in sensor_hydrant]
        
        wq_trace_kiosk = go.Scatter(
            x=[sensor_x[i] for i in sensor_kiosk],
            y=[sensor_y[i] for i in sensor_kiosk],
            mode='markers',
            marker=dict(
                size=14,
                color='black',
                line=dict(color='white', width=2),
                symbol='square'
            ),
            text=[str(sensor_data['bwfl_id'][idx]) for idx in sensor_kiosk],
            hoverinfo='text',
            name='water quality sensor (kiosk)'
        )

        wq_trace_hydrant = go.Scatter(
            x=[sensor_x[i] for i in sensor_hydrant],
            y=[sensor_y[i] for i in sensor_hydrant],
            mode='markers',
            marker=dict(
                size=14,
                color='black',
                line=dict(color='white', width=2)
            ),
            text=[str(sensor_data['bwfl_id'][idx]) for idx in sensor_hydrant],
            hoverinfo='text',
            name='water quality sensor (hydrant)'
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
                    size=8,
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
                    size=14,
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

        elif val_type == 'chlorine':
            vals_df = vals.set_index('node_id')[f'cl_{t}']
            junction_vals = [vals_df[node] for node in net_info['junction_names']]
            reservoir_vals = [vals_df[node] for node in net_info['reservoir_names']]

            junction_nodes = net_info['junction_names']
            reservoir_nodes = net_info['reservoir_names']

            junction_x = [pos[node][0] for node in junction_nodes]
            junction_y = [pos[node][1] for node in junction_nodes]
            reservoir_x = [pos[node][0] for node in reservoir_nodes]
            reservoir_y = [pos[node][1] for node in reservoir_nodes]

            filtered_junctions = [
                (node, val, x, y) 
                for node, val, x, y in zip(junction_nodes, junction_vals, junction_x, junction_y)
                if val > 1e-2
            ]

            junction_nodes_filtered = [node for node, val, x, y in filtered_junctions]
            junction_vals_filtered = [val for node, val, x, y in filtered_junctions]
            junction_x_filtered = [x for node, val, x, y in filtered_junctions]
            junction_y_filtered = [y for node, val, x, y in filtered_junctions]

            min_val = 0
            # min_val = min(junction_vals_filtered + reservoir_vals)
            max_val = max(junction_vals_filtered + reservoir_vals)

            junction_val_trace = go.Scatter(
                x=junction_x_filtered,
                y=junction_y_filtered,
                mode='markers',
                marker=dict(
                    size=8,
                    color=junction_vals_filtered,
                    colorscale='RdYlBu',
                    cmin=min_val,
                    cmax=max_val,
                    colorbar=dict(
                        title="Chlorine [mg/L]",
                        titleside="right",
                        len=0.8,
                    ),
                    symbol='circle'
                ),
                text=[f"{node}<br>{val:.2f} mg/L" for node, val in zip(junction_nodes_filtered, junction_vals_filtered)],
                hoverinfo='text',
                name="junctions"
            )

            reservoir_val_trace = go.Scatter(
                x=reservoir_x,
                y=reservoir_y,
                mode='markers',
                marker=dict(
                    size=14,
                    color=reservoir_vals,
                    colorscale='RdYlBu',
                    cmin=min_val,
                    cmax=max_val,
                    symbol='square'
                ),
                text=[f"{node}<br>{val:.2f} mg/L" for node, val in zip(reservoir_nodes, reservoir_vals)],
                hoverinfo='text',
                name="reservoirs"
            )


    fig = go.Figure()

    # plot edges
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
        name='link',
    )
    fig.add_trace(edge_trace)

    if reservoir:
        fig.add_trace(reservoir_trace)

    if wq_sensors:
        fig.add_trace(wq_trace_kiosk)
        fig.add_trace(wq_trace_hydrant)

    if flow_meters:
        fig.add_trace(flow_trace)

    if pressure_sensors:
        fig.add_trace(pressure_trace)

    if vals is not None:
        if val_type == 'pressure' or val_type == 'chlorine':
            fig.add_trace(junction_val_trace)
            fig.add_trace(reservoir_val_trace)

    # update layout
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=fig_size[0],
        height=fig_size[1],
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(size=16),
        coloraxis_colorbar=dict(
            title=dict(font=dict(size=20)),
            tickfont=dict(size=18)
        )
    )

    fig.show()





def plot_network_features(feature_df, feature, observable=False, flow_df=None, wq_sensors_used='kiosk + hydrant', fig_size=(600, 600)):

    """
    plots network with features, optionally highlighting observable paths
    """
    
    # load network data
    wdn = load_network_data(NETWORK_DIR / INP_FILE)
    link_df = wdn.link_df
    node_df = wdn.node_df
    
    # create network graph
    uG = nx.from_pandas_edgelist(link_df, source='node_out', target='node_in')
    pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}
    
    fig = go.Figure()
    
    # get observable paths if requested
    if observable:
        if flow_df is None:
            raise ValueError("flow_df must be provided when observable=True")
        # observable_path = get_observable_paths(flow_df, link_df, wq_sensors_used)

        # feature_df_temp = link_df.copy()
        # feature_df_temp['observable_path'] = observable_path
        # feature_df_temp = feature_df_temp[feature_df_temp['link_type'] != 'valve']
        # feature_df = feature_df_temp.merge(feature_df, on='link_ID', how='left')
        observable_mask = feature_df['observable_path'].values
        
        # plot non-observable paths first (thin black lines)
        non_obs_df = feature_df[~observable_mask]
        edge_x, edge_y = [], []
        for _, link in non_obs_df.iterrows():
            x0, y0 = pos[link['node_out']]
            x1, y1 = pos[link['node_in']]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='grey'),
            hoverinfo='none',
            mode='lines',
            name='unobservable',
            showlegend=False
        ))
        
        # plot observable paths with features
        obs_df = feature_df[observable_mask]
        values = obs_df[feature].unique()
        # values = ['G1', 'G2', 'G3']
        sorted_values = sorted([x for x in values if x is not None], key=lambda x: int(x[1:])) 
        sorted_values.append(None)
        # color_map = {value: color for value, color in zip(sorted_values, plotly.colors.qualitative.Dark24)}
        color_map = {value: color for value, color in zip(sorted_values, wong_colors)}
        
        for group, color in color_map.items():
            group_df = obs_df[obs_df[feature] == group]
            edge_x, edge_y = [], []
            hover_text = []
            for _, link in group_df.iterrows():
                x0, y0 = pos[link['node_out']]
                x1, y1 = pos[link['node_in']]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                hover_text.extend([f"{feature}: {group}", f"{feature}: {group}", None])
                
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1.5, color=color),
                hoverinfo='text',
                text=hover_text,
                mode='lines',
                name=f'{group}'
            ))
    
    else:
        # plot all paths with features
        values = feature_df[feature].unique()
        # values = ['G1', 'G2', 'G3']
         # color_map = {value: color for value, color in zip(unique_values, plotly.colors.qualitative.Dark24)}
        color_map = {value: color for value, color in zip(values, wong_colors)}
        
        for group, color in color_map.items():
            group_df = feature_df[feature_df[feature] == group]
            edge_x, edge_y = [], []
            hover_text = []
            for _, link in group_df.iterrows():
                x0, y0 = pos[link['node_out']]
                x1, y1 = pos[link['node_in']]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                hover_text.extend([f"{feature}: {group}", f"{feature}: {group}", None])
                
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1.5, color=color),
                hoverinfo='text',
                text=hover_text,
                mode='lines',
                name=f'{group}'
            ))

    # Add water quality sensors based on wq_sensors_used
    sensor_data = sensor_model_id('wq')
    sensor_names = sensor_data['model_id'].values
    sensor_labels = sensor_data['bwfl_id'].values
    sensor_x = [pos[node][0] for node in sensor_names]
    sensor_y = [pos[node][1] for node in sensor_names]
    sensor_hydrant = [2, 5, 6]  
    sensor_kiosk = [i for i in range(len(sensor_names)) if i not in sensor_hydrant]
    sensor_remove = [6]
    sensor_all = [i for i in range(len(sensor_names)) if i not in sensor_remove]

    # Add sensor markers
    wq_trace_sensor = go.Scatter(
            x=[sensor_x[i] for i in sensor_all],
            y=[sensor_y[i] for i in sensor_all],
            mode='markers',
            marker=dict(
                size=18,
                color='black',
                line=dict(color='white', width=2),
                symbol='circle'
            ),
            text=[str(sensor_data['bwfl_id'][idx]) for idx in sensor_all],
            hoverinfo='text',
            name='sensor'
        )
    fig.add_trace(wq_trace_sensor)

    # Add annotations for sensor labels
    for idx in sensor_all:
        fig.add_annotation(
            x=sensor_x[idx],
            y=sensor_y[idx],
            text=sensor_labels[idx],  # Using the index value as the label
            showarrow=False,
            font=dict(
                size=18,
                color="black",
                # weight="bold"
            ),
            bgcolor="rgba(0,0,0,0)",  # Transparent background
            bordercolor="rgba(0,0,0,0)",
            xanchor="center",
            yanchor="middle",
            yshift=20
        )
    
    # if wq_sensors_used in ['kiosk only', 'kiosk + hydrant']:
    #     # plot kiosk sensors
    #     wq_trace_kiosk = go.Scatter(
    #         x=[sensor_x[i] for i in sensor_kiosk],
    #         y=[sensor_y[i] for i in sensor_kiosk],
    #         mode='markers',
    #         marker=dict(
    #             size=14,
    #             color='black',
    #             line=dict(color='white', width=2),
    #             symbol='square'
    #         ),
    #         text=[str(sensor_data['bwfl_id'][idx]) for idx in sensor_kiosk],
    #         hoverinfo='text',
    #         name='water quality sensor (kiosk)'
    #     )
    #     fig.add_trace(wq_trace_kiosk)

    # if wq_sensors_used in ['hydrant only', 'kiosk + hydrant']:
    #     # plot hydrant sensors
    #     wq_trace_hydrant = go.Scatter(
    #         x=[sensor_x[i] for i in sensor_hydrant],
    #         y=[sensor_y[i] for i in sensor_hydrant],
    #         mode='markers',
    #         marker=dict(
    #             size=14,
    #             color='black',
    #             line=dict(color='white', width=2)
    #         ),
    #         text=[str(sensor_data['bwfl_id'][idx]) for idx in sensor_hydrant],
    #         hoverinfo='text',
    #         name='water quality sensor (hydrant)'
    #     )
    #     fig.add_trace(wq_trace_hydrant)
    
    # update layout
    fig.update_layout(
        showlegend=True,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=fig_size[0],
        height=fig_size[1],
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(size=22),
        coloraxis_colorbar=dict(
            title=dict(font=dict(size=22)),
            tickfont=dict(size=20)
        ),
        legend=dict(
            x=0.9, 
            xanchor="left", 
            y=1, 
            yanchor="auto"  
        )
    )
        
    fig.show()
    return fig







def animate_network(cl_sim_T, datetime_vals, timesteps=None, frame_duration=100, concentration_interval=0.1, save_movie=False):
    """
    Create an animated plot of chlorine concentrations over time.
    
    Parameters:
    -----------
    cl_sim_T : pandas.DataFrame
        Transformed chlorine simulation results with columns ['node_id', 'cl_1', 'cl_2', ...]
    datetime_vals : array-like
        Array of datetime values corresponding to each timestep
    timesteps : list, optional
        List of timesteps to animate. If None, uses hourly timesteps
    frame_duration : int, optional
        Duration of each frame in milliseconds
    concentration_interval : float, optional
        Interval size for discrete chlorine concentration bins (default: 0.1 mg/L)
    """
    import plotly.graph_objects as go
    import networkx as nx
    import numpy as np
    
    # Get network data
    wdn = load_network_data(NETWORK_DIR / INP_FILE)
    link_df = wdn.link_df
    node_df = wdn.node_df
    net_info = wdn.net_info
    
    # Create network graph
    uG = nx.from_pandas_edgelist(link_df, source='node_out', target='node_in')
    pos = {row['node_ID']: (row['xcoord'], row['ycoord']) for _, row in node_df.iterrows()}
    
    # Create base figure
    fig = go.Figure()
    
    # Add static pipe network
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
        name='link',
        showlegend=False
    )
    fig.add_trace(edge_trace)
    
    # If timesteps not specified, select hourly timesteps
    if timesteps is None:
        all_timesteps = [int(col.split('_')[1]) for col in cl_sim_T.columns if col.startswith('cl_')]
        timesteps = all_timesteps[::4]  # Step by 4 to get hourly intervals (15min * 4 = 1 hour)
    
    # Index datetime_vals according to timesteps
    datetime_vals = datetime_vals[timesteps]
    
    # Get all chlorine columns for filtering
    chlorine_cols = [col for col in cl_sim_T.columns if col.startswith('cl_')]
    
    # Filter junction nodes based on maximum chlorine concentration across all timesteps
    junction_nodes = net_info['junction_names']
    significant_junctions = []
    for node in junction_nodes:
        max_cl = cl_sim_T[cl_sim_T['node_id'] == node][chlorine_cols].max().max()
        if max_cl > 1e-2:
            significant_junctions.append(node)
    
    # Get global max for color scale
    chlorine_cols = [f'cl_{t}' for t in timesteps]
    global_max = cl_sim_T[chlorine_cols].max().max()
    global_min = 0
    
    # Create discrete color bins
    n_bins = int(np.ceil(global_max / concentration_interval))
    tickvals = np.arange(0, n_bins + 1) * concentration_interval
    ticktext = [f"{val:.1f}" for val in tickvals]
    
    # Create frames for animation
    frames = []
    for t, dt in zip(timesteps, datetime_vals):
        # Get chlorine values for current timestep
        vals_df = cl_sim_T.set_index('node_id')[f'cl_{t}']
        
        # Process junction nodes
        junction_vals = [vals_df[node] for node in significant_junctions]
        junction_x = [pos[node][0] for node in significant_junctions]
        junction_y = [pos[node][1] for node in significant_junctions]
        
        # Process reservoir nodes
        reservoir_nodes = net_info['reservoir_names']
        reservoir_vals = [vals_df[node] for node in reservoir_nodes]
        reservoir_x = [pos[node][0] for node in reservoir_nodes]
        reservoir_y = [pos[node][1] for node in reservoir_nodes]
        
        frame = go.Frame(
            data=[
                edge_trace,
                go.Scatter(
                    x=junction_x,
                    y=junction_y,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=junction_vals,
                        colorscale='RdYlBu',
                        cmin=global_min,
                        cmax=global_max,
                        colorbar=dict(
                            title="Chlorine [mg/L]",
                            titleside="right",
                            len=0.8,
                            tickmode='array',
                            tickvals=tickvals,
                            ticktext=ticktext
                        ),
                        symbol='circle'
                    ),
                    text=[f"{node}<br>{val:.2f} mg/L" for node, val in zip(significant_junctions, junction_vals)],
                    hoverinfo='text',
                    name="junctions",
                    showlegend=False
                ),
                go.Scatter(
                    x=reservoir_x,
                    y=reservoir_y,
                    mode='markers',
                    marker=dict(
                        size=14,
                        color=reservoir_vals,
                        colorscale='RdYlBu',
                        cmin=global_min,
                        cmax=global_max,
                        symbol='square'
                    ),
                    text=[f"{node}<br>{val:.2f} mg/L" for node, val in zip(reservoir_nodes, reservoir_vals)],
                    hoverinfo='text',
                    name="reservoirs",
                    showlegend=False
                )
            ],
            name=f"frame_{t}"
        )
        frames.append(frame)
    
    # Add frames to figure
    fig.frames = frames
    
    # Create initial data (first frame)
    initial_junction_vals = [cl_sim_T.set_index('node_id')[f'cl_{timesteps[0]}'][node] for node in significant_junctions]
    initial_reservoir_vals = [cl_sim_T.set_index('node_id')[f'cl_{timesteps[0]}'][node] for node in reservoir_nodes]
    
    # Add initial junction and reservoir traces
    fig.add_trace(go.Scatter(
        x=[pos[node][0] for node in significant_junctions],
        y=[pos[node][1] for node in significant_junctions],
        mode='markers',
        marker=dict(
            size=8,
            color=initial_junction_vals,
            colorscale='RdYlBu',
            cmin=global_min,
            cmax=global_max,
            colorbar=dict(
                title=dict(
                    text="Chlorine [mg/L]",
                    side="right",
                    font=dict(size=16)
                ),
                len=0.8,
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext,
                tickfont=dict(size=14)
            ),
            symbol='circle'
        ),
        text=[f"{node}<br>{val:.2f} mg/L" for node, val in zip(significant_junctions, initial_junction_vals)],
        hoverinfo='text',
        name="junctions",
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=reservoir_x,
        y=reservoir_y,
        mode='markers',
        marker=dict(
            size=14,
            color=initial_reservoir_vals,
            colorscale='RdYlBu',
            cmin=global_min,
            cmax=global_max,
            symbol='square'
        ),
        text=[f"{node}<br>{val:.2f} mg/L" for node, val in zip(reservoir_nodes, initial_reservoir_vals)],
        hoverinfo='text',
        name="reservoirs",
        showlegend=False
    ))
    
    # Add slider and play button
    fig.update_layout(
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=600,
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=frame_duration, redraw=True), 
                                       fromcurrent=True, 
                                       mode='immediate')],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                         mode='immediate',
                                         transition=dict(duration=0))],
                    ),
                ],
                x=0.1,
                y=0,
            )
        ],
        sliders=[dict(
            steps=[
                dict(
                    method='animate',
                    args=[[f'frame_{k}'], dict(mode='immediate', frame=dict(duration=frame_duration, redraw=True))],
                    label=dt
                )
                for k, dt in zip(timesteps, datetime_vals)
            ],
            x=0.1,
            y=0,
            currentvalue=dict(
                font=dict(size=12),
                prefix="Datetime: ",
                visible=True,
                xanchor="right"
            ),
            len=1.0,
            tickwidth=24,  # Remove tick marks
            minorticklen=0,  # Remove minor ticks
            ticklen=0,  # Remove major ticks
        )]
    )
    
    fig.show()

    if save_movie:
        pio.write_html(fig, RESULTS_DIR / 'wq/cl_sim_animation.html')



'''
GP validation plotting
'''
def plot_gp_validation(Y_true, Y_pred, Y_std, times, sensor, plot_every_nth=1):
    """Plot GP predictions vs EPANET with uncertainty bounds."""
    fig = go.Figure()
    
    for exp_idx in range(0, len(Y_true), plot_every_nth):
        color = default_colors[exp_idx % len(default_colors)]
        
        fig.add_trace(go.Scatter(
            x=times, 
            y=Y_true[exp_idx],
            mode='lines', 
            name=f'EPANET (Exp {exp_idx + 1})',
            line=dict(color=color)
        ))
        
        fig.add_trace(go.Scatter(
            x=times, 
            y=Y_pred[exp_idx],
            mode='lines', 
            name=f'GP (Exp {exp_idx + 1})',
            line=dict(color=color, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=times.tolist() + times.tolist()[::-1],
            y=(Y_pred[exp_idx] + 2*Y_std[exp_idx]).tolist() + 
            (Y_pred[exp_idx] - 2*Y_std[exp_idx]).tolist()[::-1],
            fill='toself',
            fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.2])}',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'95% CI (Exp {exp_idx + 1})',
            showlegend=True
        ))
    
    fig.update_layout(
        title=f'GP Validation - Sensor {sensor}',
        xaxis_title='Time',
        yaxis_title='Chlorine [mg/L]',
        template='simple_white',
        height=600,
        legend=dict(
            # yanchor="top",
            # y=0.99,
            # xanchor="left",
            # x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    fig.show()



def plot_prior_distribution(samples, param_idx, dist_type, param_group, param_mean, param_bounds, bulk_uncertainty=0.1, wall_uncertainty=0.5):

    mu = param_mean[param_idx]
    sigma = abs(mu * bulk_uncertainty) if param_group[param_idx] == 'B' else abs(mu * wall_uncertainty)
    lower_bound, upper_bound = param_bounds[param_idx]
    param_samples = samples[:, param_idx]

    if param_idx == 0:
        x = np.linspace(lower_bound, upper_bound, 500)
        y = norm.pdf(x, loc=mu, scale=sigma)
    else:
        if dist_type == 'truncated normal':
            a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma
            x = np.linspace(lower_bound, upper_bound, 500)
            y = truncnorm.pdf(x, a=a, b=b, loc=mu, scale=sigma)
        elif dist_type == 'triangle':
            c = (mu - lower_bound) / (upper_bound - lower_bound)
            x = np.linspace(lower_bound, upper_bound, 500)
            y = triang.pdf(x, c=c, loc=lower_bound, scale=upper_bound - lower_bound)
        elif dist_type == 'uniform':
            x = np.linspace(lower_bound, upper_bound, 500)
            y = uniform.pdf(x, loc=lower_bound, scale=upper_bound - lower_bound)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='black', width=2), name='prior'))
    fig.add_trace(go.Scatter(x=param_samples, y=[0] * len(param_samples), mode='markers', marker=dict(size=6, color=default_colors[0], opacity=0.8), name='samples'))
    
    fig.update_layout(
        template="simple_white",
        width=500, height=400,
        showlegend=True,
        font=dict(size=20),
        # margin=dict(l=60, r=40, t=40, b=60),
        title=dict(text=param_group[param_idx], x=0.5),
        xaxis=dict(
            title="θ [m/d]",
            showline=True,
            linecolor='black',
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title="Density",
            showline=True,
            linecolor='black',
            showgrid=False,
            zeroline=False
        )
    )
    return fig

