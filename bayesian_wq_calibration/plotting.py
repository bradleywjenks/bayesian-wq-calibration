import networkx as nx
import numpy as np
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
def plot_network(reservoir=False, wq_sensors=False, flow_meters=False, pressure_sensors=False, prvs=False, dbvs=False, vals=None, val_type='pressure', t=0, inp_file='full', feature_df=None, feature=None, show_legend=False, fig_size=(600, 600)):

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

    if feature is not None:
        if feature in feature_df.columns:
            unique_values = feature_df[feature].unique()

            color_map = {value: color for value, color in zip(unique_values, plotly.colors.qualitative.Dark24)}

            for group, color in color_map.items():
                group_df = feature_df[feature_df[feature] == group]
                uG_group = nx.from_pandas_edgelist(group_df, source='node_out', target='node_in')

                edge_x = []
                edge_y = []
                hover_text = []
                for edge in uG_group.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(width=1.0, color=color),
                    hoverinfo='text',
                    text=f"{feature}: {group}",
                    mode='lines',
                    name=f'{group}'
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

    fig.update_layout(
        showlegend=show_legend,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=fig_size[0],
        height=fig_size[1],
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    fig.show()







def animate_network(cl_sim_T, datetime_vals, timesteps=None, frame_duration=100, concentration_interval=0.1):
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
                title="Chlorine [mg/L]",
                titleside="right",
                len=0.8,
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext
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
            tickwidth=0,  # Remove tick marks
            minorticklen=0,  # Remove minor ticks
            ticklen=0,  # Remove major ticks
        )]
    )
    
    fig.show()

