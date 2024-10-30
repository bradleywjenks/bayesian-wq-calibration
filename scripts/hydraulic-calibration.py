"""
This script calibrates the Field Lab's hydraulic model using flow and pressure data. Specifically, HW coefficients are calibrated based on a selected grouping. The following steps are performed:
    1. load hydraulic model
    2. load sensor data
    3. assign isolation valve loss coefficients
    4. assign boundary heads from BWFL 19 and Woodland Way PRV (inlet)
    5. scale demands based on flow balance (DMA or WWMD resolution)
    6. get control valve losses (PRV + DBV links)
    7. pipe grouping
    8. get pressure and flow data: time series + model ID
    9. calibrate HW coefficients via SCP algorithm

"""

import numpy as np
import pandas as pd
from bayesian_wq_calibration.epanet import model_simulation
from bayesian_wq_calibration.data import load_network_data, sensor_model_id
from bayesian_wq_calibration.constants import NETWORK_DIR, TIMESERIES_DIR, INP_FILE, IV_CLOSE, IV_OPEN, IV_OPEN_PART
import plotly.express as px
import plotly.graph_objects as go
import json
import warnings
warnings.filterwarnings("ignore")



###### STEP 1: load hydraulic model ######
wdn = load_network_data(NETWORK_DIR / INP_FILE)
with open(NETWORK_DIR / 'valve_info.json') as f:
    valve_info = json.load(f)
A12 = wdn.A12
A10 = wdn.A10
net_info = wdn.net_info
link_df = wdn.link_df
node_df = wdn.node_df
demand_df = wdn.demand_df
h0_df = wdn.h0_df
C0 = link_df['C'].values



###### STEP 2: load sensor data ######
data_period = 20 # change data period!!!
flow_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(data_period).zfill(2)}-flow.csv")
flow_device_id = sensor_model_id('flow')
pressure_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(data_period).zfill(2)}-pressure.csv")
pressure_device_id = sensor_model_id('pressure')
datetime = pd.to_datetime(flow_df['datetime'].unique())

# fig = px.line(
#     flow_df,
#     x='datetime',
#     y='mean',
#     color='bwfl_id',
# )
# fig.update_layout(
#     xaxis_title='',
#     yaxis_title='Flow [L/s]',
#     legend_title_text='',
#     template='simple_white',
#     height=450,
# )
# fig.show()

# fig = px.line(
#     pressure_df,
#     x='datetime',
#     y='mean',
#     color='bwfl_id',
# )
# fig.update_layout(
#     xaxis_title='',
#     yaxis_title='Pressure [m]',
#     legend_title_text='',
#     template='simple_white',
#     height=450,
# )
# fig.show()


###### STEP 3: assign isolation/dynamic boundary valve loss coefficients ######
iv_links = valve_info['iv_link']
iv_idx = link_df[link_df['link_ID'].isin(iv_links)].index
C0[iv_idx] = IV_CLOSE

dbv_links = valve_info['dbv_link']
dbv_idx = link_df[link_df['link_ID'].isin(dbv_links)].index
C_dbv = []
for idx, link in enumerate(dbv_links):
    start_node = link_df[link_df['link_ID'] == link]['node_out'].values[0]
    end_node = link_df[link_df['link_ID'] == link]['node_in'].values[0]
    link_diam = link_df[link_df['link_ID'] == link]['diameter'].values[0]

    start_bwfl_id = pressure_device_id[pressure_device_id['model_id'] == start_node]['bwfl_id'].values[0]
    start_pressure = pressure_df[pressure_df['bwfl_id'] == start_bwfl_id]['mean'].values
    end_bwfl_id = pressure_device_id[pressure_device_id['model_id'] == end_node]['bwfl_id'].values[0]
    end_pressure = pressure_df[pressure_df['bwfl_id'] == end_bwfl_id]['mean'].values

    try:
        link_bwfl_id = flow_device_id[flow_device_id['model_id'] == link]['bwfl_id'].values[0]
        link_flow = flow_df[flow_df['bwfl_id'] == link_bwfl_id]['mean'].values / 1000 # convert to cms
        dbv_value = (abs(start_pressure - end_pressure) * 2 * 9.81 * np.pi**2 * link_diam**4) / (abs(link_flow)**2 * 4**2)
        dbv_value = np.minimum(dbv_value, IV_CLOSE)
        dbv_value = np.maximum(dbv_value, IV_OPEN)
        C_dbv.append(dbv_value)
    except:
        C_dbv_default = np.where(((datetime.hour < 5) | (datetime.hour > 21)), IV_CLOSE, IV_OPEN_PART)
        C_dbv.append(C_dbv_default)

# fig = go.Figure()
# for i, dbv_values in enumerate(C_dbv):
#     fig.add_trace(go.Scatter(
#         x=datetime,
#         y=dbv_values,
#         mode='lines',
#         name=dbv_links[i]
#     ))
# fig.update_layout(
#     xaxis_title='',
#     yaxis_title='Time-varying DBV loss coefficient',
#     legend_title_text='',
#     template='simple_white',
#     height=450,
# )
# fig.show()

###### STEP 4: get boundary heads from BWFL 19 and Woodland Way PRV (inlet) + 5 ish m ######
reservoir_nodes = net_info['reservoir_names']
reservoir_idx = node_df[node_df['node_ID'].isin(reservoir_nodes)].index
h0 = []
for idx, node in enumerate(reservoir_nodes):
    bwfl_id = pressure_device_id[pressure_device_id['model_id'] == node]['bwfl_id'].values[0]
    elev = pressure_device_id[pressure_device_id['model_id'] == node]['elev'].values[0]
    h0_temp = pressure_df[pressure_df['bwfl_id'] == bwfl_id]['mean'].values + elev
    h0.append(np.round(h0_temp, 2))

# fig = go.Figure()
# for i, head_values in enumerate(h0):
#     fig.add_trace(go.Scatter(
#         x=datetime,
#         y=head_values,
#         mode='lines',
#         name=reservoir_nodes[i]
#     ))
# fig.update_layout(
#     xaxis_title='',
#     yaxis_title='Time-varying boundary heads [m]',
#     legend_title_text='',
#     template='simple_white',
#     height=450,
# )
# fig.show()



###### STEP 5: scale demands ######
# insert code here...



###### STEP 6: get PRV local loss (inlet - outlet) values ######
prv_links = valve_info['prv_link']
prv_dir = valve_info['prv_dir']
prv_idx = link_df[link_df['link_ID'].isin(prv_links)].index
eta = []
for idx, link in enumerate(prv_links):
    print(link)
    if prv_dir[idx] == 1:
        start_node = link_df[link_df['link_ID'] == link]['node_out'].values[0]
        end_node = link_df[link_df['link_ID'] == link]['node_in'].values[0]
    else:
        start_node = link_df[link_df['link_ID'] == link]['node_in'].values[0]
        end_node = link_df[link_df['link_ID'] == link]['node_out'].values[0]

    if link == 'link_2848': # change start_node to reservoir node for Woodland Way PRV inlet
        start_node = 'node_2860'

    if link == 'link_2912': # no inlet pressure for Coldharbour PRV; use average of BWFL 19 and Stoke Lane PRV (inlet) heads
        start_bwfl_id_1 = 'BWFL 19'
        inlet_head_1 = (
            pressure_df[pressure_df['bwfl_id'] == start_bwfl_id_1]['mean'].values +
            pressure_device_id[pressure_device_id['bwfl_id'] == start_bwfl_id_1]['elev'].values[0]
        )
        start_bwfl_id_2 = 'Stoke Lane PRV (inlet)'
        inlet_head_2 = (
            pressure_df[pressure_df['bwfl_id'] == start_bwfl_id_2]['mean'].values +
            pressure_device_id[pressure_device_id['bwfl_id'] == start_bwfl_id_2]['elev'].values[0]
        )
        avg_inlet_head = np.mean([inlet_head_1, inlet_head_2], axis=0)

        start_pressure = avg_inlet_head - node_df[node_df['node_ID'] == start_node]['elev'].values
        end_bwfl_id = pressure_device_id[pressure_device_id['model_id'] == end_node]['bwfl_id'].values[0]
        end_pressure = pressure_df[pressure_df['bwfl_id'] == end_bwfl_id]['mean'].values

    else:
        start_bwfl_id = pressure_device_id[pressure_device_id['model_id'] == str(start_node)]['bwfl_id'].values[0]
        print(start_bwfl_id)
        start_pressure = pressure_df[pressure_df['bwfl_id'] == start_bwfl_id]['mean'].values
        end_bwfl_id = pressure_device_id[pressure_device_id['model_id'] == end_node]['bwfl_id'].values[0]
        print(end_bwfl_id)
        end_pressure = pressure_df[pressure_df['bwfl_id'] == end_bwfl_id]['mean'].values

    eta.append(start_pressure - end_pressure)

fig = go.Figure()
for i, local_losses in enumerate(eta):
    fig.add_trace(go.Scatter(
        x=datetime,
        y=local_losses,
        mode='lines',
        name=prv_links[i]
    ))
fig.update_layout(
    xaxis_title='',
    yaxis_title=r'$\eta \;\, \text{[m]}$',
    legend_title_text='',
    template='simple_white',
    height=450,
)
fig.show()