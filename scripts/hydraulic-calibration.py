"""
This script calibrates the Field Lab's hydraulic model using flow and pressure data. Specifically, HW coefficients are calibrated based on a selected grouping. The following steps are performed:
    1. load hydraulic model
    2. load sensor data
    3. assign isolation valve loss coefficients
    4. assign boundary heads from BWFL 19 and Woodland Way PRV (inlet)
    5. scale demands based on flow balance (DMA or WWMD resolution)
    6. get control valve losses (PRV + DBV links)
    7. split test/train data for h0, d, C_dbv, and eta
    8. get pressure and flow data: time series + model ID and check initial model performance
    9. compute initial hydraulics and plot residuals
    10. get pipe grouping
    11. calibrate HW coefficients via SCP algorithm

"""

import numpy as np
import pandas as pd
from bayesian_wq_calibration.data import load_network_data, sensor_model_id
from bayesian_wq_calibration.constants import NETWORK_DIR, TIMESERIES_DIR, INP_FILE, IV_CLOSE, IV_OPEN, IV_OPEN_PART
from bayesian_wq_calibration.simulation import hydraulic_solver
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
C_0 = link_df['C'].values





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
C_0[iv_idx] = IV_CLOSE

dbv_links = valve_info['dbv_link']
dbv_idx = link_df[link_df['link_ID'].isin(dbv_links)].index
C_dbv = np.zeros((len(dbv_idx), len(datetime)))
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
        C_dbv[idx, :] = dbv_value
    except:
        C_dbv_default = np.where(((datetime.hour < 5) | (datetime.hour > 21)), IV_CLOSE, IV_OPEN_PART)
        C_dbv[idx, :] = C_dbv_default

# fig = go.Figure()
# for idx in range(C_dbv.shape[0]):
#     fig.add_trace(go.Scatter(
#         x=datetime,
#         y=C_dbv[idx, :],
#         mode='lines',
#         name=dbv_links[idx]
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
h0 = np.zeros((len(reservoir_idx), len(datetime)))
for idx, node in enumerate(reservoir_nodes):
    bwfl_id = pressure_device_id[pressure_device_id['model_id'] == node]['bwfl_id'].values[0]
    elev = pressure_device_id[pressure_device_id['model_id'] == node]['elev'].values[0]
    h0_temp = pressure_df[pressure_df['bwfl_id'] == bwfl_id]['mean'].values + elev
    h0[idx, :] = np.round(h0_temp, 2)

# fig = go.Figure()
# for idx in range(h0.shape[0]):
#     fig.add_trace(go.Scatter(
#         x=datetime,
#         y=h0[idx, :],
#         mode='lines',
#         name=reservoir_nodes[idx]
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
num_days = int(len(datetime) / 96)
node_mapping = pd.read_excel(NETWORK_DIR / 'InfoWorks_to_EPANET_mapping.xlsx', sheet_name='Nodes')

d0 = demand_df.iloc[:, 1:].values
repeated_d0 = np.tile(d0, (1, num_days))
new_column_names = [f"demands_{i}" for i in range(1, repeated_d0.shape[1]+1)]
repeated_d0_df = pd.DataFrame(repeated_d0, columns=new_column_names)
repeated_d0_df['node_id'] = demand_df['node_ID']
repeated_d0_df['dma_id'] = None
repeated_d0_df['wwmd_id'] = None
for node in demand_df['node_ID']:
    repeated_d0_df.loc[repeated_d0_df['node_id'] == node, 'dma_id'] = str(node_mapping[node_mapping['EPANET Node ID'] == node]['DMA ID'].values[0])
    repeated_d0_df.loc[repeated_d0_df['node_id'] == node, 'wwmd_id'] = str(node_mapping[node_mapping['EPANET Node ID'] == node]['WWMD ID'].values[0])

columns_order = ['node_id', 'dma_id', 'wwmd_id'] + new_column_names
repeated_d0_df = repeated_d0_df[columns_order]

demand_resolution = 'wwmd' # 'wwmd' or 'dma'
inflow_df = pd.DataFrame({'datetime': datetime}).set_index('datetime')
has_nan = True
while has_nan:
    if demand_resolution == 'dma':
        with open(NETWORK_DIR / 'flow_balance_dma.json') as f:
            flow_balance = json.load(f)
    elif demand_resolution == 'wwmd':
        with open(NETWORK_DIR / 'flow_balance_wwmd.json') as f:
            flow_balance = json.load(f)

    for key, values in flow_balance.items():
        inflow_sum = pd.Series(0, index=inflow_df.index)
        for sensor in values['flow_in']:
            inflow_sum += flow_df[flow_df['bwfl_id'] == sensor]['mean'].values
        outflow_sum = pd.Series(0, index=inflow_df.index)
        for sensor in values['flow_out']:
            outflow_sum += flow_df[flow_df['bwfl_id'] == sensor]['mean'].values
        
        inflow_df[key] = inflow_sum - outflow_sum
        has_nan = inflow_df[key].isna().any()
        if has_nan:
            demand_resolution = 'dma'
            print(f"Significant periods of missing data for {demand_resolution} demand resolution. Switching to {demand_resolution} demand resolution")
            break

scaled_demand_df = repeated_d0_df.copy()
for key in flow_balance.keys():
    if demand_resolution == 'dma':
        filtered_rows = repeated_d0_df[repeated_d0_df['dma_id'] == str(key)].index
    elif demand_resolution == 'wwmd':
        filtered_rows = repeated_d0_df[repeated_d0_df['wwmd_id'] == str(key)].index

    total_base_demand = repeated_d0_df.loc[filtered_rows, repeated_d0_df.columns[3:]].sum(axis=0)
    scaling_factor = inflow_df[key].values / total_base_demand.values

    for idx in range(total_base_demand.shape[0]):
        demand_col_name = f'demands_{idx+1}'
        scaled_demand_df.loc[filtered_rows, demand_col_name] = repeated_d0_df.loc[filtered_rows, demand_col_name] * scaling_factor[idx]

    total_scaled_demand = scaled_demand_df.loc[filtered_rows, scaled_demand_df.columns[3:]].sum(axis=0).values
    total_inflow = inflow_df[key].values

    assert np.isclose(total_scaled_demand.sum(), total_inflow.sum()), f"Mismatch for zone {key}: {total_scaled_demand.sum()} != {total_inflow.sum()}"

d = scaled_demand_df.iloc[:, 3:].values / 1000 # back to cms





###### STEP 6: get PRV local loss (inlet - outlet) values ######
prv_links = valve_info['prv_link']
prv_dir = valve_info['prv_dir']
prv_idx = link_df[link_df['link_ID'].isin(prv_links)].index
eta = np.zeros((len(prv_idx), len(datetime)))
for idx, link in enumerate(prv_links):
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
        start_pressure = pressure_df[pressure_df['bwfl_id'] == start_bwfl_id]['mean'].values
        end_bwfl_id = pressure_device_id[pressure_device_id['model_id'] == end_node]['bwfl_id'].values[0]
        end_pressure = pressure_df[pressure_df['bwfl_id'] == end_bwfl_id]['mean'].values

    eta[idx, :] = start_pressure - end_pressure

# fig = go.Figure()
# for idx in range(eta.shape[0]):
#     fig.add_trace(go.Scatter(
#         x=datetime,
#         y=eta[idx, :],
#         mode='lines',
#         name=prv_links[idx]
#     ))
# fig.update_layout(
#     xaxis_title='',
#     yaxis_title=r'$\eta \;\, \text{[m]}$',
#     legend_title_text='',
#     template='simple_white',
#     height=450,
# )
# fig.show()





###### Step 7: split train/test data ######
n_total = len(datetime)
n_train = 7 * 24 * 4

train_range = range(n_train)
test_range = range(n_train, n_total)





###### Step 8: get sensor data ######
# pressure devices
not_these_pressure_ids = ['BWFL 19', 'Woodland Way PRV (inlet)', 'Woodland Way PRV (outlet)', 'Lodge Causeway PRV (inlet)', 'Lodge Causeway PRV (outlet)', 'Stoke Lane PRV (inlet)', 'Stoke Lane PRV (outlet)', 'Coldharbour Lane PRV (outlet)', 'Snowden Road PRV (inlet)', 'Snowden Road PRV (outlet)', 'New Station Way PRV (inlet)', 'New Station Way PRV (outlet)']
h_field_ids = pressure_device_id[~pressure_device_id['bwfl_id'].isin(not_these_pressure_ids)]['bwfl_id'].unique()

h_field = np.zeros((len(h_field_ids), len(datetime)))
h_field_node = []
for idx, bwfl_id in enumerate(h_field_ids):
    model_id = pressure_device_id[pressure_device_id['bwfl_id'] == bwfl_id]['model_id'].values[0]
    pressure_data = pressure_df[pressure_df['bwfl_id'] == bwfl_id]['mean'].values
    elev = pressure_device_id[pressure_device_id['bwfl_id'] == bwfl_id]['elev'].values[0]
    h_field[idx, :] = pressure_data + elev
    h_field_node.append(model_id)

h_field_idx = node_df[node_df['node_ID'].isin(h_field_node)].index


# flow devices
if demand_resolution == 'dma':
    not_these_flow_ids = ['inlet_2005', 'inlet_2296', 'Stoke Lane PRV', 'Woodland Way PRV', 'Coldharbour Lane PRV', 'Snowden Road DBV', 'New Station Way DBV']
    q_field_ids = flow_device_id[~flow_device_id['bwfl_id'].isin(not_these_flow_ids)]['bwfl_id'].unique()
    q_field = np.zeros((len(q_field_ids), len(datetime)))
    q_field_link = []
    for idx, bwfl_id in enumerate(q_field_ids):
        model_id = flow_device_id[flow_device_id['bwfl_id'] == bwfl_id]['model_id'].values[0]
        flow_data = flow_df[flow_df['bwfl_id'] == bwfl_id]['mean'].values  # lps
        q_field[idx, :] = flow_data
        q_field_link.append(model_id)

    q_field_idx = link_df[link_df['link_ID'].isin(q_field_link)].index
else:
    q_field_ids = []
    q_field_link = []
    q_field_idx = []




###### Step 9: compute initial hydraulics and plot residuals ######
q_0, h_0 = hydraulic_solver(wdn, d, h0, C_0, C_dbv, eta)

print(h_0)

# pressure residuals (train)
print(h_field_idx.values)
h_field_train = h_field[:, train_range]
h_sim_train = h_0[h_field_idx.values, train_range]
h_residuals = h_sim_train - h_field_train

residuals_df = pd.DataFrame(h_residuals.T, columns=h_field_ids)
residuals_df['datetime'] = datetime[train_range]

fig = px.box(residuals_df.melt(id_vars='datetime', var_name='Sensor', value_name='Residual'),
             x='Sensor', y='Residual', points='all')

fig.update_layout(
    xaxis_title='',
    yaxis_title='Residuals [m]',
    template='simple_white',
    height=450,
)
fig.show()

