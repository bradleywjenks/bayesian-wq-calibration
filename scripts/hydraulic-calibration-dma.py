"""
This script calibrates the Field Lab's hydraulic model using flow and pressure data. Specifically, HW coefficients are calibrated based on a selected grouping. The following steps are performed:
    1. load hydraulic model
    2. load sensor data
    3. assign isolation valve loss coefficients
    4. assign boundary heads from BWFL 19 and Woodland Way PRV (inlet)
    5. scale demands based on flow balance (DMA or WWMD resolution)
    6. get control valve losses (PRV + DBV links)
    7. split test/train data for h0, d, and eta
    8. get pressure and flow data: time series + model ID and check initial model performance
    9. compute initial hydraulics and plot residuals
    10. get pipe grouping
    11. calibrate HW coefficients via SCP algorithm or IPOPT

    Note!!! This script is messy. It will be cleaned up in the future...
"""

import numpy as np
import pandas as pd
from bayesian_wq_calibration.data import load_network_data, sensor_model_id
from bayesian_wq_calibration.constants import NETWORK_DIR, TIMESERIES_DIR, INP_FILE, IV_CLOSE, IV_OPEN, IV_OPEN_PART, SPLIT_INP_FILE, RESULTS_DIR
from bayesian_wq_calibration.simulation import hydraulic_solver, friction_loss, local_loss
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors
default_colors = plotly.colors.qualitative.Plotly
import json
import copy
# import gurobipy as gp
from pyomo.environ import *
import idaes
import scipy.sparse as sp
import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)




###### STEP 1: load hydraulic model ######
wdn = load_network_data(NETWORK_DIR / SPLIT_INP_FILE)
with open(NETWORK_DIR / 'valve_info.json') as f:
    valve_info = json.load(f)
A12 = wdn.A12
A10 = wdn.A10
net_info = wdn.net_info
link_df = wdn.link_df
node_df = wdn.node_df
demand_df = wdn.demand_df
C_0 = link_df['C'].values
valve_idx = link_df[link_df['link_type'] == 'valve'].index
pipe_idx = link_df[link_df['link_type'] == 'pipe'].index
prv_links = valve_info['prv_link']
prv_dir = valve_info['prv_dir']
prv_idx = link_df[link_df['link_ID'].isin(prv_links)].index



###### STEP 2: load sensor data ######
data_period = 16     # change data period!!!
flow_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(data_period).zfill(2)}-flow.csv")
flow_device_id = sensor_model_id('flow')
pressure_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(data_period).zfill(2)}-pressure.csv")
pressure_device_id = sensor_model_id('pressure')
datetime = pd.to_datetime(flow_df['datetime'].unique())

fig = px.line(
    flow_df,
    x='datetime',
    y='mean',
    color='bwfl_id',
)
fig.update_layout(
    xaxis_title='',
    yaxis_title='Flow [L/s]',
    legend_title_text='',
    template='simple_white',
    height=450,
)
fig.show()

fig = px.line(
    pressure_df,
    x='datetime',
    y='mean',
    color='bwfl_id',
)
fig.update_layout(
    xaxis_title='',
    yaxis_title='Pressure [m]',
    legend_title_text='',
    template='simple_white',
    height=450,
)
fig.show()





###### STEP 3: assign isolation/dynamic boundary valve loss coefficients ######
# iv_links = valve_info['iv_link']
# iv_idx = link_df[link_df['link_ID'].isin(iv_links)].index
# C_0[iv_idx] = IV_CLOSE

#### NOT NEEDED WHEN USING SPLIT INP FILE ####





###### STEP 4: get boundary heads from BWFL 19 and Woodland Way PRV (inlet) ######
reservoir_nodes = net_info['reservoir_names']
reservoir_idx = node_df[node_df['node_ID'].isin(reservoir_nodes)].index
h0 = np.zeros((len(reservoir_idx), len(datetime)))
for idx, node in enumerate(reservoir_nodes):
    bwfl_id = pressure_device_id[pressure_device_id['model_id'] == node]['bwfl_id'].values[0]
    elev = pressure_device_id[pressure_device_id['model_id'] == node]['elev'].values[0]
    h0_temp = pressure_df[pressure_df['bwfl_id'] == bwfl_id]['mean'].values + elev
    h0[idx, :] = np.round(h0_temp, 2)

if np.isnan(h0).any():
    logging.error("NaN values found in h0")
    raise ValueError("NaN values found in h0")

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

d = scaled_demand_df.iloc[:, 3:].values

# set demands at Snowden Road DBV
node_start = 'node_1809' 
node_start_idx = node_df[node_df['node_ID'] == node_start].index[0]
node_end = 'node_1808'
node_end_idx = node_df[node_df['node_ID'] == node_end].index[0]
dbv_flow = flow_df[flow_df['bwfl_id'] == 'Snowden Road DBV']['mean'].values
d[node_end_idx, :] = dbv_flow * -1
d[node_start_idx, :] = dbv_flow

# set demands at New Station Way DBV
node_start = 'node_1811' 
node_start_idx = node_df[node_df['node_ID'] == node_start].index[0]
node_end = 'node_1810'
node_end_idx = node_df[node_df['node_ID'] == node_end].index[0]
dbv_flow = flow_df[flow_df['bwfl_id'] == 'New Station Way DBV']['mean'].values
d[node_end_idx, :] = dbv_flow * -1
d[node_start_idx, :] = dbv_flow

d = d / 1000

if np.isnan(d).any():
    logging.error("NaN values found in d")
    raise ValueError("NaN values found in d")





###### STEP 6: get PRV local loss (inlet - outlet) values ######
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
n_train = 1 * 24 * 4
# n_train = 4 * 24 * 4

train_range = range(n_train)
test_range = range(n_train, n_total)




###### Step 8: get sensor data ######
# pressure devices
not_these_pressure_ids = ['BWFL 19', 'Woodland Way PRV (inlet)', 'Woodland Way PRV (outlet)', 'Lodge Causeway PRV (inlet)', 'Lodge Causeway PRV (outlet)', 'Stoke Lane PRV (inlet)', 'Stoke Lane PRV (outlet)', 'Coldharbour Lane PRV (outlet)', 'BWFL 41', 'BWFL_A01R']
h_field_ids = pressure_device_id[~pressure_device_id['bwfl_id'].isin(not_these_pressure_ids)]['bwfl_id'].unique()

h_field = np.zeros((len(h_field_ids), len(datetime)))
h_field_node = []
h_field_idx = []
for idx, bwfl_id in enumerate(h_field_ids):
    model_id = pressure_device_id[pressure_device_id['bwfl_id'] == bwfl_id]['model_id'].values[0]
    pressure_data = pressure_df[pressure_df['bwfl_id'] == bwfl_id]['mean'].values[0]
    elev = pressure_device_id[pressure_device_id['bwfl_id'] == bwfl_id]['elev'].values[0]
    h_field[idx, :] = pressure_data + elev
    h_field_node.append(model_id)
    h_field_idx.append(node_df[node_df['node_ID'] == model_id].index[0])

if np.isnan(h_field[:, train_range]).any():
    logging.error("NaN values found in h_field")
    raise ValueError("NaN values found in h_field")

# flow devices
if demand_resolution == 'dma':
    not_these_flow_ids = ['inlet_2005', 'inlet_2296', 'Stoke Lane PRV', 'Woodland Way PRV', 'Coldharbour Lane PRV']
    q_field_ids = flow_device_id[~flow_device_id['bwfl_id'].isin(not_these_flow_ids)]['bwfl_id'].unique()
    q_field = np.zeros((len(q_field_ids), len(datetime)))
    q_field_link = []
    q_field_idx = []
    for idx, bwfl_id in enumerate(q_field_ids):
        model_id = flow_device_id[flow_device_id['bwfl_id'] == bwfl_id]['model_id'].values[0]
        flow_data = flow_df[flow_df['bwfl_id'] == bwfl_id]['mean'].values  # lps
        q_field[idx, :] = flow_data
        q_field_link.append(model_id)
        q_field_idx.append(link_df[link_df['link_ID'].isin(q_field_link)].index[0])

else:
    q_field_ids = []
    q_field_link = []
    q_field_idx = []





###### Step 9: compute initial hydraulics and plot residuals ######
q_0, h_0 = hydraulic_solver(wdn, d, h0, C_0, eta)

def loss_fun(h_field, h_sim):
    h_field_flat = h_field.flatten()
    h_sim_flat = h_sim.flatten()
    # mask = ~np.isnan(h_field_flat) & ~np.isnan(h_sim_flat)
    return (1 / len(h_field_flat)) * sum((h_sim_flat - h_field_flat) ** 2)

# pressure initial residuals (train)
h_residuals_0 = h_0[h_field_idx, :][:, train_range] - h_field[:, train_range]
residuals_0_df = pd.DataFrame(h_residuals_0, index=h_field_ids)
residuals_0_df = residuals_0_df.reset_index().melt(id_vars='index', var_name='time_index', value_name='residual')
residuals_0_df = residuals_0_df.rename(columns={'index': 'bwfl_id'})
residuals_0_df['dma_id'] = residuals_0_df['bwfl_id'].map(pressure_device_id.set_index('bwfl_id')['dma_id'])
# residuals_0_df = residuals_0_df.dropna(subset=['residual'])


fig = px.box(
    residuals_0_df, 
    x="bwfl_id",
    y="residual",
    color="dma_id",
)
fig.update_layout(
    title="Pre-calibration residuals",
    xaxis_title="",
    yaxis_title="Pressure residual [m]",
    template='simple_white',
    height=450,
    legend_title="DMA",
    boxmode="overlay",
    xaxis=dict(
        tickangle=45,
    )
)
fig.show()





###### Step 10: make pipe grouping index list ######
C_0_pipe = np.array([C_0[idx] for idx, row in link_df.iterrows() if row['link_type'] == 'pipe'])
C_0_pipe_unique = np.unique(C_0_pipe)
group_mapping = {value: idx for idx, value in enumerate(C_0_pipe_unique)}
pipe_grouping = {idx: group_mapping[val] for idx, val in enumerate(C_0_pipe)}
num_pipe_groups = len(C_0_pipe_unique)

# plot C_0 values
data = {
    'C_0': [C_0[idx] for idx, row in link_df.iterrows()],
    'type': ['pipe' if row['link_type'] == 'pipe' else 'valve' for idx, row in link_df.iterrows()],
    'index': list(range(len(C_0)))  # Use the same index for pipes and valves
}
df = pd.DataFrame(data)
pipe_df = df[df['type'] == 'pipe'].reset_index(drop=True)
valve_df = df[df['type'] == 'valve'].reset_index(drop=True)

fig = make_subplots(rows=2, cols=1, shared_yaxes=True, vertical_spacing=0.15)
fig.add_trace(
    go.Scatter(
        x=pipe_df['index'],
        y=pipe_df['C_0'],
        mode='markers',
        marker=dict(
            symbol='circle',
            color='rgba(0,0,0,0)',
            line=dict(color=default_colors[0], width=0.75)
        ),
        name='Pipe'
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=valve_df['index'],
        y=valve_df['C_0'],
        mode='markers',
        marker=dict(
            symbol='circle',
            color='rgba(0,0,0,0)',
            line=dict(color=default_colors[1], width=0.75)
        ),
        name='Valve'
    ),
    row=2, col=1
)
fig.update_layout(
    template='simple_white',
    height=900,
    legend_title="Link type",
)
fig.update_xaxes(title_text="Pipe link", row=1, col=1)
fig.update_xaxes(title_text="Valve link", row=2, col=1)
fig.update_yaxes(title_text="Initial HW coefficient", row=1, col=1)
fig.update_yaxes(title_text="Initial valve coefficient", row=2, col=1)
fig.show()





###### Step 11: calibrate HW coefficients via SCP algorithm ######

# functions
def linear_approx_calibration(wdn, q, C):

    net_info = wdn.net_info
    link_df = wdn.link_df

    K = np.zeros((net_info['np'], 1))
    n_exp = link_df['n_exp'].astype(float).to_numpy().reshape(-1, 1)
    b1_k = copy.copy(K)
    b2_k = copy.copy(K)

    for idx, row in link_df.iterrows():
        if row['link_type'] == 'pipe':
            K_cms = 10.67 * row['length'] * (C[idx] ** -row['n_exp']) * (row['diameter'] ** -4.8704)
            # K_lps = ((1e-3) ** n_exp[idx]) * K_cms
            K[idx] = K_cms
            b1_k[idx] = K_cms
            b2_k[idx] = (-n_exp[idx] * K_cms) / C[idx]

        elif row['link_type'] == 'valve':
            K_cms = (8 / (np.pi ** 2 * 9.81)) * (row['diameter'] ** -4) * C[idx]
            # K_lps = ((1e-3) ** n_exp[idx]) * K_cms
            K[idx] = K_cms
            b1_k[idx] = -n_exp[idx] * K_cms
            b2_k[idx] = K_cms/ C[idx]

    a11_k = np.tile(K, q.shape[1]) * np.abs(q) ** (n_exp - 1)
    b1_k = np.tile(b1_k, q.shape[1]) * np.abs(q) ** (n_exp - 1)
    b2_k = np.tile(b2_k, q.shape[1]) * np.abs(q) ** (n_exp - 1) * q

    return a11_k, b1_k, b2_k

n_exp = link_df['n_exp'].astype(float).to_numpy().reshape(-1, 1)
length = link_df['length'].astype(float).to_numpy().reshape(-1, 1)
diameter = link_df['diameter'].astype(float).to_numpy().reshape(-1, 1)


#### SCP algorithm ####
solution_method = 'ipopt' # 'gurobi', 'ipopt'

Ki = np.inf
q_tol = 1e-8
iter_max = 50
delta_k = 20
C_up_pipe = 200
C_lo_pipe = 20
A13 = np.zeros((net_info['np'], len(prv_idx)))
for col, idx in enumerate(prv_idx):
    A13[idx, col] = 1
theta_k = C_0
theta_k_dict = {(j): theta_k[j] for j in range(theta_k.shape[0])}
Theta_k = C_0_pipe_unique
Theta_k_dict = {(g): Theta_k[g] for g in range(Theta_k.shape[0])}
q_k, h_k = hydraulic_solver(wdn, d[:, train_range], h0[:, train_range], theta_k, eta[:, train_range])
q_k_dict = {(i, j): q_k[i, j] for i in range(q_k.shape[0]) for j in range(q_k.shape[1])}
h_k_dict = {(i, j): h_k[i, j] for i in range(h_k.shape[0]) for j in range(h_k.shape[1])}

a11_k, b1_k, b2_k = linear_approx_calibration(wdn, q_k, theta_k)
a11_k_dict = {(i, j): a11_k[i, j] for i in range(a11_k.shape[0]) for j in range(a11_k.shape[1])}
b1_k_dict = {(i, j): b1_k[i, j] for i in range(b1_k.shape[0]) for j in range(b1_k.shape[1])}
b2_k_dict = {(i, j): b2_k[i, j] for i in range(b2_k.shape[0]) for j in range(b2_k.shape[1])}

objvals = []
objval_k = loss_fun(h_field[:, train_range], h_k[h_field_idx, :][:, train_range])
objvals.append(objval_k)

# pyomo model
model = ConcreteModel()

# define sets, variables, and parameters
model.j_set = RangeSet(0, net_info['np']-1)
model.i_set = RangeSet(0, net_info['nn']-1)
model.t_set = RangeSet(0, n_train-1)
model.p_set = RangeSet(pipe_idx[0], pipe_idx[-1])
model.v_set = RangeSet(valve_idx[0], valve_idx[-1])
model.r_set = RangeSet(0, len(prv_idx)-1)
model.s_set = RangeSet(0, h0.shape[0]-1)
model.g_set = RangeSet(0, num_pipe_groups-1)
model.q = Var(model.j_set, model.t_set, initialize=q_k_dict)
model.h = Var(model.i_set, model.t_set, initialize=h_k_dict)
model.theta = Var(model.j_set, initialize=theta_k_dict)
model.Theta = Var(model.g_set, initialize=Theta_k_dict)
if solution_method == 'gurobi':
    model.b1_k = Param(model.j_set, model.t_set, mutable=True, initialize=b1_k_dict)
    model.b2_k = Param(model.j_set, model.t_set, mutable=True, initialize=b2_k_dict)
    model.a11_k = Param(model.j_set, model.t_set, mutable=True, initialize=a11_k_dict)    
    model.q_k = Param(model.j_set, model.t_set, mutable=True, initialize=q_k_dict)
    model.h_k = Param(model.i_set, model.t_set, mutable=True, initialize=h_k_dict)
    model.theta_k = Param(model.j_set, mutable=True, initialize=theta_k_dict)
    model.delta_k = Param(mutable=True, initialize=delta_k)


# objective function
def objective_function(m):
    return (1 / len(h_field[:, m.t_set].flatten())) * sum(
        (m.h[i, t] - h_field[n, t]) ** 2 for n, i in enumerate(h_field_idx) for t in m.t_set
    )
model.objective = Objective(rule=objective_function, sense=minimize)

# constraints
nodes_map = {i: np.where(A12.toarray()[i, :] != 0) for i in range(net_info['np'])}
sources_map = {i: np.where(A10.toarray()[i, :] != 0) for i in range(net_info['np'])}
links_map = {i: np.where(A12.T.toarray()[i, :] != 0) for i in range(net_info['nn'])}
def energy_constraint_rule(m, j, t):
    if solution_method == 'ipopt':
        if j in pipe_idx:
            return (
                10.67 * length[j] * (m.theta[j] ** -n_exp[j]) * (diameter[j] ** -4.8704) * (m.q[j, t]) * abs(m.q[j, t] + q_tol)**(n_exp[j] - 1)
                + sum(A12[j, i] * m.h[i, t] for i in nodes_map[j][0])
                + sum(A10[j, s] * h0[s, t] for s in sources_map[j][0])
                + sum(A13[j, r] * eta[r, t] for r in m.r_set)
            ) == 0
        elif j in valve_idx:
            return (
                (8 / (np.pi ** 2 * 9.81)) * (diameter[j] ** -4) * m.theta[j] * (m.q[j, t]) * abs(m.q[j, t] + q_tol)**(n_exp[j] - 1)
                + sum(A12[j, i] * m.h[i, t] for i in nodes_map[j][0])
                + sum(A10[j, s] * h0[s, t] for s in sources_map[j][0])
                + sum(A13[j, r] * eta[r, t] for r in m.r_set)
            ) == 0
    elif solution_method == 'gurobi':
        return (
            m.b1_k[j, t] * m.q_k[j, t]
            + n_exp[j] * m.a11_k[j, t] * m.q[j, t]
            + m.b2_k[j, t] * m.theta[j]
            + sum(A12[j, i] * m.h[i, t] for i in nodes_map[j][0])
            + sum(A10[j, s] * h0[s, t] for s in sources_map[j][0])
            + sum(A13[j, r] * eta[r, t] for r in m.r_set)
        ) == 0
model.energy_constraint = Constraint(model.j_set, model.t_set, rule=energy_constraint_rule)

def mass_constraint_rule(m, i, t):
    return sum(A12.T[i, j] * m.q[j, t] for j in links_map[i][0]) == d[i, t]
model.mass_constraint = Constraint(model.i_set, model.t_set, rule=mass_constraint_rule)

def group_constraint_rule(m, p):
    return m.theta[p] == m.Theta[pipe_grouping[p]]
model.group_constraint = Constraint(model.p_set, rule=group_constraint_rule)

def lower_bound_rule(m, g):
    return m.Theta[g] >= C_lo_pipe
model.lower_bound_constraint = Constraint(model.g_set, rule=lower_bound_rule)

def upper_bound_rule(m, g):
    return m.Theta[g] <= C_up_pipe
model.upper_bound_constraint = Constraint(model.g_set, rule=upper_bound_rule)

def valve_constraint_rule(m, v):
    return m.theta[v] == C_0[v]
model.valve_constraint = Constraint(model.v_set, rule=valve_constraint_rule)

if solution_method == 'gurobi':

    def trust_region_rule_lower(m, j):
        return -m.delta_k <= m.theta[j] - m.theta_k[j]
    model.trust_region_constraint_lower = Constraint(model.j_set, rule=trust_region_rule_lower)

    def trust_region_rule_upper(m, j):
        return m.theta[j] - m.theta_k[j] <= m.delta_k
    model.trust_region_constraint_upper = Constraint(model.j_set, rule=trust_region_rule_upper)

    # parameters
    def update_parameters(m, q_tilde, h_tilde, theta_tilde, success):
        if success:
            a11_k, b1_k, b2_k = linear_approx_calibration(wdn, q_tilde, theta_tilde)
            for t in m.t_set:
                for j in m.j_set:
                    m.a11_k[j, t] = a11_k[j, t]
                    m.b1_k[j, t] = b1_k[j, t]
                    m.b2_k[j, t] = b2_k[j, t]
                    m.q_k[j, t] = q_tilde[j, t]
                for i in m.i_set:
                    m.h_k[i, t] = h_tilde[i, t]
            for j in m.j_set:
                m.theta_k[j] = theta_tilde[j]
            m.delta_k = value(m.delta_k) * 1.1
        else:
            m.delta_k = value(m.delta_k) * 0.25


# run algorithm   
if solution_method == 'ipopt':
    solver = SolverFactory('ipopt')
    solver.options['tol'] = 1e-3
    # solver.options['acceptable_tol'] = 1e-4
    # solver.options['constr_viol_tol'] = 1e-4
    # solver.options['compl_inf_tol'] = 1e-4

    result = solver.solve(model, tee=True)
    theta_opt = np.array([model.theta[j].value for j in model.j_set])
    q_opt, h_opt = hydraulic_solver(wdn, d[:, train_range], h0[:, train_range], theta_opt, eta[:, train_range])

elif solution_method == 'gurobi':
    solver = SolverFactory("gurobi")
    solver.options['FeasibilityTol'] = 1e-3
    solver.options['OptimalityTol'] = 1e-3
    print(f"{'iter':<10} {'objval':<15} {'Ki':<10} {'delta':<10}")     
    print(f"{0:<5} {objval_k:<15.6f} {Ki:<10.6f} {delta_k:<10.6f}")         
    for k in range(1, iter_max+1):
        results = solver.solve(model, tee=True)
        objval_convex = value(model.objective)
        print(objval_convex)
        theta_tilde = np.array([model.theta[j].value for j in model.j_set])
        q_tilde, h_tilde = hydraulic_solver(wdn, d[:, train_range], h0[:, train_range], theta_tilde, eta[:, train_range])
        objval_tilde = loss_fun(h_field[:, train_range], h_tilde[h_field_idx, :][:, train_range])
        print(objval_tilde)

        if (objval_k - objval_tilde) / abs(objval_k - objval_convex) >= 0.1:
            success = True
            update_parameters(model, q_tilde, h_tilde, theta_tilde, success)
            Ki = abs(objval_k - objval_tilde) / abs(objval_k)
            objval_k = objval_tilde
            objvals.append(objval_k)
        else:
            success = False
            update_parameters(model, q_tilde, h_tilde, theta_tilde, success)

        print(f"{k:<5} {objval_k:<15.6f} {Ki:<10.6f} {value(model.delta_k):<10.6f} \n")

        if Ki <= 1e-3 or value(model.delta_k) <= 1e-2:
                break
        
    theta_opt = np.array([model.theta[j].value for j in model.j_set])
        




###### Step 12: plot calibration results ######

# pressure residuals (test)
q_opt, h_opt = hydraulic_solver(wdn, d[:, test_range], h0[:, test_range], theta_opt, eta[:, test_range])
# h_residuals_opt = h_opt[h_field_idx, :][:, test_range] - h_field[:, test_range]
# residuals_opt_df = pd.DataFrame(h_residuals_opt, index=h_field_ids)
# residuals_opt_df = residuals_opt_df.reset_index().melt(id_vars='index', var_name='time_index', value_name='residual')
# residuals_opt_df = residuals_opt_df.rename(columns={'index': 'bwfl_id'})
# residuals_opt_df['dma_id'] = residuals_opt_df['bwfl_id'].map(pressure_device_id.set_index('bwfl_id')['dma_id'])

print(h_opt.shape)

# fig = px.box(
#     residuals_opt_df, 
#     x="bwfl_id",
#     y="residual",
#     color="dma_id",
# )
# fig.update_layout(
#     title="Post-calibration residuals (test dataset)",
#     xaxis_title="",
#     yaxis_title="Pressure residual [m]",
#     template='simple_white',
#     height=450,
#     legend_title="DMA",
#     boxmode="overlay",
#     xaxis=dict(
#         tickangle=45,
#     )
# )
# fig.show()


# optimized HW and local loss coefficients
data = {
    'C': [theta_opt[idx] for idx, row in link_df.iterrows()],
    'type': ['pipe' if row['link_type'] == 'pipe' else 'valve' for idx, row in link_df.iterrows()],
    'index': list(range(len(theta_opt))) 
}
df = pd.DataFrame(data)
df.to_csv(RESULTS_DIR / 'calibrated_coefficients.csv', index=True)
pipe_df = df[df['type'] == 'pipe'].reset_index(drop=True)
valve_df = df[df['type'] == 'valve'].reset_index(drop=True)

fig = make_subplots(rows=2, cols=1, shared_yaxes=True, vertical_spacing=0.15)
fig.add_trace(
    go.Scatter(
        x=pipe_df['index'],
        y=pipe_df['C'],
        mode='markers',
        marker=dict(
            symbol='circle',
            color='rgba(0,0,0,0)',
            line=dict(color=default_colors[0], width=0.75)
        ),
        name='Pipe'
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=valve_df['index'],
        y=valve_df['C'],
        mode='markers',
        marker=dict(
            symbol='circle',
            color='rgba(0,0,0,0)',
            line=dict(color=default_colors[1], width=0.75)
        ),
        name='Valve'
    ),
    row=2, col=1
)
fig.update_layout(
    template='simple_white',
    height=900,
    legend_title="Link type",
)
fig.update_xaxes(title_text="Pipe link", row=1, col=1)
fig.update_xaxes(title_text="Valve link", row=2, col=1)
fig.update_yaxes(title_text="Calibrated HW coefficient", row=1, col=1)
fig.update_yaxes(title_text="Calibrated valve coefficient", row=2, col=1)
fig.show()
