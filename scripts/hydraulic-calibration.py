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
from plotly.subplots import make_subplots
import plotly.colors
default_colors = plotly.colors.qualitative.Plotly
import json
import copy
import gurobipy as gp
from gurobipy import GRB
import pyomo.environ as pyo
import scipy.sparse as sp
from pyomo.opt import SolverFactory
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

def loss_fun(h_field, h_sim):
    h_field_flat = h_field.flatten()
    h_sim_flat = h_sim.flatten()
    mask = ~np.isnan(h_field_flat) & ~np.isnan(h_sim_flat)
    return (1 / mask.sum()) * np.sum((h_sim_flat[mask] - h_field_flat[mask]) ** 2)

# pressure initial residuals (train)
h_residuals_0 = h_0[h_field_idx, :][:, train_range] - h_field[:, train_range]

residuals_0_df = pd.DataFrame(h_residuals_0, index=h_field_ids)
residuals_0_df = residuals_0_df.reset_index().melt(id_vars='index', var_name='time_index', value_name='residual')
residuals_0_df = residuals_0_df.rename(columns={'index': 'bwfl_id'})
residuals_0_df['dma_id'] = residuals_0_df['bwfl_id'].map(pressure_device_id.set_index('bwfl_id')['dma_id'])
# residuals_0_df = residuals_0_df.dropna(subset=['residual'])

mse_train_0 = loss_fun(h_field[:, train_range], h_0[h_field_idx, :][:, train_range])
print(f'Initial mse on training data: {round(mse_train_0, 2)}')


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
pipe_grouping = np.array([group_mapping[val] for val in C_0_pipe])

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
    # unload data
    A12 = wdn.A12
    A10 = wdn.A10
    net_info = wdn.net_info
    link_df = wdn.link_df

    K = np.zeros((net_info['np'], 1))
    n_exp = link_df['n_exp'].astype(float).to_numpy().reshape(-1, 1)
    b1_k = copy.copy(K)
    b2_k = copy.copy(K)

    for idx, row in link_df.iterrows():
        if row['link_type'] == 'pipe':
            K[idx] = 10.67 * row['length'] * (C[idx] ** -row['n_exp']) * (row['diameter'] ** -4.8704)
            b1_k[idx] = copy.copy(K[idx])
            b2_k[idx] = (-n_exp[idx] * K[idx]) / C[idx]

        elif row['link_type'] == 'valve':
            K[idx] = (8 / (np.pi ** 2 * 9.81)) * (row['diameter'] ** -4) * C[idx]
            b1_k[idx] = -n_exp[idx] * copy.copy(K[idx]) 
            b2_k[idx] = copy.copy(K[idx]) / C[idx]

    a11_k = np.tile(K, q.shape[1]) * np.abs(q) ** (n_exp - 1)
    b1_k = np.tile(b1_k, q.shape[1]) * np.abs(q) ** (n_exp - 1)
    b2_k = np.tile(b2_k, q.shape[1]) * np.abs(q) ** (n_exp - 1) * q

    return a11_k, b1_k, b2_k


#### SCP algorithm ####
Ki = np.inf
iter_max = 50
delta_k = 20
C_up_pipe = 200
C_lo_pipe = 1e-4
A13 = np.zeros((net_info['np'], len(prv_idx)))
for col, idx in enumerate(prv_idx):
    A13[idx, col] = 1
A13 = sp.csr_matrix(A13)
n_exp = link_df['n_exp']
theta_k = C_0
q_k, h_k = hydraulic_solver(wdn, d[:, train_range], h0[:, train_range], theta_k, C_dbv[:, train_range], eta[:, train_range])
a11_k, b1_k, b2_k = linear_approx_calibration(wdn, q_k, theta_k)

# pyomo model
model = pyo.ConcreteModel()

# define sets, variables, and parameters
model.np = pyo.Set(initialize=range(net_info['np']-1))
model.nn = pyo.Set(initialize=range(net_info['nn']-1))
model.n_train = pyo.Set(initialize=train_range)
model.q = pyo.Var(model.np, model.n_train, domain=pyo.Reals)
model.h = pyo.Var(model.nn, model.n_train, domain=pyo.Reals)
model.theta = pyo.Var(model.np, domain=pyo.Reals)
model.b1_k = pyo.Param(model.np, model.n_train, mutable=True, initialize=b1_k)
model.b2_k = pyo.Param(model.np, model.n_train, mutable=True, initialize=b2_k)
model.a11_k = pyo.Param(model.np, model.n_train, mutable=True, initialize=a11_k)
model.q_k = pyo.Param(model.np, model.n_train, mutable=True, initialize=q_k)
model.h_k = pyo.Param(model.nn, model.n_train, mutable=True, initialize=h_k)
model.theta_k = pyo.Param(model.np, mutable=True, initialize=theta_k)
model.delta_k = pyo.Param(mutable=True, initialize=delta_k)

# objective function
def objective_function(m):
    return (1 / len(h_field[:, train_range].flatten())) * sum(
        (m.h[i, t] - h_field[n, t]) ** 2 for n, i in enumerate(h_field_idx) for t in train_range
    )
model.objective = pyo.Objective(rule=objective_function, sense=pyo.minimize)

# constraints
def energy_constraint_rule(m, j, t):
    return (
        m.b1_k[j, t] * m.q_k[j, t] +
        n_exp[j] * m.a11_k[j, t] * m.q[j, t] +
        m.b2_k[j, t] * m.theta[j] +
        sum(A12[j, i] * m.h[i, t] for i in m.nn) +
        sum(A10[j, s] * h0[s, t] for s in range(len(reservoir_idx))) +
        sum(A13[j, p] * eta[p, t] for p in range(len(prv_idx)))
    ) == 0
model.energy_constraint = pyo.Constraint(model.np, model.n_train, rule=energy_constraint_rule)

def mass_constraint_rule(m, i, t):
    return sum(A12[i, j] * m.q[j, t] for j in model.np) == d[i, t]
model.mass_constraint = pyo.Constraint(model.nn, model.n_train, rule=mass_constraint_rule)

def trust_region_rule(m, j):
    return abs(m.theta[j] - m.theta_k[j]) <= m.delta_k
model.trust_region_constraint = pyo.Constraint(model.np, rule=trust_region_rule)

def lower_bound_rule(m, j):
    return m.theta[j] >= C_lo_pipe
model.lower_bound_constraint = pyo.Constraint(model.np, rule=lower_bound_rule)

def upper_bound_rule(m, j):
    return m.theta[j] <= C_up_pipe
model.upper_bound_constraint = pyo.Constraint(model.np, rule=upper_bound_rule)

def valve_constraint_rule(m, j):
        if link_df.iloc[j]['link_type'] == 'valve':
            return m.theta[j] == C_0[j]
model.valve_constraint = pyo.Constraint(model.np, rule=valve_constraint_rule)

# parameters
def update_parameters(m, k, q_tilde, h_tilde, theta_tilde, success):
    if success:
        a11_k, b1_k, b2_k = linear_approx_calibration(wdn, q_tilde, theta_tilde)
        for t in m.n_train:
            for j in m.np:
                m.a11_k[j, t] = a11_k[j, t]
                m.b1_k[j, t] = b1_k[j, t]
                m.b2_k[j, t] = b2_k[j, t]
                m.q_k[j, t] = q_tilde[j, t]
                m.h_k[j, t] = h_tilde[j, t]
            for i in m.nn:
                m.h_k[i, t] = h_tilde[i, t]
        for j in m.np:
            m.theta_k[j] = theta_tilde[j]
        m.delta_k = m.delta_k * 1.1
        print(f"Iteration {k} successful! Updating estimate and increasing trust region size.")
    else:
        m.delta_k = m.delta_k / 4
        print(f"Iteration {k} unsuccessful! Reducing trust region size.")


# # run algorithm
# solver = SolverFactory("gurobi_persistent")                             
# for k in range(iter_max):
#     results = solver.solve(model, tee=False)

#     # check convergence
#     q_tilde = np.array([[model.q[j, t].value for j in model.np] for t in model.n_train])
#     h_tilde = np.array([[model.h[i, t].value for i in model.nn] for t in model.n_train])
#     theta_tilde = np.array([model.theta[j].value for j in model.np])

#     if results.solver.status == pyo.SolverStatus.ok and 
#     success = results.solver.status == pyo.SolverStatus.ok
#     update_parameters(model, k, q_tilde, h_tilde, theta_tilde, success)

#     # check convergence
#     if success:
#         q_k, h_k = hydraulic_solver(wdn, d[:, train_range], h0[:, train_range], theta_tilde, C_dbv[:, train_range], eta[:, train_range])
#         a11_k, b1_k, b2_k = linear_approx_calibration(wdn, q_k, theta_tilde)
#         mse_train = loss_fun(h_field[:, train_range], h_k[h_field_idx, :][:, train_range])
#         print(f"Iteration {k} - MSE on training data: {round(mse_train, 2)}")
#         if mse_train < Ki:
#             Ki = mse_train
#             theta_star = theta_tilde
#             q_star = q_k
#             h_star = h_k
#     else:
#         print(f"Iteration {k} - Solver failed to converge.")

#     if model.delta_k < 1e-5:
#         print("Trust region too small. Exiting SCP algorithm.")
#         break