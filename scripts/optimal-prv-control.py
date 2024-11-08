"""
This script optimizes the Field Lab's PRV settings using its hydraulic model and demand scaling from field data. The following steps are performed:
    1. load hydraulic model
    2. load sensor data
    3. assign boundary heads from BWFL 19 and Woodland Way PRV (inlet)
    4. scale demands based on flow balance (DMA or WWMD resolution)
    5. formulate model parameters and variable bounds
    6. optimize PRV settings using Ipopt
    7. plot results
    8. create flow modulation profiles for each PRV

    Note!!! This script is messy. It will be cleaned up in the future...
"""

import numpy as np
import pandas as pd
from bayesian_wq_calibration.data import load_network_data, sensor_model_id
from bayesian_wq_calibration.constants import NETWORK_DIR, TIMESERIES_DIR, INP_FILE, IV_CLOSE, IV_OPEN, IV_OPEN_PART, SPLIT_INP_FILE, RESULTS_DIR
from bayesian_wq_calibration.simulation import hydraulic_solver, friction_loss, local_loss
from bayesian_wq_calibration.plotting import plot_network
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors
default_colors = plotly.colors.qualitative.Plotly
import json
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
C = link_df['C'].values.astype(float).reshape(-1, 1)
D = link_df['diameter'].values.astype(float).reshape(-1, 1)
L = link_df['length'].values.astype(float).reshape(-1, 1)
n_exp = link_df['n_exp'].values
elevation = node_df.loc[node_df['node_ID'].isin(net_info['junction_names']), 'elev'].astype(float).to_numpy().reshape(-1, 1)
valve_idx = link_df[link_df['link_type'] == 'valve'].index
pipe_idx = link_df[link_df['link_type'] == 'pipe'].index
prv_links = valve_info['prv_link']
prv_dir = valve_info['prv_dir']
prv_idx = link_df[link_df['link_ID'].isin(prv_links)].index
prv_settings = valve_info['prv_settings']
prv_ids = ['Lodge Causeway PRV', 'Stoke Lane PRV', 'Woodland Way PRV', 'Cold Harbour Lane PRV']
critical_nodes = ['node_2697', 'node_1266', 'node_2552', 'node_2661'] # ['BWFL 22 (CP)', 'BWFL 9 (CP)', 'BWFL 16 (CP)', 'Cold Harbour Lane (CP)']





###### STEP 2: load sensor data ######
data_period = 18     # change data period!!!
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



###### STEP 3: get boundary heads from BWFL 19 and Woodland Way PRV (inlet) ######
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





###### STEP 4: scale demands ######
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

d = d / 1000

if np.isnan(d).any():
    logging.error("NaN values found in d")
    raise ValueError("NaN values found in d")






###### STEP 5: formulate model parameters, variable bounds and variable bounds ######
# select day to optimize settings (8 days available); recommend using 1 day as the model can become quite large over multiple days
day = 0
nt_range = range(day*96, (day+1)*96)
nt = len(nt_range)
datetime_range = flow_df['datetime'].sort_values().unique()[nt_range]

# make A13 matrix (pcv mapping)
A13 = np.zeros([net_info['np'], len(prv_idx)])
for idx, loc in enumerate(prv_idx):
    A13[loc, idx] = 1
A13 = sp.csr_matrix(A13)

# head loss model parameters
K = np.zeros((net_info['np'], 1))
for idx, row in link_df.iterrows():
    if row['link_type'] == 'pipe':
        K[idx] = friction_loss(net_info, row, C[idx])

    elif row['link_type'] == 'valve':
        K[idx] = local_loss(row, C[idx])

# node and lint maps
nodes_map = {i: np.where(A12.toarray()[i, :] != 0) for i in range(0, net_info['np'])}
sources_map = {i: np.where(A10.toarray()[i, :] != 0) for i in range(0, net_info['np'])}
links_map = {i: np.where(A12.T.toarray()[i, :] != 0) for i in range(0, net_info['nn'])}

# initialise continuous variables
d_opt = d[:, nt_range]
h0_opt = h0[:, nt_range]
eta_0 = np.zeros([len(prv_idx), nt])
# eta_0 = np.tile([5, 5, 5, 5], (nt, 1)).T
# eta_0 = np.tile(prv_settings, (nt, 1)).T
q_0, h_0 = hydraulic_solver(wdn, d_opt, h0_opt, C, eta_0) # no control hydraulics

# h_plot = h_0 - np.tile(elevation, (1, nt))
# h_plot_df = pd.DataFrame(h_plot[:])
# fig = px.box(h_plot_df)
# fig.show()

# AZP weights (w)
w = np.abs(A12.T) * np.array(L) / 2
azp_weights = w / np.sum(w)

# h and q variable bounds
p_min = 15 # minimum pressure head
h_min = np.tile(elevation, (1, nt))
h_min[d_opt > 0] += p_min
h_min[d_opt == 0] += 5
h_max = np.ones([net_info['nn'], 1]) * np.max(np.vstack((h0_opt, h_0)))
h_max = np.tile(h_max, (1, nt))

q_min = -100 * np.ones([net_info['np'], nt])
q_max = 100 * np.ones([net_info['np'], nt])

# eta variable bounds
A = np.hstack([A12.toarray(), A10.toarray()])
all_h_min = np.vstack([h_min, h0_opt])
all_h_max = np.vstack([h_max, h0_opt])

eta_min = np.zeros([net_info['np'], nt])
eta_max = np.zeros([net_info['np'], nt])

for link in range(net_info['np']):
    out_node = np.where(A[link, :] == -1)[0]
    in_node = np.where(A[link, :] == 1)[0]

    eta_min[link, :] = all_h_min[out_node[0], :] - all_h_max[in_node[0], :]
    eta_max[link, :] = all_h_max[out_node[0], :] - all_h_min[in_node[0], :]

eta_min = eta_min[prv_idx, :]
eta_max = eta_max[prv_idx, :]

# # enforce q and eta bounds at PCV links
# q_min[prv_idx, :] = 0
eta_min[:] = 0





###### STEP 6: optimize PRV settings ######
# create a Pyomo Concrete Model
model = ConcreteModel()

# define index sets
model.i_set = RangeSet(0, net_info['nn'] - 1)
model.j_set = RangeSet(0, net_info['np'] - 1)
model.t_set = RangeSet(0, nt - 1)
model.n_set = RangeSet(0, len(prv_idx) - 1)
model.s_set = RangeSet(0, A10.shape[1] - 1)

# define function for variable bounds
def bounds_h(model, i, t):
    return (h_min[i, t], h_max[i, t])
def bounds_q(model, j, t):
    return (q_min[j, t], q_max[j, t])
def bounds_eta(model, n, t):
    return (eta_min[n, t], eta_max[n, t])

# create dictionaries of initial values
q_0_dict = {(j, t): q_0[j, t] for j in model.j_set for t in model.t_set}
h_0_dict = {(i, t): h_0[i, t] for i in model.i_set for t in model.t_set}
eta_0_dict = {(n, t): eta_0[n, t] for n in model.n_set for t in model.t_set}

# define variables
model.q = Var(model.j_set, model.t_set, bounds=bounds_q, initialize=q_0_dict)
model.h = Var(model.i_set, model.t_set, bounds=bounds_h, initialize=h_0_dict)
model.eta = Var(model.n_set, model.t_set, bounds=bounds_eta, initialize=eta_0_dict)

# energy conservation constraint
q_tol = 1e-8
def energy_constraint_rule(model, j, t):
    return (
        K[j] * (model.q[j, t]+q_tol) * abs(model.q[j, t]+q_tol)**(n_exp[j] - 1)
        + sum(A12[j, i] * model.h[i, t] for i in nodes_map[j][0])
        + sum(A10[j, s] * h0_opt[s, t] for s in sources_map[j][0])
        + sum(A13[j, n] * model.eta[n, t] for n in model.n_set) == 0
    )
model.energy_constraints = Constraint(model.j_set, model.t_set, rule=energy_constraint_rule)

# mass conservation constraint
def mass_constraint_rule(model, i, t):
    return (sum(A12.T[i, j] * model.q[j, t] for j in links_map[i][0]) == d_opt[i, t])
model.mass_constraints = Constraint(model.i_set, model.t_set, rule=mass_constraint_rule)

# # prv direction constraint
# prv_tol = -1e-1
# def prv_direction_rule(model, n, t):
#     return (model.eta[n, t] * model.q[prv_idx[n], t] >= prv_tol)
# model.prv_constraints = Constraint(model.n_set, model.t_set, rule=prv_direction_rule)

# objective function
def objective_rule(model):
    return (
        (1 / nt) * sum(sum(azp_weights[i] * (model.h[i, t] - elevation[i]) for i in model.i_set) for t in model.t_set)
    )
model.objective = Objective(rule=objective_rule, sense=minimize)

solver = SolverFactory('ipopt')
solver.options['tol'] = 1e-3
solver.options['max_iter'] = 1000
solver.options['print_level'] = 5
# solver.options['linear_solver'] = 'ma57'
solver.options['mu_strategy'] = 'adaptive'
solver.options['mu_oracle'] = 'quality-function'
solver.options['warm_start_init_point'] = 'yes'
# solver.options['acceptable_tol'] = 1e-4
# solver.options['constr_viol_tol'] = 1e-4
# solver.options['compl_inf_tol'] = 1e-4

result = solver.solve(model, tee=True)

# get results
q_opt = np.reshape([value(model.q[i]) for i in model.q], (net_info['np'], nt))
column_names_q = [f'q_{t+1}' for t in range(nt)]
q_df = pd.DataFrame(q_opt, columns=column_names_q)
q_df.insert(0, 'link_id', link_df['link_ID'])

h_opt = np.reshape([value(model.h[i]) for i in model.h], (net_info['nn'], nt))
column_names_h = [f'h_{t+1}' for t in range(nt)]
h_df = pd.DataFrame(h_opt, columns=column_names_h)
h_df.insert(0, 'node_id', node_df['node_ID'])

eta_opt = np.reshape([value(model.eta[i]) for i in model.eta], (len(prv_idx), nt))
column_names_eta = [f'eta_{t+1}' for t in range(nt)]
eta_df = pd.DataFrame(eta_opt, columns=column_names_eta)
eta_df.insert(0, 'link_id', prv_links)

p_opt = h_opt - np.tile(elevation, (1, nt))
column_names_p = [f'p_{t+1}' for t in range(nt)]
p_df = pd.DataFrame(p_opt, columns=column_names_p)
p_df.insert(0, 'node_id', node_df['node_ID'])





###### STEP 7: plot optimal pressure results ######
# save and plot outlet pressure at PRVs
prv_data = []
for idx, prv_link in enumerate(prv_links):

    prv_id = prv_ids[idx]
    link_idx = prv_idx[idx]
    flow = q_opt[link_idx, :] * 1000

    start_node = link_df.loc[link_idx, 'node_out']
    end_node = link_df.loc[link_idx, 'node_in']
    start_node_idx = node_df[node_df['node_ID'] == start_node].index[0]
    end_node_idx = node_df[node_df['node_ID'] == end_node].index[0]
    inlet_p = h_opt[start_node_idx, :] - elevation[start_node_idx]
    outlet_p = h_opt[end_node_idx, :] - elevation[end_node_idx]
    prv_elev = elevation[end_node_idx][0]

    critical_node = critical_nodes[idx]
    critical_node_idx = node_df[node_df['node_ID'] == critical_node].index[0]
    critical_p = h_opt[critical_node_idx, :] - elevation[critical_node_idx]
    critical_elev = elevation[critical_node_idx]
    
    for t in range(nt):
        prv_data.append({
            'datetime': datetime[nt_range][t],
            'bwfl_id': prv_id,
            'prv_link': prv_link,
            'flow': flow[t],
            'inlet_pressure': inlet_p[t],
            'outlet_pressure': outlet_p[t],
            'prv_elevation': prv_elev,
            'critical_pressure': critical_p[t],
            'critical_elevation': critical_elev
        })

prv_data_df = pd.DataFrame(prv_data)


# data plots
fig = make_subplots(
    rows=len(prv_links), cols=1,
    shared_xaxes=False,
    vertical_spacing=0.1,
    subplot_titles=prv_ids,
    specs=[[{"secondary_y": True}] for _ in range(len(prv_links))]
)

for idx, prv_link in enumerate(prv_links):
    prv_df = prv_data_df[prv_data_df['prv_link'] == prv_link]
    
    fig.add_trace(
        go.Scatter(
            x=prv_df['datetime'],
            y=prv_df['outlet_pressure'],
            mode='lines',
            name='outlet pressure',
            line=dict(color=default_colors[0], dash='solid'),
            showlegend=False
        ),
        row=idx+1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=prv_df['datetime'],
            y=prv_df['critical_pressure'],
            mode='lines',
            name='critical pressure',
            line=dict(color=default_colors[0], dash='dash'),
            showlegend=False
        ),
        row=idx+1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=prv_df['datetime'],
            y=prv_df['flow'],
            mode='lines',
            name='flow',
            line=dict(color=default_colors[1], dash='solid'),
            showlegend=False
        ),
        row=idx+1, col=1, secondary_y=True
    )

    # Color the primary y-axis using default_colors[0] and secondary y-axis using default_colors[1]
    fig.update_yaxes(
        title_text="Pressure [m]", 
        title_font=dict(color=default_colors[0]),
        tickfont=dict(color=default_colors[0]),
        ticks="outside",
        tickcolor=default_colors[0],
        showline=True,
        linecolor=default_colors[0],
        row=idx+1, col=1, secondary_y=False
    )
    
    fig.update_yaxes(
        title_text="Flow [L/s]", 
        title_font=dict(color=default_colors[1]),
        tickfont=dict(color=default_colors[1]),
        ticks="outside",
        tickcolor=default_colors[1],
        showline=True,
        linecolor=default_colors[1],
        row=idx+1, col=1, secondary_y=True
    )

fig.update_layout(
    height=300 * len(prv_links),
    width=800,
    title_text="",
    template='simple_white'
)

fig.show()
fig.write_html(RESULTS_DIR / f"optimal-prv-settings-period-{str(data_period).zfill(2)}-day-{str(day).zfill(1)}.html")


# plot spatial pressure heads
plot_network(vals=p_df, val_type='pressure', t=8, inp_file='split')





###### STEP 8: create flow modulation profiles ######
# flow modulation curves
fm_curve_df = pd.DataFrame()
for idx, prv_link in enumerate(prv_links):

    fm_data = pd.DataFrame()
    flow = prv_data_df[prv_data_df['prv_link'] == prv_link]['flow'].ravel()
    outlet_p = prv_data_df[prv_data_df['prv_link'] == prv_link]['outlet_pressure'].ravel()

    coeff = np.polyfit(flow, outlet_p, 2)
    flow_min = min(flow)
    flow_max = max(flow)
    fm_curve_min = int((round(flow_min)*0.75))
    fm_curve_max = int((round(flow_max)*1.25))
    x = np.arange(0, fm_curve_max)

    fm_data['prv_id'] = [prv_ids[idx]] * len(x) 
    fm_data['prv_link'] = [prv_link] * len(x) 
    fm_data['flow'] = x
    fm_data['outlet_pressure'] = np.polyval(coeff, x)
    new_row = {
        'prv_id': prv_ids[idx],
        'prv_link': prv_link,
        'flow': 0,
        'outlet_pressure': fm_data['outlet_pressure'].max()
    }
    fm_data = pd.concat([pd.DataFrame([new_row]), fm_data], ignore_index=True)
    fm_curve_df = pd.concat([fm_curve_df, fm_data])

# save flow modulation curve to csv
fm_curve_df.to_csv(RESULTS_DIR / f'fm-curves-period-{str(data_period).zfill(2)}-day-{str(day).zfill(1)}.csv', index=False)

# plotting
fig = make_subplots(
    rows=len(prv_links), cols=1,
    shared_xaxes=False,
    vertical_spacing=0.1,
    subplot_titles=prv_ids,
)

for idx, prv_link in enumerate(prv_links):

    prv_df = prv_data_df[prv_data_df['prv_link'] == prv_link]
    fm_df = fm_curve_df[fm_curve_df['prv_link'] == prv_link]

    fig.add_trace(
        go.Scatter(
            x=prv_df['flow'],
            y=prv_df['outlet_pressure'],
            mode='markers',
            name='simulated',
            line=dict(color=default_colors[0]),
            showlegend=False
        ),
        row=idx+1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=flow_df[(flow_df['bwfl_id'] == prv_ids[idx]) & (flow_df['datetime'].isin(datetime_range))]['mean'],
            y=pressure_df[(pressure_df['bwfl_id'] == prv_ids[idx]+' (outlet)') & (pressure_df['datetime'].isin(datetime_range))]['mean'],
            mode='markers',
            name='data',
            line=dict(color='black'),
            showlegend=False
        ),
        row=idx+1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=fm_df['flow'],
            y=fm_df['outlet_pressure'],
            mode='lines',
            name='flow modulation curve',
            line=dict(color=default_colors[1]),
            showlegend=False
        ),
        row=idx+1, col=1,
    )
    fig.update_yaxes(
        title_text="Outlet pressure [m]", 
        ticks="outside",
        row=idx+1, col=1, secondary_y=False
    )
    fig.update_xaxes(
        title_text="Flow [L/s]", 
        ticks="outside",
        row=idx+1, col=1,
    )
fig.update_layout(
    legend_title_text='',
    template='simple_white',
    height=300 * len(prv_links),
    width=600
)

fig.show()
fig.write_html(RESULTS_DIR / f"fm-curves-period-{str(data_period).zfill(2)}-day-{str(day).zfill(1)}.html")