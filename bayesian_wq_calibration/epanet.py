from bayesian_wq_calibration.constants import NETWORK_DIR, INP_FILE, IV_CLOSE, IV_OPEN, IV_OPEN_PART
from bayesian_wq_calibration.data import sensor_model_id
import wntr
import pandas as pd
import json
import numpy as np
import datetime
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.WARNING)

class SimResults:
    def __init__(self):
        self.datetime = None
        self.flow = None
        self.velocity = None
        self.pressure = None
        self.head = None
        self.chlorine = None
        self.age = None
        self.trace = None
        self.demand = None


""""
    Main model simulation function using EPANET solver
"""

def model_simulation(flow_df, pressure_df, cl_df, sim_type='hydraulic', demand_resolution='wwmd', iv_status='closed', dbv_status='active', trace_node=None, grouping='single', wall_coeffs={'single':-0.3}, bulk_coeff=-0.5, flush_data=None):

    # 1. build network model
    wn = build_model(flow_df, pressure_df, cl_df, sim_type=sim_type, demand_resolution=demand_resolution, iv_status=iv_status, dbv_status=dbv_status, trace_node=trace_node, grouping=grouping, wall_coeffs=wall_coeffs, bulk_coeff=bulk_coeff, flush_data=flush_data)
    wn.options.hydraulic.accuracy = 1e-3

    # 2. output results as structure
    sim_results = epanet_simulator(wn, sim_type, cl_df)

    return sim_results


def build_model(flow_df, pressure_df, cl_df, sim_type='hydraulic', demand_resolution='wwmd', iv_status='closed', dbv_status='active', trace_node=None, grouping='single', wall_coeffs={'single':-0.3}, bulk_coeff=-0.5, flush_data=None):

    # 1. load network
    wn = wntr.network.WaterNetworkModel(NETWORK_DIR / INP_FILE)

    # 2. assign boundary heads at reservoir nodes
    wn, h0 = set_reservoir_heads(wn, pressure_df)

    # 3. assign control valve settings (or assign defaults)
    wn = set_control_settings(wn, flow_df, pressure_df, iv_status, dbv_status)

    # 4. scale demands based on DMA or WWMD resolution
    wn = scale_demand(wn, flow_df, demand_resolution, flush_data) # NB: remove flushing demands from demand scaling

    # 5. assign time information:
    wn = set_simulation_time(wn, cl_df)

    # 6. set water quality simulation:
    if grouping == 'material-velocity':
        sim_results = epanet_simulator(wn, 'velocity', cl_df)
        vel_sim = sim_results.velocity.T
        mean_vel = vel_sim.mean(axis=1)
        mean_vel = mean_vel.reset_index().rename(columns={'name': 'link_id', 0: 'mean_vel'})
    else:
        mean_vel = None
    wn = set_wq_parameters(wn, sim_type, cl_df, trace_node, grouping, wall_coeffs, bulk_coeff, mean_vel)

    return wn




"""
    Helper functions to main model simulation function
"""

def set_simulation_time(wn, cl_df):

    datetime = pd.to_datetime(cl_df['datetime'].unique())
    time_step = (datetime[1] - datetime[0]).total_seconds()
    total_duration = (datetime.max() - datetime.min()).total_seconds()

    wn.options.time.duration = total_duration
    wn.options.time.hydraulic_timestep = time_step
    wn.options.time.quality_timestep = min(time_step, 60*5)
    wn.options.time.report_timestep = time_step
    wn.options.time.rule_timestep = time_step
    wn.options.time.pattern_timestep = time_step
    wn.options.time.pattern_start = 0
    wn.options.time.report_start = 0

    return wn



def epanet_simulator(wn, sim_type, cl_df):

    # print('Simulating!')

    datetime = pd.to_datetime(cl_df['datetime'].unique())
    results = wntr.sim.EpanetSimulator(wn).run_sim()
    sim_results = SimResults()
    sim_results.datetime = datetime
    sim_results.flow = results.link['flowrate'] * 1000 # convert to Lps 
    sim_results.velocity = results.link['velocity']
    sim_results.pressure = results.node['pressure']
    sim_results.head = results.node['head']

    if sim_type == 'age':
        sim_results.age = results.node['quality'] / 3600 # convert to hours
    elif sim_type == 'chlorine':
        sim_results.chlorine = results.node['quality']
    elif sim_type == 'trace':
        sim_results.trace = results.node['quality']

    return sim_results



def set_wq_parameters(wn, sim_type, cl_df, trace_node, grouping, wall_coeffs, bulk_coeff, mean_vel):

    if sim_type == 'age':
        wn.options.quality.parameter = "AGE"
    elif sim_type == 'chlorine':
        wn.options.quality.parameter = "CHEMICAL"
        wn = set_source_cl(wn, cl_df)
        wn = set_reaction_parameters(wn, grouping, wall_coeffs, bulk_coeff, mean_vel)
    elif sim_type == 'trace' and trace_node is not None:
        wn.options.quality.parameter = 'TRACE'
        if trace_node in wn.reservoir_name_list:
            wn.options.quality.trace_node = trace_node
        else:
            logging.error(f"Error. {trace_node} is not a reservoir.")
    else:
        wn.options.quality.parameter = "NONE"

    return wn



def set_source_cl(wn, cl_df):

    reservoir_nodes = wn.reservoir_name_list
    device_id = sensor_model_id('wq')
    datetime = pd.to_datetime(cl_df['datetime'].unique())

    cl0 = {}

    for idx, node in enumerate(reservoir_nodes):

        bwfl_id = device_id[device_id['model_id'] == node]['bwfl_id'].values[0]
        cl0[idx] = cl_df[cl_df['bwfl_id'] == str(bwfl_id)]['mean'].values

        if np.isnan(cl0[idx]).any():
            logging.info(f"Error setting source chlorine values at reservoir {node}. Default values used.")
            cl0[idx] = [0.8] * len(datetime)
        else:
            try:
                wn.add_pattern(node + "_cl", cl0[idx][1:])
                wn.add_source(node + "_cl", node, "CONCEN", 1, node + "_cl")
            except:
                logging.info(f"Error setting source chlorine values at reservoir {node}. Default values used.")
                cl0[idx] = [0.8] * len(datetime)
                wn.add_pattern(node + "_cl", cl0[idx][1:])
                wn.add_source(node + "_cl", node, "CONCEN", 1, node + "_cl")

        reservoir_data = wn.get_node(node)
        reservoir_data.initial_quality = cl0[idx][0]

    
    return wn



def set_reaction_parameters(wn, grouping, wall_coeffs, bulk_coeff, mean_vel):

    if bulk_coeff is not None:
        wn.options.reaction.bulk_coeff = (bulk_coeff/3600/24) # (FROM BW)

    if grouping == 'single':
        wn.options.reaction.wall_coeff = (wall_coeffs['single']/3600/24)

    elif grouping == 'material':
        material_df = pd.read_excel(NETWORK_DIR / 'gis_data.xlsx')
        for name, link in wn.links():
            if isinstance(link, wntr.network.Pipe):
                material = material_df[material_df['model_id'] == name]['material'].values
                if material[0] in ['CI', 'SI', 'Pb', 'DI', 'ST']:
                    link.wall_coeff = wall_coeffs['metallic']/3600/24
                elif material[0] in ['AC']:
                    link.wall_coeff = wall_coeffs['cement']/3600/24
                elif material[0] in ['HPPE', 'HPPE+FOIL', 'LDPE', 'MDPE', 'MDPE+FOIL', 'PE100+Skin', 'PVC', 'Unknown']:
                    link.wall_coeff = wall_coeffs['plastic_unknown']/3600/24

    elif grouping == 'material-diameter':
        material_df = pd.read_excel(NETWORK_DIR / 'gis_data.xlsx')
        for name, link in wn.links():
            if isinstance(link, wntr.network.Pipe):
                material = material_df[material_df['model_id'] == name]['material'].values
                if material[0] in ['CI', 'SI', 'Pb', 'DI', 'ST'] and link.diameter * 1000 < 150:
                    link.wall_coeff = wall_coeffs['metallic_less_than_150']/3600/24
                elif material[0] in ['CI', 'SI', 'Pb', 'DI', 'ST'] and link.diameter * 1000 >= 150:
                    link.wall_coeff = wall_coeffs['metallic_greater_than_150']/3600/24
                elif material[0] in ['AC']:
                    link.wall_coeff = wall_coeffs['cement']/3600/24
                elif material[0] in ['HPPE', 'HPPE+FOIL', 'LDPE', 'MDPE', 'MDPE+FOIL', 'PE100+Skin', 'PVC', 'Unknown']:
                    link.wall_coeff = wall_coeffs['plastic_unknown']/3600/24

    elif grouping == 'material-velocity' and mean_vel is not None:
        _n_perc_vel = mean_vel['mean_vel'].quantile(0.6)
        material_df = pd.read_excel(NETWORK_DIR / 'gis_data.xlsx')
        for name, link in wn.links():
            if isinstance(link, wntr.network.Pipe):
                material = material_df[material_df['model_id'] == name]['material'].values
                mean_vel_link = mean_vel[mean_vel['link_id'] == name]['mean_vel'].values
                if material[0] in ['CI', 'SI', 'Pb', 'DI', 'ST'] and mean_vel_link < _n_perc_vel:
                    link.wall_coeff = wall_coeffs['metallic_low_velocity']/3600/24
                elif material[0] in ['CI', 'SI', 'Pb', 'DI', 'ST'] and mean_vel_link >= _n_perc_vel:
                    link.wall_coeff = wall_coeffs['metallic_high_velocity']/3600/24
                elif material[0] in ['AC'] and mean_vel_link < _n_perc_vel:
                    link.wall_coeff = wall_coeffs['cement_low_velocity']/3600/24
                elif material[0] in ['AC'] and mean_vel_link >= _n_perc_vel:
                    link.wall_coeff = wall_coeffs['cement_high_velocity']/3600/24
                elif material[0] in ['HPPE', 'HPPE+FOIL', 'LDPE', 'MDPE', 'MDPE+FOIL', 'PE100+Skin', 'PVC', 'Unknown'] and mean_vel_link < _n_perc_vel:
                    link.wall_coeff = wall_coeffs['plastic_low_velocity']/3600/24
                elif material[0] in ['HPPE', 'HPPE+FOIL', 'LDPE', 'MDPE', 'MDPE+FOIL', 'PE100+Skin', 'PVC', 'Unknown'] and mean_vel_link >= _n_perc_vel:
                    link.wall_coeff = wall_coeffs['plastic_high_velocity']/3600/24
    else:
        print('Wall grouping not valid.')
    return wn



def set_reservoir_heads(wn, pressure_df):

    reservoir_nodes = wn.reservoir_name_list
    device_id = sensor_model_id('pressure')

    h0 = {}

    for idx, node in enumerate(reservoir_nodes):

        bwfl_id = device_id[device_id['model_id'] == node]['bwfl_id'].values[0]
        elev = device_id[device_id['model_id'] == node]['elev'].values[0]
        h0[idx] = pressure_df[pressure_df['bwfl_id'] == bwfl_id]['mean'].values + elev
        h0[idx] = np.round(h0[idx], 2)

        if np.isnan(h0[idx]).any():
            logging.info(f"Error setting boundary head values at reservoir {node}. Default values used.")
        else:
            try:
                wn.add_pattern(node + "_h0", h0[idx])
                reservoir = wn.get_node(node)
                reservoir.head_timeseries.base_value = 1
                reservoir.head_timeseries.pattern_name = wn.get_pattern(node + "_h0")
            except:
                logging.info(f"Error setting boundary head values at reservoir {node}. Default values used.")

    return wn, h0



def set_control_settings(wn, flow_df, pressure_df, iv_status, dbv_status):

    # load valve_info file
    with open(NETWORK_DIR / 'valve_info.json') as f:
        valve_info = json.load(f)

    datetime = pd.to_datetime(flow_df['datetime'].unique())
    # datetime = np.array(datetime, dtype='datetime64[h]')

    pressure_device_id = sensor_model_id('pressure')
    flow_device_id = sensor_model_id('flow')

    # set iv settings
    iv_links = valve_info['iv_link']
    if iv_status == 'closed':
        for link in iv_links:
            wn.get_link(link).initial_setting = IV_CLOSE
            wn.get_link(link).initial_status = "Active"

    elif iv_status == 'open':
        for link in iv_links:
            wn.get_link(link).initial_setting = IV_OPEN
            wn.get_link(link).initial_status = "Active"

    # set dbv settings
    dbv_links = valve_info['dbv_link']
    if dbv_status == 'active':
        dbv_K_default = np.where(((datetime.hour < 5) | (datetime.hour > 21)), IV_CLOSE, IV_OPEN_PART)
        dbv_K_default = np.tile(dbv_K_default, (2, 1))
        
        dbv_K = get_dbv_settings(wn, flow_df, pressure_df, dbv_links)

        for idx, link in enumerate(dbv_links):

            if any(np.isnan(value).any() for value in dbv_K[idx]):
                logging.info(f"Error setting DBV settings @ {link}. Default values used.")
                dbv_K_idx = dbv_K_default[idx] // 1
            else:
                dbv_K_idx = dbv_K[idx] // 1

            valve = wn.get_link(link)
            wn.get_link(link).initial_setting = 1e-4
            wn.get_link(link).initial_status = "Active"
        
            dbv_controls = []
            for t in range(len(datetime)):
                dbv_controls = f"{link}_dbv_setting_t_{t/4}"
                cond = wntr.network.controls.SimTimeCondition(wn, '=', t*900)  # 900 seconds = 15 minutes
                act = wntr.network.controls.ControlAction(valve, "setting", dbv_K_idx[t])
                rule = wntr.network.controls.Rule(cond, [act], name=dbv_controls)
                wn.add_control(dbv_controls, rule)

    # set prv settings
    prv_links = valve_info['prv_link']
    prv_dir = valve_info['prv_dir']
    prv_settings_default = {i: np.tile(np.array(valve_info['prv_settings'][i], dtype=float), len(datetime)) for i in range(len(prv_links))}

    prv_settings = get_prv_settings(wn, pressure_df, prv_links, prv_dir)

    for idx, link in enumerate(prv_links):
        if any(np.isnan(value).any() for value in prv_settings[idx]):
            logging.info(f"Error setting PRV settings @ {link}. Default values used.")
            prv_settings_idx = prv_settings_default[idx]
        else:
            prv_settings_idx = prv_settings[idx]

        prv_settings_idx = np.round(prv_settings_idx, 2)

        valve = wn.get_link(link)
        wn.remove_link(link)
        if prv_dir[idx] == 1:
            wn.add_valve(link, valve.start_node_name, valve.end_node_name, diameter=valve.diameter, valve_type="PRV", minor_loss=0.0001, initial_setting=prv_settings_idx[0], initial_status="Active")
        elif prv_dir[idx] == -1:
            wn.add_valve(link, valve.end_node_name, valve.start_node_name, diameter=valve.diameter, valve_type="PRV", minor_loss=0.0001, initial_setting=prv_settings_idx[0], initial_status="Active")

        prv_controls = []
        for t in range(len(datetime)):
            prv_controls = f"{link}_prv_setting_t_{t/4}"
            cond = wntr.network.controls.SimTimeCondition(wn, '=', t*900)  # 900 seconds = 15 minutes
            act = wntr.network.controls.ControlAction(valve, "setting", prv_settings_idx[t])
            rule = wntr.network.controls.Rule(cond, [act], name=prv_controls)
            wn.add_control(prv_controls, rule)

    return wn



def scale_demand(wn, flow_df, demand_resolution, flush_data):

    datetime = pd.to_datetime(flow_df['datetime'].unique())
    flow_device_id = sensor_model_id('flow')

    # compute zonal flow balance
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
                logging.info(f"Significant periods of missing data for {demand_resolution} demand resolution. Switching to {demand_resolution} demand resolution")
                break


    ##### REMOVE KNOWN FLUSHING DEMANDS FROM INFLOW_DF #####
    # insert code here...

    # scale demands
    junction_names = wn.junction_name_list
    node_mapping = pd.read_excel(NETWORK_DIR / 'InfoWorks_to_EPANET_mapping.xlsx', sheet_name='Nodes')
    demand_df = pd.DataFrame({'node_id': junction_names, 'dma_id': None, 'wwmd_id': None, 'base_demand': 0.0, 'old_pattern': None})

    for name in junction_names:

        junction = wn.get_node(name)
        base_demand = 0
        patterns = []

        for idx in range(len(junction.demand_timeseries_list)):
            try:
                patt_name = junction.demand_timeseries_list[idx].pattern_name
                if patt_name is not None:
                    patterns.append(junction.demand_timeseries_list[idx].pattern_name)
                    base_demand += junction.demand_timeseries_list[idx].base_value
                else:
                    junction.demand_timeseries_list[idx].base_value = 0
                    junction.demand_timeseries_list[idx].pattern_name = None
            except:
                patterns.append([None])

        if patterns:
            demand_df.loc[demand_df['node_id'] == name, 'old_pattern'] = patterns[0]
        else:
            demand_df.loc[demand_df['node_id'] == name, 'old_pattern'] = None

        demand_df.loc[demand_df['node_id'] == name, 'base_demand'] = base_demand
        demand_df.loc[demand_df['node_id'] == name, 'dma_id'] = str(node_mapping[node_mapping['EPANET Node ID'] == name]['DMA ID'].values[0])
        demand_df.loc[demand_df['node_id'] == name, 'wwmd_id'] = str(node_mapping[node_mapping['EPANET Node ID'] == name]['WWMD ID'].values[0])

    scale_df = pd.DataFrame(index=inflow_df.index, columns=flow_balance.keys())

    for key in flow_balance.keys():

        if demand_resolution == 'dma':
            total_base_demand = demand_df[demand_df['dma_id'] == key]['base_demand'].sum() * 1000
        elif demand_resolution == 'wwmd':
            total_base_demand = demand_df[demand_df['wwmd_id'] == str(key)]['base_demand'].sum() * 1000

        if total_base_demand > 0:
            scale_df[key] = inflow_df[key] / total_base_demand
        else:
            scale_df[key] = 0

    for name, col in scale_df.items():
        if not np.isnan(col.iloc[0]):
            wn.add_pattern(name, col.values)

    for idx, name in enumerate(junction_names):
        junction = wn.get_node(name)
        if demand_resolution == 'dma':
            new_patt = demand_df[demand_df['node_id'] == name]['dma_id'].values[0]
        elif demand_resolution == 'wwmd':
            new_patt = demand_df[demand_df['node_id'] == name]['wwmd_id'].values[0]
        base = demand_df[demand_df['node_id'] == name]['base_demand'].values[0]
    
        if new_patt in wn.pattern_name_list:
            while len(junction.demand_timeseries_list) < 2:
                junction.add_demand(0.0, None)
            junction.demand_timeseries_list[0].pattern_name = None
            junction.demand_timeseries_list[0].base_value = 0.0
            junction.demand_timeseries_list[1].pattern_name = new_patt
            junction.demand_timeseries_list[1].base_value = base


    ##### ADD KNOWN FLUSHING DEMANDS AS ADDITIONAL DEMAND #####
    # insert code here...

    wn.convert_controls_to_rules(priority=3)

    return wn




def get_dbv_settings(wn, flow_df, pressure_df, dbv_links):

    pressure_device_id = sensor_model_id('pressure')
    flow_device_id = sensor_model_id('flow')

    dbv_K = {}

    for idx, link in enumerate(dbv_links):

        try:

            start_node = wn.get_link(link).start_node_name
            start_bwfl_id = pressure_device_id[pressure_device_id['model_id'] == start_node]['bwfl_id'].values[0]
            start_pressure = pressure_df[pressure_df['bwfl_id'] == start_bwfl_id]['mean'].values

            end_node = wn.get_link(link).end_node_name
            end_bwfl_id = pressure_device_id[pressure_device_id['model_id'] == end_node]['bwfl_id'].values[0]
            end_pressure = pressure_df[pressure_df['bwfl_id'] == end_bwfl_id]['mean'].values

            dbv_bwfl_id = flow_device_id[flow_device_id['model_id'] == str(link)]['bwfl_id'].values[0]
            dbv_flow = flow_df[flow_df['bwfl_id'] == dbv_bwfl_id]['mean'].values / 1000 # convert to cms

            dbv_diam = wn.get_link(link).diameter
            dbv_value = (abs(start_pressure - end_pressure) * 2 * 9.81 * np.pi**2 * dbv_diam**4) / (abs(dbv_flow)**2 * 4**2)
            dbv_value = np.minimum(dbv_value, IV_CLOSE)
            dbv_value = np.maximum(dbv_value, IV_OPEN)

            dbv_K[idx] = dbv_value

        except:
            dbv_K[idx] = [np.nan] * len(pressure_df['datetime'].unique())

    return dbv_K



def get_prv_settings(wn, pressure_df, prv_links, prv_dir):

    pressure_device_id = sensor_model_id('pressure')

    prv_settings = {}

    for idx, link in enumerate(prv_links):

        try:
            if prv_dir[idx] == 1:
                node = wn.get_link(link).end_node_name
                bwfl_id = pressure_device_id[pressure_device_id['model_id'] == node]['bwfl_id'].values[0]
                pressure = pressure_df[pressure_df['bwfl_id'] == bwfl_id]['mean'].values
            elif prv_dir[idx] == -1:
                node = wn.get_link(link).start_node_name
                bwfl_id = pressure_device_id[pressure_device_id['model_id'] == node]['bwfl_id'].values[0]
                pressure = pressure_df[pressure_df['bwfl_id'] == bwfl_id]['mean'].values
            prv_settings[idx] = pressure
        except:
            prv_settings[idx] = [np.nan] * len(pressure_df['datetime'].unique())

    return prv_settings





