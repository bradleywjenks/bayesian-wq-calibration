from bayesian_wq_calibration.constants import NETWORK_DIR, DEVICE_DIR, INP_FILE, IV_CLOSE, IV_OPEN, IV_OPEN_PART
import wntr
import pandas as pd
import json
import numpy as np
import datetime

class SimResults:
    def __init__(self):
        self.flow = None
        self.velocity = None
        self.pressure = None
        self.head = None
        self.chlorine = None
        self.age = None


""""
    Main model simulation function using EPANET solver
"""

def model_simulation(flow_df, pressure_df, wq_df, sim_type='hydraulic', demand_resolution='dma', iv_status='closed', dbv_status='active', trace_node=None, flush_data=None):

    # 1. load network
    wn = wntr.network.WaterNetworkModel(NETWORK_DIR / INP_FILE)

    # 2. assign boundary heads at reservoir nodes
    wn, h0 = set_reservoir_heads(wn, pressure_df)

    # 3. assign control valve settings (or assign defaults)
    wn = set_control_settings(wn, flow_df, pressure_df, iv_status, dbv_status)

    # 4. scale demands based on DMA or WWMD resolution

    # 5. assign time information:
    datetime = flow_df['datetime'].unique()
    wn = set_simulation_time(wn, datetime)

    # 6. assign input chlorine values, if chemical simulation selected

    # 7. output results as structure
    sim_results = epanet_simulator(wn, sim_type)

    return sim_results





"""
    Helper functions to main model simulation function
"""

def set_simulation_time(wn, datetime):

    datetime = pd.to_datetime(datetime)
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
    wn.options.time.start_clocktime = wn.options.time.start_clocktime

    return wn



def epanet_simulator(wn, sim_type):

    wn.convert_controls_to_rules(priority=3)
    results = wntr.sim.EpanetSimulator(wn).run_sim()
    sim_results = SimResults()
    sim_results.flow = results.link['flowrate']
    sim_results.velocity = results.link['velocity']
    sim_results.pressure = results.node['pressure']
    sim_results.head = results.node['head']

    if sim_type == 'age':
        sim_results.age = results.node['age']
    elif sim_type == 'chlorine':
        sim_results.chlorine = results.node['chlorine']

    return sim_results



def set_reservoir_heads(wn, pressure_df):

    reservoir_nodes = wn.reservoir_name_list
    device_id = sensor_model_id('pressure')

    h0 = {}

    for idx, node in enumerate(reservoir_nodes):

        bwfl_id = device_id[device_id['model_id'] == node]['bwfl_id'].values[0]
        elev = device_id[device_id['model_id'] == node]['elev'].values[0]
        h0[idx] = pressure_df[pressure_df['bwfl_id'] == bwfl_id]['mean'].values + elev

        if np.isnan(h0[idx]).any():
            print(f"Error setting boundary head values at reservoir {node}. Default values used.")
        else:
            try:
                wn.add_pattern(node + "_h0", h0[idx])
                reservoir = wn.get_node(node)
                reservoir.head_timeseries.base_value = 1
                reservoir.head_timeseries.pattern_name = wn.get_pattern(node + "_h0")
            except:
                print(f"Error setting boundary head values at reservoir {node}. Default values used.")

    return wn, h0



def set_control_settings(wn, flow_df, pressure_df, iv_status, dbv_status):

    # load valve_info file
    with open(NETWORK_DIR / 'valve_info.json') as f:
        valve_info = json.load(f)

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
        dbv_K = []
        try:
            dbv_K = get_dbv_settings(wn, flow_df, pressure_df, dbv_links)
        except:
            print("Error setting DBV settings. Default values used.")
            unique_datetimes = np.unique(pressure_df['datetime'])
            dbv_K = np.where(((unique_datetimes.astype('datetime64[h]').astype(int) % 24) < 5) | ((unique_datetimes.astype('datetime64[h]').astype(int) % 24) > 21), IV_CLOSE, IV_OPEN_PART)
            dbv_K = np.tile(dbv_K, (2, 1))
        if any(np.isnan(value).sum() for value in dbv_K.values()):
            print("Error setting DBV settings. Default values used.")
            dbv_K = np.where(((unique_datetimes.astype('datetime64[h]').astype(int) % 24) < 5) | ((unique_datetimes.astype('datetime64[h]').astype(int) % 24) > 21), IV_CLOSE, IV_OPEN_PART)
            dbv_K = np.tile(dbv_K, (2, 1))

        for idx, link in enumerate(dbv_links):
            wn.get_link(link).initial_setting = dbv_K[idx][0]
            wn.get_link(link).initial_status = "Active"
        
        dbv_controls = []
        for t in np.arange(1, len(dbv_K)):
            for idx, link in enumerate(dbv_links):
                valve = wn.get_link(link)
                dbv_controls = f"{link}_dbv_setting_t_{t/4}"
                cond = wntr.network.controls.SimTimeCondition(wn, '=', t * 900)  # 900 seconds = 15 minutes
                act = wntr.network.controls.ControlAction(valve, "setting", dbv_K[idx][t])
                rule = wntr.network.controls.Rule(cond, [act], name=dbv_controls)
                wn.add_control(dbv_controls, rule)

    # set prv settings
    prv_links = valve_info['prv_link']

    return wn



def sensor_model_id(device_type):

    node_mapping = pd.read_excel(NETWORK_DIR / 'InfoWorks_to_EPANET_mapping.xlsx', sheet_name='Nodes')
    link_mapping = pd.read_excel(NETWORK_DIR / 'InfoWorks_to_EPANET_mapping.xlsx', sheet_name='Links')

    if device_type == 'pressure':
        device_df = pd.read_excel(DEVICE_DIR / 'pressure_device_database.xlsx')
        element_type = 'node'
        device_id = device_df[['BWFL ID', 'Asset ID', 'Final Elevation (m)']].drop_duplicates(subset=['BWFL ID'])
        device_id.rename(columns={'BWFL ID': 'bwfl_id', 'Asset ID': 'asset_id', 'Final Elevation (m)': 'elev'}, inplace=True)
    elif device_type == 'flow':
        device_df = pd.read_excel(DEVICE_DIR / 'flow_device_database.xlsx')
        element_type = 'link'
        device_id = device_df[['BWFL ID', 'Asset ID']].drop_duplicates(subset=['BWFL ID'])
        device_id.rename(columns={'BWFL ID': 'bwfl_id', 'Asset ID': 'asset_id'}, inplace=True)
    elif device_type == 'wq':
        device_df = pd.read_excel(DEVICE_DIR / 'wq_device_database.xlsx')
        element_type = 'node'
        device_id = device_df[['BWFL ID', 'Asset ID']].drop_duplicates(subset=['BWFL ID'])
        device_id.rename(columns={'BWFL ID': 'bwfl_id', 'Asset ID': 'asset_id'}, inplace=True)

    device_id['asset_id'] = device_id['asset_id'].astype(str)
    
    if element_type == 'node':
        mapping_dict = node_mapping.set_index('InfoWorks Node ID')['EPANET Node ID'].to_dict()
        device_id['model_id'] = device_id['asset_id'].map(mapping_dict)
    elif element_type == 'link':
        mapping_dict = link_mapping.set_index('InfoWorks Link ID')['EPANET Link ID'].to_dict()
        device_id['model_id'] = device_id['asset_id'].map(mapping_dict)


    return device_id



def get_dbv_settings(wn, flow_df, pressure_df, dbv_links):

    pressure_device_id = sensor_model_id('pressure')
    flow_device_id = sensor_model_id('flow')

    dbv_K = {}

    for idx, link in enumerate(dbv_links):

        start_node = wn.get_link(link).start_node_name
        start_bwfl_id = pressure_device_id[pressure_device_id['model_id'] == start_node]['bwfl_id'].values[0]
        start_pressure = pressure_df[pressure_df['bwfl_id'] == start_bwfl_id]['mean'].values

        end_node = wn.get_link(link).end_node_name
        end_bwfl_id = pressure_device_id[pressure_device_id['model_id'] == end_node]['bwfl_id'].values[0]
        end_pressure = pressure_df[pressure_df['bwfl_id'] == end_bwfl_id]['mean'].values

        dbv_bwfl_id = flow_device_id[flow_device_id['model_id'] == str(link)]['bwfl_id'].values[0]
        dbv_flow = flow_df[flow_df['bwfl_id'] == dbv_bwfl_id]['mean'].values / 1000 # convert to cms

        dbv_diam = wn.get_link(link).diameter
        dbv_value = (abs(start_pressure - end_pressure) * 2 * 9.81 * np.pi**2 * dbv_diam) / (abs(dbv_flow)**2 * 4**2)
        dbv_value = np.minimum(dbv_value, IV_CLOSE)
        dbv_value = np.maximum(dbv_value, IV_OPEN)

        dbv_K[idx] = dbv_value

    return dbv_K
