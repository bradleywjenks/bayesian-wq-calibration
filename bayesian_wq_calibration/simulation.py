from bayesian_wq_calibration.constants import NETWORK_DIR, DEVICE_DIR, INP_FILE
import wntr
import pandas as pd

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

def model_simulation(flow_df, pressure_df, wq_df, sim_type='hydraulic', demand_res='dma', trace_node=None, flush_data=None):

    # 1. load network
    wn = wntr.network.WaterNetworkModel(NETWORK_DIR / INP_FILE)

    # 2. assign boundary heads at reservoir nodes
    wn, h0 = set_reservoir_heads(wn, pressure_df)

    # 3. assign pressure reducing valve outlet pressures (or assign defaults)

    # 4. compute dynamic boundary valve loss coefficients (or assign defaults)

    # 5. scale demands based on DMA or WWMD resolution

    # 6. assign time information:
    datetime = flow_df['datetime'].unique()
    wn = set_simulation_time(wn, datetime)

    # 7. assign input chlorine values, if chemical simulation selected

    # 8. output results as structure

    # Create a results structure
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

        # get bounday head patterns
        bwfl_id = device_id[device_id['model_id'] == node]['bwfl_id'].values[0]
        elev = device_id[device_id['model_id'] == node]['elev'].values[0]
        h0[idx] = pressure_df[pressure_df['bwfl_id'] == bwfl_id]['mean'].values + elev

        # set boundary head values in wn
        wn.add_pattern(node + "_h0", h0[idx])
        reservoir = wn.get_node(node)
        reservoir.head_timeseries.base_value = 1
        reservoir.head_timeseries.pattern_name = wn.get_pattern(node + "_h0")

    return wn, h0



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

