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

    # 2. assign head data at 2005 and 2296 inlets

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

    wn.options.time.duration = total_duration + time_step
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