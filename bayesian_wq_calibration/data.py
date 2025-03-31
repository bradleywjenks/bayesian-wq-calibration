import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel
from typing import Any
from bayesian_wq_calibration.constants import NETWORK_DIR, TIMESERIES_DIR, DEVICE_DIR

os.environ['DYLD_LIBRARY_PATH'] = "/Users/bradwjenks/Code/tools/WNTR/wntr/epanet/libepanet/darwin-arm"
import wntr

class WDN(BaseModel):
    A12: Any
    A10: Any
    net_info: dict
    link_df: pd.DataFrame
    node_df: pd.DataFrame
    demand_df: pd.DataFrame
    h0_df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True



"""
Network functions
""" 
def load_network_data(inp_file):

    wn = wntr.network.WaterNetworkModel(inp_file)
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # get network elements and simulation info
    nt = int(wn.options.time.duration / wn.options.time.report_timestep) + 1
    nt = nt if nt>0 else 1
    net_info = dict(
        np=wn.num_links,
        nn=wn.num_junctions,
        n0=wn.num_reservoirs,
        nt=nt,
        headloss=wn.options.hydraulic.headloss,
        units=wn.options.hydraulic.inpfile_units,
        reservoir_names=wn.reservoir_name_list,
        junction_names=wn.junction_name_list,
        pipe_names=wn.pipe_name_list,
        valve_names=wn.valve_name_list,
        prv_names=wn.prv_name_list
    )

    
    # extract link data
    if net_info['headloss'] == 'H-W':
        n_exp = 1.852
    elif net_info['headloss'] == 'D-W':
        n_exp = 2

    link_df = pd.DataFrame(
        index=pd.RangeIndex(net_info['np']),
        columns=['link_ID', 'link_type', 'diameter', 'length', 'n_exp', 'C', 'node_out', 'node_in'],
    ) # NB: 'C' denotes roughness or HW coefficient for pipes and local (minor) loss coefficient for valves

    def link_dict(link):
        if isinstance(link, wntr.network.Pipe):  # check if the link is a pipe
            return dict(
                link_ID=link.name,
                link_type='pipe',
                diameter=link.diameter,
                length=link.length,
                n_exp=n_exp,
                C=link.roughness,
                node_out=link.start_node_name,
                node_in=link.end_node_name
            )
        elif isinstance(link, wntr.network.Valve): # check if the link is a valve
            return dict(
                link_ID=link.name,
                link_type='valve',
                diameter=link.diameter,
                length=2*link.diameter,
                n_exp=2,
                C=link.minor_loss,
                node_out=link.start_node_name,
                node_in=link.end_node_name
            )
        
    for idx, link in enumerate(wn.links()):
        link_df.loc[idx] = link_dict(link[1])

    
    # extract node data
    node_df = pd.DataFrame(
        index=pd.RangeIndex(wn.num_nodes), columns=["node_ID", "elev", "xcoord", "ycoord"]
    )

    def node_dict(node):
        if isinstance(node, wntr.network.elements.Reservoir):
            elev = 0
        else:
            elev = node.elevation
        return dict(
            node_ID=node.name,
            elev=elev,
            xcoord=node.coordinates[0],
            ycoord=node.coordinates[1]
        )

    for idx, node in enumerate(wn.nodes()):
        node_df.loc[idx] = node_dict(node[1])


    # compute graph data
    A = np.zeros((net_info['np'], net_info['nn']+net_info['n0']), dtype=int)
    for k, row in link_df.iterrows():
        out_name = row['node_out']
        out_idx = node_df[node_df['node_ID']==out_name].index[0]
        in_name = row['node_in']
        in_idx = node_df[node_df['node_ID']==in_name].index[0]
        A[k, out_idx] = -1
        A[k, in_idx] = 1
        
    junction_idx = node_df.index[node_df['node_ID'].isin(net_info['junction_names'])].tolist()
    reservoir_idx = node_df.index[node_df['node_ID'].isin(net_info['reservoir_names'])].tolist()
    A12 = A[:, junction_idx]; A12 = sp.csr_matrix(A12) # link-junction incident matrix
    A10 = A[:, reservoir_idx]; A10 = sp.csr_matrix(A10) # link-reservoir indicent matrix

    # extract demand data
    demand_df = results.node['demand'].T * 1000 # Lps
    col_names = [f'demands_{t}' for t in range(1, len(demand_df.columns)+1)]
    demand_df.columns = col_names
    demand_df.reset_index(drop=False, inplace=True)
    demand_df = demand_df.rename(columns={'name': 'node_ID'})
    demand_df = demand_df[~demand_df['node_ID'].isin(net_info['reservoir_names'])] # delete reservoir nodes

    # extract boundary data
    h0_df = results.node['head'].T
    col_names = [f'h0_{t}' for t in range(1, len(h0_df.columns)+1)]
    h0_df.columns = col_names
    h0_df.reset_index(drop=False, inplace=True)
    h0_df = h0_df.rename(columns={'name': 'node_ID'})
    h0_df = h0_df[h0_df['node_ID'].isin(net_info['reservoir_names'])] # only reservoir nodes

    # load data to WDN object
    wdn = WDN(
            A12=A12,
            A10=A10,
            net_info=net_info,
            link_df=link_df,
            node_df=node_df,
            demand_df=demand_df,
            h0_df=h0_df
    )

    return wdn



def sensor_model_id(device_type):

    node_mapping = pd.read_excel(NETWORK_DIR / 'InfoWorks_to_EPANET_mapping.xlsx', sheet_name='Nodes')
    link_mapping = pd.read_excel(NETWORK_DIR / 'InfoWorks_to_EPANET_mapping.xlsx', sheet_name='Links')

    if device_type == 'pressure':
        device_df = pd.read_excel(DEVICE_DIR / 'pressure_device_database.xlsx')
        element_type = 'node'
        device_id = device_df[['BWFL ID', 'Asset ID', 'DMA ID', 'Final Elevation (m)']].drop_duplicates(subset=['BWFL ID'])
        device_id.rename(columns={'BWFL ID': 'bwfl_id', 'Asset ID': 'asset_id', 'DMA ID': 'dma_id', 'Final Elevation (m)': 'elev'}, inplace=True)
        device_id = device_id[device_id['dma_id'].isin([2296, 2005])]
    elif device_type == 'flow':
        device_df = pd.read_excel(DEVICE_DIR / 'flow_device_database.xlsx')
        element_type = 'link'
        device_id = device_df[['BWFL ID', 'Asset ID', 'DMA ID']].drop_duplicates(subset=['BWFL ID'])
        device_id.rename(columns={'BWFL ID': 'bwfl_id', 'Asset ID': 'asset_id', 'DMA ID': 'dma_id'}, inplace=True)
    elif device_type == 'wq':
        device_df = pd.read_excel(DEVICE_DIR / 'wq_device_database.xlsx')
        element_type = 'node'
        device_id = device_df[['BWFL ID', 'Asset ID']].drop_duplicates(subset=['BWFL ID'])
        device_id.rename(columns={'BWFL ID': 'bwfl_id', 'Asset ID': 'asset_id', 'DMA ID': 'dma_id'}, inplace=True)

    device_id['asset_id'] = device_id['asset_id'].astype(str)
    
    if element_type == 'node':
        mapping_dict = node_mapping.set_index('InfoWorks Node ID')['EPANET Node ID'].to_dict()
        device_id['model_id'] = device_id['asset_id'].map(mapping_dict)
    elif element_type == 'link':
        mapping_dict = link_mapping.set_index('InfoWorks Link ID')['EPANET Link ID'].to_dict()
        device_id['model_id'] = device_id['asset_id'].map(mapping_dict)


    return device_id




"""
Time series data functions
""" 

def get_sensor_stats(data_type, sensor_names, N=19):

    sensor_df = pd.DataFrame()

    if data_type == 'temperature' or data_type == 'ph':
        for idx in range(1, N+1):
            data_df = None
            data_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(idx).zfill(2)}-wq.csv", low_memory=False)
            if data_df is not None:
                # print(f"Data frame {idx} was successfully read.")
                sensor_df = pd.concat([sensor_df, data_df])
        sensor_df = sensor_df[sensor_df['data_type'] == data_type]
        if data_type == 'ph':
            sensor_df.loc[(sensor_df['mean'] < 4) | (sensor_df['mean'] > 11), 'mean'] = np.nan
    elif data_type == 'flow':
        for idx in range(1, N+1):
            data_df = None
            data_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(idx).zfill(2)}-flow.csv")
            if data_df is not None:
                # print(f"Data frame {idx} was successfully read.")
                sensor_df = pd.concat([sensor_df, data_df])
    elif data_type == 'pressure':
        for idx in range(1, N+1):
            data_df = None
            data_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(idx).zfill(2)}-pressure.csv")
            if data_df is not None:
                # print(f"Data frame {idx} was successfully read.")
                sensor_df = pd.concat([sensor_df, data_df])

    try:
        sensor_df = sensor_df[sensor_df['bwfl_id'].isin(sensor_names)]
    except:
        sensor_df = []
        print(f"Sensor names not found in {data_type} time series data.")

    # sensor_df.dropna(subset=['mean'], inplace=True)
    sensor_df.drop_duplicates(subset=['datetime', 'bwfl_id'], inplace=True)

    stats = sensor_df.groupby('bwfl_id')['mean'].describe(percentiles=[.01, .10, .25, .50, .75, .90, .99])

    return stats



def count_pressure_events(threshold=10, N=20):

    count = []
    sensor_data = sensor_model_id('pressure')
    bwfl_ids = sensor_data['bwfl_id'].unique()

    for idx in range(1, N+1):
        data_df = None
        data_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(idx).zfill(2)}-pressure.csv")
        data_df = data_df[data_df['bwfl_id'].isin(bwfl_ids)]
        if data_df is not None:
            count_idx = data_df[data_df['max'] - data_df['min'] > threshold].groupby('bwfl_id')
            if count_idx.ngroups != 0:
                count.append(count_idx.size().max())
            else:
                count.append(np.nan)
        else:
            print("Error loading data.")
            break

    return count



def count_turbidity_events(threshold=2, N=20):

    sensor_data = sensor_model_id('wq')
    bwfl_ids = sensor_data['bwfl_id'].unique()

    count_df = pd.DataFrame()

    for idx in range(1, N+1):
        data_df = None
        data_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(idx).zfill(2)}-wq.csv")
        data_df = data_df[(data_df['bwfl_id'].isin(bwfl_ids)) & (data_df['data_type'] == 'turbidity')]

        if data_df is not None:
            count_idx_df = pd.DataFrame({'bwfl_id': bwfl_ids})
            count_idx_df['data_period'] = idx
            for bwfl_id in bwfl_ids:
                count_idx_df.loc[count_idx_df['bwfl_id'] == bwfl_id, 'num_events'] = data_df[(data_df['bwfl_id'] == bwfl_id) & (data_df['mean'] > threshold)].shape[0]

            count_df = pd.concat([count_df, count_idx_df])

        else:
            print("Error loading data.")
            break

    count_df.reset_index(drop=True, inplace=True)

    return count_df




"""
Bulk water testing function
"""
def bulk_temp_adjust(bulk_coeff, field_temp):
    test_temp = 20 + 273.15 # test temperature (change accordingly)
    field_temp = field_temp + 273.15
    Ea_R_ratio = 6000 # ratio of activation energy to ideal gas constant for chlorine decay in water networks (AWWARF, 1996)
    bulk_adjust = bulk_coeff * np.exp((-Ea_R_ratio * (test_temp - field_temp)) / (test_temp * field_temp))

    return bulk_adjust




