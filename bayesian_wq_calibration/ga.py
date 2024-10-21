from bayesian_wq_calibration.simulation import model_simulation, sensor_model_id
import pandas as pd
import numpy as np


"""
Genetic algorithm fitness function
"""
def fitness(flow_df, pressure_df, cl_df, wall_coeffs, obj_function='mse', wall_grouping='single', bulk_coeff=-0.55, demand_resolution='dma'):

    # translate decision variables to dict for model simulation
    if wall_grouping == 'single':
        wall_coeffs_dict = {'single': wall_coeffs[0]}
    elif wall_grouping == 'diameter-based':
        wall_coeffs = {
            'less_than_75': wall_coeffs[0],
            'between_75_and_150': wall_coeffs[1],
            'between_150_and_250': wall_coeffs[2],
            'greater_than_250': wall_coeffs[3]
        }
    elif wall_grouping == 'roughness-based':
        wall_coeffs = {
            'less_than_50': wall_coeffs[0],
            'between_50_and_55': wall_coeffs[1],
            'between_55_and_65': wall_coeffs[2],
            'between_65_and_80': wall_coeffs[3],
            'between_80_and_100': wall_coeffs[4],
            'between_100_and_120': wall_coeffs[5],
            'between_120_and_138': wall_coeffs[6],
            'greater_than_138': wall_coeffs[7],
        }
    elif wall_grouping == 'material-based':
        wall_coeffs = {
            'metallic': wall_coeffs[0],
            'plastic': wall_coeffs[1],
            'cement': wall_coeffs[2],
        }
    else:
        raise ValueError('Wall grouping type is not valid. Please choose from: single, diameter-based, roughness-based, or material-based.')

    # simulate water quality dynamics
    sim_results = model_simulation(flow_df, pressure_df, cl_df, sim_type='chlorine', demand_resolution=demand_resolution, iv_status='closed', dbv_status='active', trace_node=None, wall_grouping=wall_grouping, wall_coeffs=wall_coeffs_dict, bulk_coeff=bulk_coeff, flush_data=None)

    cl_sim = sim_results.chlorine
    sensor_data = sensor_model_id('wq')
    cl_sim = cl_sim[sensor_data['model_id'].unique()]
    name_mapping = sensor_data.set_index('model_id')['bwfl_id'].to_dict()
    cl_sim = cl_sim.rename(columns=name_mapping)

    # compute and return fitness value
    bwfl_ids = [sensor for sensor in sensor_data['bwfl_id'].unique() if sensor not in ['BW1', 'BW4']]
    datetime = cl_df['datetime'].unique()

    if obj_function == 'mse':
        mse = 0
        for name in bwfl_ids:
            sim = cl_sim[name].values
            data = cl_df.loc[cl_df['bwfl_id'] == name, 'mean'].values
            mask = ~np.isnan(sim) & ~np.isnan(data)
            mse += (1 / (len(datetime) * len(bwfl_ids))) * np.sum((sim[mask] - data[mask]) ** 2)
        return mse

    elif obj_function == 'rmse':
        rmse = 0
        for name in bwfl_ids:
            sim = cl_sim[name].values
            data = cl_df.loc[cl_df['bwfl_id'] == name, 'mean'].values
            mask = ~np.isnan(sim) & ~np.isnan(data)
            rmse += (1 / (len(datetime) * len(bwfl_ids))) * np.sum((sim[mask] - data[mask]) ** 2)
        rmse = np.sqrt(rmse)
        return rmse
    
    elif obj_function == 'mae':
        mae = 0
        for name in bwfl_ids:
            sim = cl_sim[name].values
            data = cl_df.loc[cl_df['bwfl_id'] == name, 'mean'].values
            mask = ~np.isnan(sim) & ~np.isnan(data)
            mae += (1 / (len(datetime) * len(bwfl_ids))) * np.sum(np.abs(sim[mask] - data[mask]))
        return mae
    
    elif obj_function == 'mape':
        mape = 0
        for name in bwfl_ids:
            sim = cl_sim[name].values
            data = cl_df.loc[cl_df['bwfl_id'] == name, 'mean'].values
            mask = ~np.isnan(sim) & ~np.isnan(data) & (data != 0)  # Ensure data is not zero to avoid division by zero
            mape += (1 / (len(datetime) * len(bwfl_ids))) * np.sum(np.abs((sim[mask] - data[mask]) / data[mask]))
        mape *= 100
        return mape
    
    else:
        raise ValueError('Objective function is not valid. Please choose from: mse, rmse, mae, or mape.')
