from bayesian_wq_calibration.simulation import epanet_simulator, sensor_model_id, set_reaction_parameters
import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms


"""
Genetic algorithm fitness function
"""
def evaluate(individual, wn, cl_df, grouping):
    wall_coeffs = individual
    bulk_coeff = -0.4
    obj_function = 'mse' # 'mse', 'rmse', 'mae', 'mape'
    fitness_value =  fitness(wn, cl_df, wall_coeffs, obj_function=obj_function, grouping=grouping, bulk_coeff=bulk_coeff)
    return (fitness_value,)


def fitness(wn, cl_df, wall_coeffs, obj_function='mse', grouping='single', bulk_coeff=-0.4):

    # translate decision variables to dict for model simulation
    if grouping == 'single':
        wall_coeffs = {'single': wall_coeffs[0]}
    elif grouping == 'diameter-based':
        wall_coeffs = {
            'less_than_75': wall_coeffs[0],
            'between_75_and_150': wall_coeffs[1],
            'between_150_and_250': wall_coeffs[2],
            'greater_than_250': wall_coeffs[3]
        }
    elif grouping == 'roughness-based':
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
    elif grouping == 'material-based':
        wall_coeffs = {
            'metallic': wall_coeffs[0],
            'plastic': wall_coeffs[1],
            'cement': wall_coeffs[2],
        }
    else:
        raise ValueError('Wall grouping type is not valid. Please choose from: single, diameter-based, roughness-based, or material-based.')

    # update wq reaction coefficients
    wn = set_reaction_parameters(wn, grouping, wall_coeffs, bulk_coeff)

    # simulate water quality dynamics
    datetime = cl_df['datetime'].unique()
    sim_type = 'chlorine'
    sim_results = epanet_simulator(wn, sim_type, datetime)

    # organize chlorine simulation results
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
        obj_val = mse

    elif obj_function == 'rmse':
        rmse = 0
        for name in bwfl_ids:
            sim = cl_sim[name].values
            data = cl_df.loc[cl_df['bwfl_id'] == name, 'mean'].values
            mask = ~np.isnan(sim) & ~np.isnan(data)
            rmse += (1 / (len(datetime) * len(bwfl_ids))) * np.sum((sim[mask] - data[mask]) ** 2)
        rmse = np.sqrt(rmse)
        obj_val = rmse
    
    elif obj_function == 'mae':
        mae = 0
        for name in bwfl_ids:
            sim = cl_sim[name].values
            data = cl_df.loc[cl_df['bwfl_id'] == name, 'mean'].values
            mask = ~np.isnan(sim) & ~np.isnan(data)
            mae += (1 / (len(datetime) * len(bwfl_ids))) * np.sum(np.abs(sim[mask] - data[mask]))
        obj_val = mae
    
    elif obj_function == 'mape':
        mape = 0
        for name in bwfl_ids:
            sim = cl_sim[name].values
            data = cl_df.loc[cl_df['bwfl_id'] == name, 'mean'].values
            mask = ~np.isnan(sim) & ~np.isnan(data) & (data != 0)  # Ensure data is not zero to avoid division by zero
            mape += (1 / (len(datetime) * len(bwfl_ids))) * np.sum(np.abs((sim[mask] - data[mask]) / data[mask]))
        mape *= 100
        obj_val = mape
    
    else:
        raise ValueError('Objective function is not valid. Please choose from: mse, rmse, mae, or mape.')

    return obj_val