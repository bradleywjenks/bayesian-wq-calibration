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
    bulk_coeff = -0.5 # from BW
    obj_function = 'mse' # 'mse', 'rmse', 'mae', 'mape'
    fitness_value =  fitness(wn, cl_df, wall_coeffs, obj_function, grouping, bulk_coeff)
    return (fitness_value,)


def fitness(wn, cl_df, wall_coeffs, obj_function, grouping, bulk_coeff):

    # translate decision variables to dict for model simulation
    wall_coeffs = decision_variables_to_dict(grouping, wall_coeffs)

    # update wq reaction coefficients
    wn = set_reaction_parameters(wn, grouping, wall_coeffs, bulk_coeff)

    # simulate water quality dynamics
    sim_type = 'chlorine'
    sim_results = epanet_simulator(wn, sim_type, cl_df)

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
            mask = ~np.isnan(sim) & ~np.isnan(data) & (np.arange(len(sim)) >= 4 * 24)
            mse += (1 / (len(datetime) * len(bwfl_ids))) * np.sum((sim[mask] - data[mask]) ** 2)
        obj_val = mse

    elif obj_function == 'rmse':
        rmse = 0
        for name in bwfl_ids:
            sim = cl_sim[name].values
            data = cl_df.loc[cl_df['bwfl_id'] == name, 'mean'].values
            mask = ~np.isnan(sim) & ~np.isnan(data) & (np.arange(len(sim)) >= 4 * 24)
            rmse += (1 / (len(datetime) * len(bwfl_ids))) * np.sum((sim[mask] - data[mask]) ** 2)
        rmse = np.sqrt(rmse)
        obj_val = rmse
    
    elif obj_function == 'mae':
        mae = 0
        for name in bwfl_ids:
            sim = cl_sim[name].values
            data = cl_df.loc[cl_df['bwfl_id'] == name, 'mean'].values
            mask = ~np.isnan(sim) & ~np.isnan(data) & (np.arange(len(sim)) >= 4 * 24)
            mae += (1 / (len(datetime) * len(bwfl_ids))) * np.sum(np.abs(sim[mask] - data[mask]))
        obj_val = mae
    
    elif obj_function == 'mape':
        mape = 0
        for name in bwfl_ids:
            sim = cl_sim[name].values
            data = cl_df.loc[cl_df['bwfl_id'] == name, 'mean'].values
            mask = ~np.isnan(sim) & ~np.isnan(data) & (np.arange(len(sim)) >= 4 * 24)
            mape += (1 / (len(datetime) * len(bwfl_ids))) * np.sum(np.abs((sim[mask] - data[mask]) / data[mask]))
        mape *= 100
        obj_val = mape
    
    else:
        raise ValueError('Objective function is not valid. Please choose from: mse, rmse, mae, or mape.')

    return obj_val



def decision_variables_to_dict(grouping, wall_coeffs):

    if grouping == 'single':
        wall_coeffs = {'single': wall_coeffs[0]}
    elif grouping == 'material':
        wall_coeffs = {
            'metallic': wall_coeffs[0],
            'cement': wall_coeffs[1],
            'plastic_unknown': wall_coeffs[2],
        }
    elif grouping == 'material-diameter':
        wall_coeffs = {
            'metallic_less_than_150': wall_coeffs[0],
            'metallic_greater_than_150': wall_coeffs[1],
            'cement_less_than_150': wall_coeffs[2],
            'cement_greater_than_150': wall_coeffs[3],
            'plastic_unknown_less_than_150': wall_coeffs[4],
            'plastic_unknown_greater_than_150': wall_coeffs[5],
        }
    elif grouping == 'roughness':
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
    else:
        raise ValueError('Wall grouping type is not valid. Please choose from: single, material, material-diameter, or roughness.')
    
    return wall_coeffs