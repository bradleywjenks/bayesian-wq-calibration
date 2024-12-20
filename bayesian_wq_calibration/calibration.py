from bayesian_wq_calibration.epanet import epanet_simulator, set_reaction_parameters
from bayesian_wq_calibration.data import sensor_model_id
import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms



"""
Misc. functions
"""
def decision_variables_to_dict(grouping, params):

    if grouping == 'single':
        wall_coeffs = {'G0': params[0]}
    elif grouping == 'material-only':
        wall_coeffs = {
            'G0': params[0],
            'G1': params[1],
        }
    elif grouping in ['material-age-diameter', 'material-age-mrt']:
        wall_coeffs = {
            'G0': params[0],
            'G1': params[1],
            'G2': params[2],
            'G3': params[3],
            'G4': params[4],
        }
    else:
        raise ValueError('Wall grouping type is not valid.')
    
    return wall_coeffs






"""
Genetic algorithm functions
"""
def evaluate(individual, wn, cl_df, grouping):
    params = individual
    wall_coeffs = decision_variables_to_dict(grouping, params)
    obj_function = 'mse' # 'mse', 'rmse', 'mae', 'mape'
    fitness_value =  fitness(wn, cl_df, grouping, wall_coeffs, obj_function)
    return (fitness_value,)


def fitness(wn, cl_df, grouping, wall_coeffs, obj_function):

    # update wq reaction coefficients
    bulk_coeff = wn.options.reaction.bulk_coeff * 3600 * 24
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
    cl_df_bwfl_ids = cl_df['bwfl_id'].unique()
    bwfl_ids = [sensor for sensor in sensor_data['bwfl_id'].unique() if sensor in cl_df_bwfl_ids and sensor not in ['BW1', 'BW4']]
    datetime = cl_df['datetime'].unique()[96:]

    if obj_function == 'mse':
        mse = 0
        for name in bwfl_ids:   
            sim = cl_sim[name].values
            data = cl_df.loc[cl_df['bwfl_id'] == name, 'mean'].values
            mask = ~np.isnan(sim) & ~np.isnan(data) & (np.arange(len(sim)) >= 96)
            mse += (1 / (len(datetime) * len(bwfl_ids))) * np.sum((sim[mask] - data[mask]) ** 2)
        obj_val = mse

    elif obj_function == 'rmse':
        rmse = 0
        for name in bwfl_ids:
            sim = cl_sim[name].values
            data = cl_df.loc[cl_df['bwfl_id'] == name, 'mean'].values
            mask = ~np.isnan(sim) & ~np.isnan(data) & (np.arange(len(sim)) >= 96)
            rmse += (1 / (len(datetime) * len(bwfl_ids))) * np.sum((sim[mask] - data[mask]) ** 2)
        rmse = np.sqrt(rmse)
        obj_val = rmse
    
    elif obj_function == 'mae':
        mae = 0
        for name in bwfl_ids:
            sim = cl_sim[name].values
            data = cl_df.loc[cl_df['bwfl_id'] == name, 'mean'].values
            mask = ~np.isnan(sim) & ~np.isnan(data) & (np.arange(len(sim)) >= 96)
            mae += (1 / (len(datetime) * len(bwfl_ids))) * np.sum(np.abs(sim[mask] - data[mask]))
        obj_val = mae
    
    elif obj_function == 'mape':
        mape = 0
        for name in bwfl_ids:
            sim = cl_sim[name].values
            data = cl_df.loc[cl_df['bwfl_id'] == name, 'mean'].values
            mask = ~np.isnan(sim) & ~np.isnan(data) & (np.arange(len(sim)) >= 96)
            mape += (1 / (len(datetime) * len(bwfl_ids))) * np.sum(np.abs((sim[mask] - data[mask]) / data[mask]))
        mape *= 100
        obj_val = mape
    
    else:
        raise ValueError('Objective function is not valid. Please choose from: mse, rmse, mae, or mape.')

    return obj_val





"""
Statistical emulator/surrogate modelling functions.
"""
# insert functions here...





"""
Markov Chain Monte Carlo sampling functions.
"""
# insert functions here...