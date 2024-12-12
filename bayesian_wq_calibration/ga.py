from bayesian_wq_calibration.epanet import epanet_simulator, set_reaction_parameters
from bayesian_wq_calibration.data import sensor_model_id
from bayesian_wq_calibration.mcmc import decision_variables_to_dict
import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms


"""
Genetic algorithm fitness function
"""
def evaluate(individual, wn, cl_df, grouping, mean_vel):
    wall_coeffs = individual
    obj_function = 'mse' # 'mse', 'rmse', 'mae', 'mape'
    fitness_value =  fitness(wn, cl_df, wall_coeffs, obj_function, grouping, mean_vel)
    return (fitness_value,)


def fitness(wn, cl_df, wall_coeffs, obj_function, grouping, mean_vel):

    # translate decision variables to dict for model simulation
    wall_coeffs = decision_variables_to_dict(grouping, wall_coeffs)

    # update wq reaction coefficients
    bulk_coeff = wn.options.reaction.bulk_coeff * 3600 * 24
    wn = set_reaction_parameters(wn, grouping, wall_coeffs, bulk_coeff, mean_vel)

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
