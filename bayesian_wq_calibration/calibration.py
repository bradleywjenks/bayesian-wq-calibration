from bayesian_wq_calibration.epanet import epanet_simulator, set_reaction_parameters
from bayesian_wq_calibration.data import sensor_model_id
import networkx as nx
import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.colors
default_colors = plotly.colors.qualitative.Plotly
from scipy.stats.qmc import Sobol, Halton
from scipy.stats import norm, truncnorm, triang, uniform
import random
from pyDOE import lhs
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C



"""
Misc. functions
"""

def get_observable_paths(flow_df, link_df, wq_sensors_used='kiosk + hydrant'):
    
    """Finds links on paths between sensors based on flow direction."""
    
    # get sensor nodes based on type
    sensor_data = sensor_model_id('wq')
    sensor_nodes = sensor_data['model_id'].values
    sensor_hydrant = [2, 5, 6]
    sensor_kiosk = [i for i in range(len(sensor_nodes)) if i not in sensor_hydrant]
    sensor_remove = [6]
    sensor_all = [i for i in range(len(sensor_nodes)) if i not in sensor_remove]
    
    if wq_sensors_used == 'kiosk only':
        sensor_indices = sensor_kiosk
    elif wq_sensors_used == 'hydrant only':
        sensor_indices = sensor_hydrant
    else:  # 'kiosk + hydrant'
        sensor_indices = list(range(len(sensor_nodes)))
        sensor_indices = sensor_all
        
    selected_sensor_nodes = sensor_nodes[sensor_indices]
    sensor_pairs = list(combinations(selected_sensor_nodes, 2))
    
    # precompute link mapping
    link_mapping = {(row['node_out'], row['node_in']): idx for idx, row in link_df.iterrows()}
    link_mapping.update({(row['node_in'], row['node_out']): idx for idx, row in link_df.iterrows()})
    observable = np.zeros(len(link_df))
    
    def process_timestep(t):
        # create directed graph based on flow direction
        flow_dir = np.sign(flow_df.iloc[:, t])
        edges = [
            (row['node_out'], row['node_in']) if flow_dir[row['link_ID']] >= 0 
            else (row['node_in'], row['node_out'])
            for _, row in link_df.iterrows()
        ]
        G = nx.DiGraph(edges)
        timestep_observable = np.zeros(len(link_df))
        
        # find paths for each sensor pair
        for s1, s2 in sensor_pairs:
            if nx.has_path(G, s1, s2):
                for path in nx.all_simple_paths(G, s1, s2):
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i + 1])
                        if edge in link_mapping:
                            timestep_observable[link_mapping[edge]] = 1
        return timestep_observable
    
    # parallelize timesteps
    results = Parallel(n_jobs=-1)(delayed(process_timestep)(t) for t in range(flow_df.shape[1]))
    observable = np.sum(results, axis=0) > 0
    
    return observable






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

def generate_samples(param_mean, param_bounds, param_group, n_samples, sampling_method='lhs', dist_type='truncated normal', bulk_uncertainty=0.1, wall_uncertainty=0.5):

    n_params = len(param_mean)

    if sampling_method == 'lhs':
        samples = lhs(n_params, samples=n_samples)
    elif sampling_method == 'monte carlo':
        samples = np.random.rand(n_samples, n_params)
    elif sampling_method == 'sobol':
        sampler = Sobol(d=n_params, scramble=True)
        samples = sampler.random_base2(m=int(np.log2(n_samples)))
    elif sampling_method == 'halton':
        sampler = Halton(d=n_params, scramble=True)
        samples = sampler.random(n_samples)
    else:
        raise ValueError(f"Unsupported sampling method: {sampling_method}")
    
    scaled_samples = []

    for i, sample in enumerate(samples.T):

        mu = param_mean[i]
        
        if param_group[i] == 'B':
            sigma = abs(mu * bulk_uncertainty)
            param_samples = norm.ppf(sample, loc=mu, scale=sigma)

        else:
            sigma = abs(mu * wall_uncertainty)
            lower_bound, upper_bound = param_bounds[i]

            if dist_type == 'truncated normal':
                a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma # upper bound is 0
                param_samples = truncnorm.ppf(sample, a=a, b=b, loc=mu, scale=sigma)

            elif dist_type == 'triangle':
                c = (mu - lower_bound) / (upper_bound - lower_bound) # mean is the mode
                param_samples = triang.ppf(sample, c=c, loc=lower_bound, scale=upper_bound - lower_bound)

            elif dist_type == 'uniform':
                param_samples = uniform.ppf(sample, loc=lower_bound, scale=upper_bound - lower_bound) # uniform between bounds

            else:
                raise ValueError(f"Unsupported distribution type: {dist_type}")

        scaled_samples.append(param_samples)

    return np.array(scaled_samples).T



def setup_gp_model(n_features, kernel_type='RBF', nu=1.5, n_restarts=50, normalize_y=True):
    """Set up GP model with specified kernel."""
    if kernel_type == 'RBF':
        kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0] * n_features, length_scale_bounds=(1e-3, 1e3))
    elif kernel_type == 'Matern':
        kernel = C(1.0, (1e-3, 1e3)) * Matern([1.0] * n_features, nu=nu)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
        
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=normalize_y,
        random_state=42
    )






"""
Markov Chain Monte Carlo sampling functions.
"""
# insert functions here...