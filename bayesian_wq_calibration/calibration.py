from bayesian_wq_calibration.epanet import epanet_simulator, set_reaction_parameters
from bayesian_wq_calibration.data import sensor_model_id
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
    elif grouping in ['material-age-diameter', 'material-age-velocity']:
        wall_coeffs = {
            'G0': params[0],
            'G1': params[1],
            'G2': params[2],
            'G3': params[3],
            'G4': params[4],
            'G5': params[5],
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

def generate_samples(param_mean, param_bounds, param_group, n_samples, sampling_method='lhs', dist_type='truncated normal', rel_uncertainty=0.5, plot=False):

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

    if plot:
        cols = 2
        rows = (n_params + cols - 1) // cols
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=param_group, horizontal_spacing=0.2)

    for i, sample in enumerate(samples.T):
        mu = param_mean[i]
        if mu == 0:
            mu = -1e-4
        lower_bound, upper_bound = param_bounds[i]

        if dist_type == 'truncated normal':
            sigma = abs(mu * rel_uncertainty)
            a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma # upper bound is 0
            param_samples = truncnorm.ppf(sample, a=a, b=b, loc=mu, scale=sigma)

            if plot:
                x = np.linspace(lower_bound, 0, 500)
                y = truncnorm.pdf(x, a=a, b=b, loc=mu, scale=sigma)

        elif dist_type == 'triangle':
            c = (mu - lower_bound) / (upper_bound - lower_bound) # mean is the mode
            param_samples = triang.ppf(sample, c=c, loc=lower_bound, scale=upper_bound - lower_bound)

            if plot:
                x = np.linspace(lower_bound, upper_bound, 500)
                y = triang.pdf(x, c=c, loc=lower_bound, scale=upper_bound - lower_bound)

        elif dist_type == 'uniform':
            param_samples = uniform.ppf(sample, loc=lower_bound, scale=upper_bound - lower_bound) # uniform between bounds

            if plot:
                x = np.linspace(lower_bound, upper_bound, 500)
                y = uniform.pdf(x, loc=lower_bound, scale=upper_bound - lower_bound)

        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

        if plot:
            row, col = divmod(i, cols)
            fig.add_trace(
                go.Scatter(x=x, y=y, mode='lines', line=dict(dash='solid', color='grey'), name=f'{param_group[i]} pdf'),
                row=row + 1, col=col + 1
            )
            fig.add_trace(
                go.Scatter(x=param_samples, y=[0] * len(param_samples), mode='markers',
                           marker=dict(size=6, color=default_colors[0], opacity=0.8), name=f'{param_group[i]} samples'),
                row=row + 1, col=col + 1
            )
            fig.update_xaxes(title_text="decay coefficient [m/d]", row=row + 1, col=col + 1)
            fig.update_yaxes(title_text="density", row=row + 1, col=col + 1)

        scaled_samples.append(param_samples)

    if plot:
        fig.update_layout(
            template="simple_white",
            width=650,
            height=300 * rows,
            showlegend=False
        )
        fig.show()

    return np.array(scaled_samples).T





"""
Markov Chain Monte Carlo sampling functions.
"""
# insert functions here...