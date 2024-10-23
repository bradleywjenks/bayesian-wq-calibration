"""
This script checks the quality of chlorine time series data using the expected directional conditions based on network hydraulics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from bayesian_wq_calibration.simulation import model_simulation, sensor_model_id
from bayesian_wq_calibration.constants import TIMESERIES_DIR
import plotly.express as px
from dtaidistance import dtw
import warnings
warnings.filterwarnings("ignore")


# pass script arguments
parser = argparse.ArgumentParser(description='Check quality of processed time series data for model calibration.')
parser.add_argument('--tol', type=float, default=0.05, help='Error tolerance for comparing downstream sensor data (default: 0.05 mg/L).')
parser.add_argument('--lag_method', type=str, default='x-corr', help='Method for computing time lag between sensor time series (default: cross-correlation).')
args = parser.parse_args()

tol = args.tol # [mg/L] tolerance for directional conditions
lag_method = args.lag_method
quality_threshold = 0.95 # percent of data points that must satisfy directional conditions; not 100% due to errors computing lag between time series

if lag_method == 'dtw':
    print('Note! Dynamic time warping method generally performs well, but some time series shifting is erroneous. To be investigated at a later date.')


# define functions
def shift_time_series_cross_correlation(cl_df, series_1, series_2, bwfl_id, mean_age_after_1_day, idx):
    """Shift time series using cross-correlation lag."""
    cross_corr = np.correlate(series_1 - np.mean(series_1), series_2 - np.mean(series_2), mode='full')
    lags = np.arange(-len(series_1) + 1, len(series_1))
    best_lag = lags[np.argmax(cross_corr)]

    if best_lag < 0 and best_lag > -(24 * 4):
        cl_df.loc[cl_df['bwfl_id'] == bwfl_id, 'datetime'] += pd.Timedelta(minutes=15 * best_lag)
        print(f"Data period {idx}. Shifted {bwfl_id} by {best_lag} time steps (cross-correlation).")
    else:
        best_lag = -int(round(mean_age_after_1_day[bwfl_id] * 4))
        cl_df.loc[cl_df['bwfl_id'] == bwfl_id, 'datetime'] += pd.Timedelta(minutes=15 * best_lag)
        print(f"Data period {idx}. Shifted {bwfl_id} by {best_lag} time steps (cross-correlation not valid; apply simulated age).")



def shift_time_series_dtw(cl_df, series_1, series_2, bwfl_id, mean_age_after_1_day, idx):
    """Shifts the time series for a sensor using Dynamic Time Warping (DTW)."""
    if bwfl_id == 'BW12':
        w_hr = 2
    elif bwfl_id == 'BW3':
        w_hr = 8
    elif bwfl_id == 'BW5':
        w_hr = 6
    elif bwfl_id == 'BW6':
        w_hr = 14
    elif bwfl_id == 'BW7':
        w_hr = 14
    elif bwfl_id == 'BW2':
        w_hr = 3    
    elif bwfl_id == 'BW9':
        w_hr = 2
    try:
        distance, paths = dtw.warping_paths(series_2, series_1, window=w_hr*4)
        optimal_path = dtw.best_path(paths)
    except:
        print(f"Data period {idx}. Shifted {bwfl_id} by simulated water age (DTW failed due to missing data).")
        return
    
    lag_values = []
    for p in optimal_path:
        idx1 = p[0]
        idx2 = p[1]
        lag = idx2 - idx1  # calculate lag based on DTW path
        
        if lag > 0 or lag < -(24 * 4):
            age_lag = -int(round(mean_age_after_1_day[bwfl_id] * 4))
            lag_values.append((idx1, age_lag))
        else:
            lag_values.append((idx1, lag))

    if lag_values:
        for idx2, lag in lag_values:
            cl_df.loc[(cl_df['bwfl_id'] == bwfl_id) & (cl_df['datetime'] == cl_df['datetime'].unique()[idx2]), 'datetime'] += pd.Timedelta(minutes=15 * lag)

        lags = [lag for _, lag in lag_values]
        min_lag = max(lags)
        max_lag = min(lags)
        print(f"Data period {idx}. Shifted {bwfl_id} with DTW (lag range: {min_lag} to {max_lag} time steps).")



### Check data satisfies expected directional conditions based on network hydraulics ###
N = 19 # 19 data periods
for idx in range(1, N+1):

    print(f"Checking time series for data period {idx}...")

    # load data
    flow_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(idx).zfill(2)}-flow.csv")
    pressure_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(idx).zfill(2)}-pressure.csv")
    wq_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(idx).zfill(2)}-wq.csv", low_memory=False)

    cl_df = wq_df[wq_df['data_type'] == 'chlorine'].copy()
    cl_df['datetime'] = pd.to_datetime(cl_df['datetime'])

    # simulate water age
    sim_results = model_simulation(flow_df, pressure_df, wq_df[wq_df['data_type'] == 'chlorine'], sim_type='age', demand_resolution='wwmd')
    sensor_data = sensor_model_id('wq')
    age = sim_results.age[sensor_data['model_id'].unique()]
    name_mapping = sensor_data.set_index('model_id')['bwfl_id'].to_dict()
    age = age.rename(columns=name_mapping)
    age_after_1_day = age.iloc[4*24:]
    mean_age_after_1_day = age_after_1_day.mean()


    ### 1. shift time series data based on time series lag; use simulated water age if lag value not available ###

    sensors = ['BW12', 'BW3', 'BW7', 'BW9', 'BW2', 'BW5', 'BW6']
    ref_sensors = {
        'BW12': 'BW1', 'BW3': 'BW1', 'BW7': 'BW1',
        'BW9': 'BW4', 'BW2': 'BW4', 'BW5': 'BW1', 'BW6': 'BW1'
    }

    for bwfl_id in sensors:
        ref_sensor_id = ref_sensors[bwfl_id]

        series_1 = cl_df[cl_df['bwfl_id'] == ref_sensor_id]['mean'].values
        series_2 = cl_df[cl_df['bwfl_id'] == bwfl_id]['mean'].values

        if lag_method == 'x-corr':
            shift_time_series_cross_correlation(cl_df, series_1, series_2, bwfl_id, mean_age_after_1_day, idx)
        elif lag_method == 'dtw':
            shift_time_series_dtw(cl_df, series_1, series_2, bwfl_id, mean_age_after_1_day, idx)

    
    # truncate datetime range to match flow data
    cl_df = cl_df[cl_df['datetime'] >= flow_df['datetime'].unique()[0]]
    min_end_datetime = cl_df.groupby('bwfl_id')['datetime'].max().min()
    cl_df = cl_df[cl_df['datetime'] <= min_end_datetime]
    cl_df = cl_df.drop_duplicates(subset=['datetime', 'bwfl_id'], keep='first')

    # plot shifted time series
    cl_df['series'] = 'shifted'
    original_df = wq_df[wq_df['data_type'] == 'chlorine'].copy()
    original_df['series'] = 'original'
    combined_df = pd.concat([original_df, cl_df])
    fig = px.line(
        combined_df,
        x='datetime',
        y='mean',
        color='bwfl_id',
        line_dash='series'
    )
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Chlorine [mg/L]',
        legend_title_text='',
        template='simple_white',
        height=450,
    )
    fig.show()



    ### 2. check the that the following conditions are met within the specified tolerance ###

    # (a) BW1 > BW12
    bw1_values, bw12_values = cl_df.pivot(index='datetime', columns='bwfl_id', values='mean')[['BW1', 'BW12']].dropna().values.T
    if len(bw12_values) == 0:
        print(f"Data period {idx}. No values found in BW12 or BW1.")
        bw1_bw12_check = None
    else:
        bw1_bw12_check = (bw1_values + tol > bw12_values).mean() >= quality_threshold
        print(f"Data period {idx}. BW1 (+ {tol} mg/L tolerance) > BW12: {bw1_bw12_check}")

    # (b) BW12 > BW3
    bw12_values, bw3_values = cl_df.pivot(index='datetime', columns='bwfl_id', values='mean')[['BW12', 'BW3']].dropna().values.T
    if len(bw3_values) == 0:
        print(f"Data period {idx}. No values found in BW3 or BW12.")
        bw12_bw3_check = None
    else:
        bw12_bw3_check = (bw12_values + tol > bw3_values).mean() >= quality_threshold
        print(f"Data period {idx}. BW12 (+ {tol} mg/L tolerance) > BW3: {bw12_bw3_check}")

    # (c) BW12 > BW7
    bw12_values, bw7_values = cl_df.pivot(index='datetime', columns='bwfl_id', values='mean')[['BW12', 'BW7']].dropna().values.T
    if len(bw7_values) == 0:
        print(f"Data period {idx}. No values found in BW7 or BW12.")
        bw12_bw7_check = None
    else:
        bw12_bw7_check = (bw12_values + tol > bw7_values).mean() >= quality_threshold
        print(f"Data period {idx}. BW12 (+ {tol} mg/L tolerance) > BW7: {bw12_bw7_check}")

    # (d) BW4 > BW9
    bw4_values, bw9_values = cl_df.pivot(index='datetime', columns='bwfl_id', values='mean')[['BW4', 'BW9']].dropna().values.T
    if len(bw9_values) == 0:
        print(f"Data period {idx}. No values found in BW9 or BW4.")
        bw4_bw9_check = None
    else:
        bw4_bw9_check = (bw4_values + tol > bw9_values).mean() >= quality_threshold
        print(f"Data period {idx}. BW4 (+ {tol} mg/L tolerance) > BW9: {bw4_bw9_check}")

    # (e) BW4 > BW2
    bw4_values, bw2_values = cl_df.pivot(index='datetime', columns='bwfl_id', values='mean')[['BW4', 'BW2']].dropna().values.T
    if len(bw2_values) == 0:
        print(f"Data period {idx}. No values found in BW2 or BW4.")
        bw4_bw2_check = None
    else:
        bw4_bw2_check = (bw4_values + tol > bw2_values).mean() >= quality_threshold
        print(f"Data period {idx}. BW4 (+ {tol} mg/L tolerance) > BW2: {bw4_bw2_check}")

    # (f) (BW12 or BW2) > BW6
    bw12_values, bw2_values, bw6_values = cl_df.pivot(index='datetime', columns='bwfl_id', values='mean')[['BW12', 'BW2', 'BW6']].dropna().values.T
    if len(bw6_values) == 0 or len(bw2_values) == 0 or len(bw12_values) == 0:
        print(f"Data period {idx}. No values found in BW6 or BW12 or BW2.")
        bw12_bw2_bw6_check = None
    else:
        max_bw12_bw2_values = np.maximum(bw12_values, bw2_values)
        bw12_bw2_bw6_check = (max_bw12_bw2_values + tol > bw6_values).mean() >= quality_threshold
        print(f"Data period {idx}. BW12 or BW2 (+ {tol} mg/L tolerance) > BW6: {bw12_bw2_bw6_check}")

    # (g) (BW12 or BW9) > BW5
    bw12_values, bw9_values, bw5_values = cl_df.pivot(index='datetime', columns='bwfl_id', values='mean')[['BW12', 'BW9', 'BW5']].dropna().values.T
    if len(bw5_values) == 0 or len(bw9_values) ==0 or len(bw12_values) == 0:
        print(f"Data period {idx}. No values found in BW5 or BW12 or BW9.")
        bw12_bw9_bw5_check = None
    else:
        max_bw12_bw9_values = np.maximum(bw12_values, bw9_values)
        bw12_bw9_bw5_check = (max_bw12_bw9_values + tol > bw5_values).mean() >= quality_threshold
        print(f"Data period {idx}. BW12 or BW9 (+ {tol} mg/L tolerance) > BW5: {bw12_bw9_bw5_check}")