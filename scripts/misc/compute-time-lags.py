import pandas as pd
import numpy as np
from pathlib import Path
from bayesian_wq_calibration.epanet import model_simulation
from bayesian_wq_calibration.data import sensor_model_id
from bayesian_wq_calibration.constants import TIMESERIES_DIR

def compute_cross_correlation_lag(series_1, series_2):
    cross_corr = np.correlate(series_1 - np.mean(series_1), series_2 - np.mean(series_2), mode='full')
    lags = np.arange(-len(series_1) + 1, len(series_1))
    return lags[np.argmax(cross_corr)]

# reference mapping of sensors to expected DMA inlet
dma_inlets = {
    'BW12': 'BW1', 'BW3': 'BW1', 'BW9': 'BW4', 'BW2_1': 'BW4', 'BW5_1': 'BW4', 'BW6': 'BW1'
}

# load data
idx = 12
flow_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(idx).zfill(2)}-flow.csv")
pressure_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(idx).zfill(2)}-pressure.csv")
wq_df = pd.read_csv(TIMESERIES_DIR / f"processed/{str(idx).zfill(2)}-wq.csv", low_memory=False)
cl_df = wq_df[wq_df['data_type'] == 'chlorine'].copy()
cl_df['datetime'] = pd.to_datetime(cl_df['datetime'])

# simulate water age
sim_results = model_simulation(flow_df, pressure_df, cl_df, sim_type='age', demand_resolution='wwmd')
sensor_data = sensor_model_id('wq')
age = sim_results.age[sensor_data['model_id'].unique()]
name_mapping = sensor_data.set_index('model_id')['bwfl_id'].to_dict()
age = age.rename(columns=name_mapping)
age_after_1_day = age.iloc[4 * 24:]
mean_age_after_1_day = age_after_1_day.mean()

# compute lag values
summary_rows = []
for sensor, inlet in dma_inlets.items():
    series_sensor = cl_df[cl_df['bwfl_id'] == sensor]['mean'].dropna().values
    series_inlet = cl_df[cl_df['bwfl_id'] == inlet]['mean'].dropna().values

    cross_corr_lag = compute_cross_correlation_lag(series_inlet, series_sensor)
    water_age_lag = -round(mean_age_after_1_day[sensor] * 4)

    summary_rows.append({
        'Sensor': sensor,
        'DMA Inlet': inlet,
        'Cross-Corr Lag [steps]': cross_corr_lag,
        'Water Age Lag [steps]': water_age_lag,
        'Lag Diff [%]': round(100 * abs(cross_corr_lag - water_age_lag) / max(1, abs(water_age_lag)), 1)
    })
    print(f"Processed {sensor}: Cross-Corr Lag = {cross_corr_lag}, Water Age Lag = {water_age_lag}")

# # Display results
# summary_df = pd.DataFrame(summary_rows)
# print(summary_df.head())