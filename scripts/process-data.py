"""
This script outputs flow, pressure, and wq time series data for a period of n_days after each metrinet sensor calibration event.
"""

import pandas as pd
import numpy as np
import zipfile as zip
from datetime import datetime, timedelta
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
parser = argparse.ArgumentParser(description='Process raw time series files to create data periods for model calibration.')
parser.add_argument('--n_days', type=int, default=7, help='Number of good days for data filtering (default: 7 days).')
args = parser.parse_args()


# get good data periods
calibration_data = pd.read_excel(TIMESERIES_DIR / 'raw/metrinet-calibration-records.xlsx')
if 'good_datetime' not in calibration_data.columns:
   raise ValueError("The specified file must contain a 'datetime' column.")
calibration_dates = pd.Series(calibration_data['good_datetime'].dt.to_period('D').dt.to_timestamp().unique()).sort_values()

good_data = {}
for idx, cal_date in calibration_dates.items():
   start_date = cal_date
   end_date = start_date + pd.Timedelta(days=args.n_days-1, hours=23, minutes=45, seconds=0)

   good_data[idx+1] = { 
       'start_date': start_date,
       'end_date': end_date
   }


keys_to_delete = []
for idx, cal_date in calibration_dates.items():
   for j in good_data.keys():
       if good_data[j]['start_date'] < cal_date < good_data[j]['end_date']:
           if j not in keys_to_delete:
               print(f'Note! Good data period {j} violated by calibration date {cal_date}. Good data period removed.')
               keys_to_delete.append(j)

for key in keys_to_delete:
   del good_data[key]

good_data = {new_idx+1: value for new_idx, (old_idx, value) in enumerate(good_data.items())}


# read time series data
pressure_df = pd.DataFrame()
flow_df = pd.DataFrame()
wq_df = pd.DataFrame()

with zip.ZipFile(TIMESERIES_DIR / 'imputed/field_lab-data-2021-2024.zip', 'r') as z:
   for filename in z.namelist():
      if filename.endswith('.csv'):
         with z.open(filename) as f:
            data = pd.read_csv(f, low_memory=False)
            data_type = filename.split('/')[-1].split('.')[0].split('_')[0]

            if data_type == 'pressure':
               pressure_df = pd.concat([pressure_df, data])
            elif data_type == 'flow':
               flow_df = pd.concat([flow_df, data])
            else:
               wq_df = pd.concat([wq_df, data])

pressure_df['datetime'] = pd.to_datetime(pressure_df['datetime'])
flow_df['datetime'] = pd.to_datetime(flow_df['datetime'])
wq_df['datetime'] = pd.to_datetime(wq_df['datetime'])




# process time series data
output_dir = TIMESERIES_DIR / 'processed/'
value_columns = ['min', 'mean', 'max']

for idx, period in good_data.items():

   filtered_pressure_df = pressure_df[(pressure_df['datetime'] >= period['start_date']) & (pressure_df['datetime'] <= period['end_date'])]
   filtered_flow_df = flow_df[(flow_df['datetime'] >= period['start_date']) & (flow_df['datetime'] <= period['end_date'])]
   filtered_wq_df = wq_df[(wq_df['datetime'] >= period['start_date']) & (wq_df['datetime'] <= period['end_date'])]



   ### Manual data filtering ###
   # 1. remove one time series at dupicated sensor locations based on comparison between upstream/downstream sensor (i.e. BW1/BW12 and BW4/BW9)
   # 2. remove erroneous data (e.g. zero data, downstream sensors with noticeably higher residuals than at the source)
   if idx == 1:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
   elif idx == 2:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
   elif idx == 3:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW6') & (filtered_wq_df['data_type'] == 'chlorine')].index, value_columns] = np.nan
   elif idx == 4:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, value_columns] = np.nan
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
   elif idx == 5:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
   elif idx == 6:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW2') & (filtered_wq_df['data_type'] == 'chlorine') & (filtered_wq_df['datetime'] <= pd.to_datetime('2022-12-22 16:00:00'))].index, value_columns] = np.nan
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW2') & (filtered_wq_df['data_type'] == 'chlorine') & (filtered_wq_df['datetime'] >= pd.to_datetime('2022-12-25 15:15:00'))].index, value_columns] = np.nan
   elif idx == 7:
      filtered_pressure_df = pressure_df[(pressure_df['datetime'] >= period['start_date']) & (pressure_df['datetime'] <= pd.to_datetime('2023-01-29 06:00:00'))]
      filtered_flow_df = flow_df[(flow_df['datetime'] >= period['start_date']) & (flow_df['datetime'] <= pd.to_datetime('2023-01-29 06:00:00'))]
      filtered_wq_df = wq_df[(wq_df['datetime'] >= period['start_date']) & (wq_df['datetime'] <= pd.to_datetime('2023-01-29 06:00:00'))]
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW2') & (filtered_wq_df['data_type'] == 'chlorine')].index, value_columns] = np.nan
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW5') & (filtered_wq_df['data_type'] == 'chlorine')].index, value_columns] = np.nan
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
   elif idx == 8:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
   elif idx == 9:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
   elif idx == 10:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
   elif idx == 11:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW3') & (filtered_wq_df['data_type'] == 'chlorine') & (filtered_wq_df['datetime'] <= pd.to_datetime('2023-08-09 18:30:00'))].index, value_columns] = np.nan
   elif idx == 12:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
   elif idx == 13:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW2') & (filtered_wq_df['data_type'] == 'chlorine') & (filtered_wq_df['datetime'] >= pd.to_datetime('2024-04-04 11:00:00'))].index, value_columns] = np.nan
   elif idx == 14:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
   elif idx == 15:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW3') & (filtered_wq_df['data_type'] == 'chlorine')].index, value_columns] = np.nan
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW7') & (filtered_wq_df['datetime'] >= pd.to_datetime('2024-05-31 08:30:00')) & (filtered_wq_df['datetime'] <= pd.to_datetime('2024-06-03 09:45:00'))].index, value_columns] = np.nan
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW6') & (filtered_wq_df['datetime'] <= pd.to_datetime('2024-06-03 23:15:00'))].index, value_columns] = np.nan
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW5') & (filtered_wq_df['datetime'] >= pd.to_datetime('2024-06-03 08:45:00'))].index, value_columns] = np.nan
   elif idx == 16:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW5') & (filtered_wq_df['data_type'] == 'chlorine')].index, value_columns] = np.nan
   elif idx == 17:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
   elif idx == 18:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_2') & (filtered_wq_df['data_type'] == 'chlorine') & (filtered_wq_df['datetime'] >= pd.to_datetime('2024-08-23 12:15:00')) & (filtered_wq_df['datetime'] <= pd.to_datetime('2024-08-23 13:00:00'))].index, value_columns] = np.nan
   elif idx == 19:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW7') & (filtered_wq_df['datetime'] >= pd.to_datetime('2024-09-26 17:30:00'))].index, value_columns] = np.nan
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW5') & (filtered_wq_df['data_type'] == 'chlorine') & (filtered_wq_df['datetime'] >= pd.to_datetime('2024-09-25 11:45:00')) & (filtered_wq_df['datetime'] <= pd.to_datetime('2024-09-25 13:00:00'))].index, value_columns] = np.nan

   filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'].isin(['BW1_2', 'BW12_2', 'BW4_2', 'BW9_2'])) & (filtered_wq_df['data_type'] != 'chlorine')].index,inplace=True)
   filtered_wq_df['bwfl_id'] = filtered_wq_df['bwfl_id'].str.split('_').str[0]




   ### Check data satisfies expected directional conditions based on network hydraulics ###
   shift_method = 'lag' # 'age' or 'lag'
   tol = 0.05 # [mg/L] tolerance for directional conditions
   quality_threshold = 0.95 # percent of data points that must satisfy directional conditions; not 100% due to errors computing lag between time series
   cl_df = filtered_wq_df[filtered_wq_df['data_type'] == 'chlorine'].copy()

   # simulate water age
   sim_results = model_simulation(filtered_flow_df, filtered_pressure_df, filtered_wq_df[filtered_wq_df['data_type'] == 'chlorine'], sim_type='age', demand_resolution='wwmd')
   sensor_data = sensor_model_id('wq')
   age = sim_results.age[sensor_data['model_id'].unique()]
   name_mapping = sensor_data.set_index('model_id')['bwfl_id'].to_dict()
   age = age.rename(columns=name_mapping)
   age_after_1_day = age.iloc[4*24:]
   mean_age_after_1_day = age_after_1_day.mean()
   
   # 1. shift time series based on age or lag
   if shift_method == 'age':
      
      for bwfl_id in cl_df['bwfl_id'].unique():
         if bwfl_id in mean_age_after_1_day:
            time_steps_to_shift = int(round(mean_age_after_1_day[bwfl_id] * 4)) # convert to number of 15-minute time steps
            cl_df.loc[cl_df['bwfl_id'] == bwfl_id, 'datetime'] -= pd.Timedelta(minutes=15 * time_steps_to_shift)

   elif shift_method == 'lag':

      # BW12
      series_1 = cl_df[cl_df['bwfl_id'] == 'BW1']['mean'].values
      series_2 = cl_df[cl_df['bwfl_id'] == 'BW12']['mean'].values
      cross_corr = np.correlate(series_1 - np.mean(series_1), series_2 - np.mean(series_2), mode='full')
      lags = np.arange(-len(series_1) + 1, len(series_1))
      best_lag = lags[np.argmax(cross_corr)]
      if best_lag < 0 and best_lag > -(24*4):
         cl_df.loc[cl_df['bwfl_id'] == 'BW12', 'datetime'] += pd.Timedelta(minutes=15 * best_lag)
         # print(f"Data period {idx}. Shifted BW12 by {best_lag} time steps.")
      else:
         best_lag = -int(round(mean_age_after_1_day['BW12'] * 4))
         cl_df.loc[cl_df['bwfl_id'] == 'BW12', 'datetime'] += pd.Timedelta(minutes=15 * best_lag)
         # print(f"Data period {idx}. Shifted BW12 by {best_lag} time steps (based on simulation).")

      # BW3
      series_1 = cl_df[cl_df['bwfl_id'] == 'BW1']['mean'].values
      series_2 = cl_df[cl_df['bwfl_id'] == 'BW3']['mean'].values
      cross_corr = np.correlate(series_1 - np.mean(series_1), series_2 - np.mean(series_2), mode='full')
      lags = np.arange(-len(series_1) + 1, len(series_1))
      best_lag = lags[np.argmax(cross_corr)]
      if best_lag < 0 and best_lag > -(24*4):
         cl_df.loc[cl_df['bwfl_id'] == 'BW3', 'datetime'] += pd.Timedelta(minutes=15 * best_lag)
         # print(f"Data period {idx}. Shifted BW3 by {best_lag} time steps.")
      else:
         best_lag = -int(round(mean_age_after_1_day['BW3'] * 4))
         cl_df.loc[cl_df['bwfl_id'] == 'BW3', 'datetime'] += pd.Timedelta(minutes=15 * best_lag)
         # print(f"Data period {idx}. Shifted BW3 by {best_lag} time steps (based on simulation).")

      # BW7
      series_1 = cl_df[cl_df['bwfl_id'] == 'BW1']['mean'].values
      series_2 = cl_df[cl_df['bwfl_id'] == 'BW7']['mean'].values
      cross_corr = np.correlate(series_1 - np.mean(series_1), series_2 - np.mean(series_2), mode='full')
      lags = np.arange(-len(series_1) + 1, len(series_1))
      best_lag = lags[np.argmax(cross_corr)]
      if best_lag < 0 and best_lag > -(24*4):
         cl_df.loc[cl_df['bwfl_id'] == 'BW7', 'datetime'] += pd.Timedelta(minutes=15 * best_lag)
         # print(f"Data period {idx}. Shifted BW7 by {best_lag} time steps.")
      else:
         best_lag = -int(round(mean_age_after_1_day['BW7'] * 4))
         cl_df.loc[cl_df['bwfl_id'] == 'BW7', 'datetime'] += pd.Timedelta(minutes=15 * best_lag)
         # print(f"Data period {idx}. Shifted BW7 by {best_lag} time steps (based on simulation).")

      # BW9
      series_1 = cl_df[cl_df['bwfl_id'] == 'BW4']['mean'].values
      series_2 = cl_df[cl_df['bwfl_id'] == 'BW9']['mean'].values
      cross_corr = np.correlate(series_1 - np.mean(series_1), series_2 - np.mean(series_2), mode='full')
      lags = np.arange(-len(series_1) + 1, len(series_1))
      best_lag = lags[np.argmax(cross_corr)]
      if best_lag < 0 and best_lag > -(24*4):
         cl_df.loc[cl_df['bwfl_id'] == 'BW9', 'datetime'] += pd.Timedelta(minutes=15 * best_lag)
         # print(f"Data period {idx}. Shifted BW9 by {best_lag} time steps.")
      else:
         best_lag = -int(round(mean_age_after_1_day['BW9'] * 4))
         cl_df.loc[cl_df['bwfl_id'] == 'BW9', 'datetime'] += pd.Timedelta(minutes=15 * best_lag)
         # print(f"Data period {idx}. Shifted BW9 by {best_lag} time steps (based on simulation).")

      # BW2 (source is almost entirely BW4)
      series_1 = cl_df[cl_df['bwfl_id'] == 'BW4']['mean'].values
      series_2 = cl_df[cl_df['bwfl_id'] == 'BW2']['mean'].values
      cross_corr = np.correlate(series_1 - np.mean(series_1), series_2 - np.mean(series_2), mode='full')
      lags = np.arange(-len(series_1) + 1, len(series_1))
      best_lag = lags[np.argmax(cross_corr)]
      if best_lag < 0 and best_lag > -(24*4):
         cl_df.loc[cl_df['bwfl_id'] == 'BW2', 'datetime'] += pd.Timedelta(minutes=15 * best_lag)
         # print(f"Data period {idx}. Shifted BW2 by {best_lag} time steps.")
      else:
         best_lag = -int(round(mean_age_after_1_day['BW2'] * 4))
         cl_df.loc[cl_df['bwfl_id'] == 'BW2', 'datetime'] += pd.Timedelta(minutes=15 * best_lag)
         # print(f"Data period {idx}. Shifted BW2 by {best_lag} time steps (based on simulation).")

      # BW5 (source is almost entirely BW1)
      series_1 = cl_df[cl_df['bwfl_id'] == 'BW1']['mean'].values
      series_2 = cl_df[cl_df['bwfl_id'] == 'BW5']['mean'].values
      cross_corr = np.correlate(series_1 - np.mean(series_1), series_2 - np.mean(series_2), mode='full')
      lags = np.arange(-len(series_1) + 1, len(series_1))
      best_lag = lags[np.argmax(cross_corr)]
      if best_lag < 0 and best_lag > -(24*4):
         cl_df.loc[cl_df['bwfl_id'] == 'BW5', 'datetime'] += pd.Timedelta(minutes=15 * best_lag)
         # print(f"Data period {idx}. Shifted BW5 by {best_lag} time steps.")
      else:
         best_lag = -int(round(mean_age_after_1_day['BW5'] * 4))
         cl_df.loc[cl_df['bwfl_id'] == 'BW5', 'datetime'] += pd.Timedelta(minutes=15 * best_lag)
         # print(f"Data period {idx}. Shifted BW5 by {best_lag} time steps (based on simulation).")

      # BW6 (source is mostly BW1)
      series_1 = cl_df[cl_df['bwfl_id'] == 'BW1']['mean'].values
      series_2 = cl_df[cl_df['bwfl_id'] == 'BW6']['mean'].values
      cross_corr = np.correlate(series_1 - np.mean(series_1), series_2 - np.mean(series_2), mode='full')
      lags = np.arange(-len(series_1) + 1, len(series_1))
      best_lag = lags[np.argmax(cross_corr)]
      if best_lag < 0 and best_lag > -(24*4):
         cl_df.loc[cl_df['bwfl_id'] == 'BW6', 'datetime'] += pd.Timedelta(minutes=15 * best_lag)
         # print(f"Data period {idx}. Shifted BW6 by {best_lag} time steps.")
      else:
         best_lag = -int(round(mean_age_after_1_day['BW6'] * 4))
         cl_df.loc[cl_df['bwfl_id'] == 'BW6', 'datetime'] += pd.Timedelta(minutes=15 * best_lag)
         # print(f"Data period {idx}. Shifted BW6 by {best_lag} time steps (based on simulation).")

   cl_df = cl_df[cl_df['datetime'] >= period['start_date']]
   min_end_datetime = cl_df.groupby('bwfl_id')['datetime'].max().min()
   cl_df = cl_df[cl_df['datetime'] <= min_end_datetime]

   # 2. check the that the following conditions are met within the specified tolerance

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

   
   # export to csv files
   if not filtered_wq_df.empty:
      filtered_pressure_df.to_csv(output_dir / f"{str(idx).zfill(2)}-pressure.csv", index=False)
      filtered_flow_df.to_csv(output_dir / f"{str(idx).zfill(2)}-flow.csv", index=False)
      filtered_wq_df.to_csv(output_dir / f"{str(idx).zfill(2)}-wq.csv", index=False)

data_period_json_path = output_dir / 'data_periods.json'
with open(data_period_json_path, 'w') as json_file:
   json.dump(good_data, json_file, default=str)