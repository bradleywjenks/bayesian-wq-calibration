"""
This script outputs flow, pressure, and wq time series data for a period of n_days after each metrinet sensor calibration event.
"""

import pandas as pd
import numpy as np
import zipfile as zip
from pathlib import Path
import argparse
import json
from bayesian_wq_calibration.epanet import model_simulation
from bayesian_wq_calibration.data import sensor_model_id
from bayesian_wq_calibration.constants import TIMESERIES_DIR
import warnings
warnings.filterwarnings("ignore")


# pass script arguments
parser = argparse.ArgumentParser(description='Process raw time series files to create data periods for model calibration.')
parser.add_argument('--n_days', type=int, default=8, help='Number of good days for data filtering (default: 7 days).')
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
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW3') & (filtered_wq_df['datetime'] >= pd.to_datetime('2024-06-09 07:15:00')) & (filtered_wq_df['datetime'] <= pd.to_datetime('2024-06-12 07:15:00'))].index, value_columns] = np.nan
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
   elif idx == 20:
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW9_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW4_2') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW12_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW1_1') & (filtered_wq_df['data_type'] == 'chlorine')].index, inplace=True)
      filtered_wq_df.loc[filtered_wq_df[(filtered_wq_df['bwfl_id'] == 'BW7') & (filtered_wq_df['data_type'] == 'chlorine')].index, value_columns] = np.nan

   filtered_wq_df.drop(filtered_wq_df[(filtered_wq_df['bwfl_id'].isin(['BW1_2', 'BW12_2', 'BW4_2', 'BW9_2'])) & (filtered_wq_df['data_type'] != 'chlorine')].index,inplace=True)
   filtered_wq_df['bwfl_id'] = filtered_wq_df['bwfl_id'].str.split('_').str[0]

   
   # export to csv files
   if not filtered_wq_df.empty:
      filtered_pressure_df.to_csv(output_dir / f"{str(idx).zfill(2)}-pressure.csv", index=False)
      filtered_flow_df.to_csv(output_dir / f"{str(idx).zfill(2)}-flow.csv", index=False)
      filtered_wq_df.to_csv(output_dir / f"{str(idx).zfill(2)}-wq.csv", index=False)


data_period_json_path = output_dir / 'data_periods.json'
with open(data_period_json_path, 'w') as json_file:
   json.dump(good_data, json_file, default=str)