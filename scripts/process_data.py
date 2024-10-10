"""
This script outputs flow, pressure, and wq time series data for a period of N days after each metrinet sensor calibration date.
"""

import pandas as pd
import numpy as np
import zipfile as zip
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import json
from bayesian_wq_calibration.constants import TIMESERIES_DIR


# pass script arguments
parser = argparse.ArgumentParser(description='Process raw time series files to create data periods for model calibration.')
parser.add_argument('--n_days', type=int, default=7, help='Number of good days for data filtering (default: 7 days).')
args = parser.parse_args()




##### GET GOOD DATA PERIODS #####

calibration_data = pd.read_excel(TIMESERIES_DIR / 'raw/metrinet-calibration-records.xlsx')
if 'datetime' not in calibration_data.columns:
   raise ValueError("The specified file must contain a 'datetime' column.")
calibration_dates = pd.Series(calibration_data['datetime'].dt.to_period('D').dt.to_timestamp().unique()).sort_values()

good_data = {}
for idx, cal_date in calibration_dates.items():
   start_date = cal_date + pd.Timedelta(days=1)
   end_date = start_date + pd.Timedelta(days=args.n_days-1, hours=23, minutes=45, seconds=0)

   good_data[idx+1] = { 
       'start_date': start_date,
       'end_date': end_date
   }

keys_to_delete = []
for idx, cal_date in calibration_dates.items():
   for j in good_data.keys():
       if good_data[j]['start_date'] <= cal_date <= good_data[j]['end_date']:
           if j not in keys_to_delete:
               print(f'Note! Good data period {j} violated by calibration date {cal_date}. Good data period removed.')
               keys_to_delete.append(j)

for key in keys_to_delete:
   del good_data[key]

good_data = {new_idx+1: value for new_idx, (old_idx, value) in enumerate(good_data.items())}



##### PROCESS TIME SERIES DATA #####

output_dir = TIMESERIES_DIR / 'processed/'
pressure_df = pd.DataFrame()
flow_df = pd.DataFrame()
wq_df = pd.DataFrame()

with zip.ZipFile(TIMESERIES_DIR / 'raw/field_lab-data-2021-2024.zip', 'r') as z:
   for filename in z.namelist():
      if filename.endswith('.csv'):
         with z.open(filename) as f:
            data = pd.read_csv(f, sep=';', low_memory=False)
            data_type = filename.split('/')[-1].split('.')[0].split('_')[1]
            if data_type == 'pressure':
               pressure_df = pd.concat([pressure_df, data])
            elif data_type == 'flow':
               flow_df = pd.concat([flow_df, data])
            else:
               wq_df = pd.concat([wq_df, data])
               
pressure_df['datetime'] = pd.to_datetime(pressure_df['datetime'])
flow_df['datetime'] = pd.to_datetime(flow_df['datetime'])
wq_df['datetime'] = pd.to_datetime(wq_df['datetime'])

# count missing data
for sensor in pressure_df['bwfl_id'].unique():
   print(f"Pressure data. Sensor: {sensor}. Missing data count: {pressure_df[pressure_df['bwfl_id'] == sensor]['mean'].isna().sum()} of {len(pressure_df[pressure_df['bwfl_id'] == sensor])}")
for sensor in flow_df['bwfl_id'].unique():
   print(f"Flow data. Sensor: {sensor}. Missing data count: {flow_df[flow_df['bwfl_id'] == sensor]['mean'].isna().sum()} of {len(flow_df[flow_df['bwfl_id'] == sensor])}")
for sensor in wq_df['bwfl_id'].unique():
   print(f"WQ data. Sensor: {sensor}. Missing data count: {wq_df[wq_df['bwfl_id'] == sensor]['mean'].isna().sum()} of {len(wq_df[wq_df['bwfl_id'] == sensor])}")


# impute flow data and boundary head data ('BWFL 19' and 'Woodland Way PRV (inlet)')


for idx, period in good_data.items():
   filtered_pressure_df = pressure_df[(pressure_df['datetime'] >= period['start_date']) & (pressure_df['datetime'] <= period['end_date'])]
   filtered_flow_df = flow_df[(flow_df['datetime'] >= period['start_date']) & (flow_df['datetime'] <= period['end_date'])]
   filtered_wq_df = wq_df[(wq_df['datetime'] >= period['start_date']) & (wq_df['datetime'] <= period['end_date'])]
   if not filtered_wq_df.empty:
      filtered_pressure_df.to_csv(output_dir / f"{str(idx).zfill(2)}-pressure.csv", index=False)
      filtered_flow_df.to_csv(output_dir / f"{str(idx).zfill(2)}-flow.csv", index=False)
      filtered_wq_df.to_csv(output_dir / f"{str(idx).zfill(2)}-wq.csv", index=False)

data_period_json_path = output_dir / 'data_periods.json'
with open(data_period_json_path, 'w') as json_file:
   json.dump(good_data, json_file, default=str)