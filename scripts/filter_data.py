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

# data directory
data_dir = '/home/bradw/workspace/bayesian-wq-calibration/data/'

# pass script arguments
parser = argparse.ArgumentParser(description='Process raw time series files to create data periods for model calibration.')
parser.add_argument('--n_days', type=int, default=7, help='Number of good days for data filtering (default: 7 days).')
args = parser.parse_args()




##### GET GOOD DATA PERIODS #####

calibration_data = pd.read_excel(data_dir + 'raw/metrinet-calibration-records.xlsx')
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



##### FILTER TIME SERIES DATA #####

output_dir = Path(data_dir + 'filtered/')

with zip.ZipFile(data_dir + 'raw/field_lab-data-2021-2024.zip', 'r') as z:

   for filename in z.namelist():
       if filename.endswith('.csv'):
           with z.open(filename) as f:
               data = pd.read_csv(f, sep=';')
               data['datetime'] = pd.to_datetime(data['datetime'])

               for idx, period in good_data.items():
                   filtered_data = data[(data['datetime'] >= period['start_date']) & (data['datetime'] <= period['end_date'])]
                   if not filtered_data.empty:
                       data_type = filename.split('/')[-1].split('.')[0].split('_')[1]
                       output_filename = output_dir / f"{str(idx).zfill(2)}-{data_type}.csv"
                       filtered_data.to_csv(output_filename, index=False)
                       print(f'Saved: {output_filename}')

data_period_json_path = output_dir / 'data_periods.json'
with open(data_period_json_path, 'w') as json_file:
   json.dump(good_data, json_file, default=str)
   print(f'Saved data period file to: {data_period_json_path}')