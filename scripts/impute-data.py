"""
This script imputes missing values in flow_df and boundary heads in pressure_df for the entire dataset.
"""

import pandas as pd
import numpy as np
import zipfile as zip
from datetime import datetime, timedelta
from pathlib import Path
from bayesian_wq_calibration.constants import TIMESERIES_DIR
import argparse
import json
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


# pass script arguments
parser = argparse.ArgumentParser(description='Impute time series data.')
parser.add_argument('--method', type=str, default="mean", help='Data imputation method (default: mean).')
args = parser.parse_args()

# get time series data
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




# ensure dataframe size is correct
def fill_missing_rows(df, datetime_col, freq, id_cols):
    min_datetime = df[datetime_col].min()
    max_datetime = df[datetime_col].max()
    complete_datetime_range = pd.date_range(start=min_datetime, end=max_datetime, freq=freq)

    bwfl_ids = df['bwfl_id'].unique()
    complete_df = pd.MultiIndex.from_product([complete_datetime_range, bwfl_ids], names=[datetime_col, 'bwfl_id']).to_frame(index=False)
    merged_df = pd.merge(complete_df, df, on=[datetime_col, 'bwfl_id'], how='left')

    for col in id_cols:
        merged_df[col] = merged_df.groupby('bwfl_id')[col].transform('first')

    return merged_df

pressure_df = fill_missing_rows(
    pressure_df, 
    datetime_col='datetime', 
    freq='15min', 
    id_cols=['dma_id', 'wwmd_id']
)

flow_df = fill_missing_rows(
    flow_df, 
    datetime_col='datetime', 
    freq='15min', 
    id_cols=['dma_id', 'wwmd_id']
)

wq_df = fill_missing_rows(
    wq_df, 
    datetime_col='datetime', 
    freq='15min', 
    id_cols=['dma_id', 'wwmd_id', 'data_type']
)


#  treat zero data points as NaNs for specific bwfl_ids
bwfl_ids_to_exclude = ['wastemeter_2307', 'wastemeter_2308']
value_columns = ['min', 'mean', 'max']
for col in value_columns:
    flow_df.loc[~flow_df['bwfl_id'].isin(bwfl_ids_to_exclude) & (flow_df[col] == 0), col] = np.nan


# check missing data
for sensor in pressure_df['bwfl_id'].unique():
   print(f"Pressure data. Sensor: {sensor}. Missing data count: {pressure_df[pressure_df['bwfl_id'] == sensor]['mean'].isna().sum()} of {len(pressure_df[pressure_df['bwfl_id'] == sensor])}")
for sensor in flow_df['bwfl_id'].unique():
   print(f"Flow data. Sensor: {sensor}. Missing data count: {flow_df[flow_df['bwfl_id'] == sensor]['mean'].isna().sum()} of {len(flow_df[flow_df['bwfl_id'] == sensor])}")
for sensor in wq_df['bwfl_id'].unique():
   print(f"WQ data. Sensor: {sensor}. Missing data count: {wq_df[wq_df['bwfl_id'] == sensor]['mean'].isna().sum()} of {len(wq_df[wq_df['bwfl_id'] == sensor])}")


# impute missing flow data
flow_impute_df = flow_df.copy()
flow_impute_df['time'] = flow_impute_df['datetime'].dt.strftime('%H:%M')
flow_impute_df['time'] = flow_impute_df['datetime'].dt.hour.astype(float) + flow_impute_df['datetime'].dt.minute.astype(float) / 60
flow_impute_df['time'] = (flow_impute_df['time'] * 4).round() / 4
flow_impute_df['dayofweek'] = flow_impute_df['datetime'].dt.dayofweek.astype(float)
flow_impute_df["month"] = flow_impute_df["datetime"].dt.month
# flow_impute_df["year"] = flow_impute_df["datetime"].dt.year


if args.method == 'mean':
    value_columns = ['min', 'mean', 'max']
    grouped = flow_impute_df.groupby(['bwfl_id', 'time', 'dayofweek', 'month'])[value_columns]
    flow_impute_df[value_columns] = grouped.transform(lambda x: x.fillna(x.mean()))

elif args.method == 'knn':
    for sensor in flow_df['bwfl_id'].unique():
        print(f"Imputing sensor: {sensor}")
        temp_df = flow_impute_df[flow_impute_df['bwfl_id'] == sensor].drop(columns=['datetime', 'bwfl_id', 'dma_id', 'wwmd_id']).copy()
        print(temp_df)
        impute_features = ['time', 'dayofweek', 'min', 'mean', 'max']
        knn_imputer = KNNImputer(n_neighbors=3)
        imputed_array = knn_imputer.fit_transform(temp_df[impute_features])
        temp_df_imputed = pd.DataFrame(imputed_array, columns=impute_features)
        flow_impute_df.loc[flow_impute_df['bwfl_id'] == sensor, ['min', 'mean', 'max']] = temp_df_imputed[['min', 'mean', 'max']]

else:
    raise ValueError(f"Invalid imputation method: {args.method}")

# bound minimum maximum values from data imputer
max_values = flow_df[['min', 'mean', 'max']].max()
min_vales = flow_df[['min', 'mean', 'max']].min()
flow_impute_df['min'] = flow_impute_df['min'].clip(upper=max_values['min'], lower=min_vales['min'])
flow_impute_df['mean'] = flow_impute_df['mean'].clip(upper=max_values['mean'], lower=min_vales['mean'])
flow_impute_df['max'] = flow_impute_df['max'].clip(upper=max_values['max'], lower=min_vales['max'])

flow_df[value_columns] = flow_impute_df[value_columns]


# save imputed date to zip folder
with zip.ZipFile(TIMESERIES_DIR / 'imputed/field_lab-data-2021-2024.zip', 'w') as z:
    with z.open('flow_data.csv', 'w') as f:
        flow_df.to_csv(f, index=False)
    with z.open('pressure_data.csv', 'w') as f:
        pressure_df.to_csv(f, index=False)
    with z.open('wq_data.csv', 'w') as f:
        wq_df.to_csv(f, index=False)