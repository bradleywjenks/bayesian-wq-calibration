"""
This script identifies low chlorine residuals (< 0.1 mg/L) in water quality data, removes outliers, and generates interactive plots and statistics.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import os

TIMESERIES_PATH = "/Users/bradwjenks/Code/PhD/bayesian-wq-calibration/data/timeseries"

###### Step 1: Load and prepare data ######
df = pd.read_csv(os.path.join(TIMESERIES_PATH, 'bwfl_wq_2024-02-20_to_2024-10-02.csv'), sep=';', low_memory=False)
# df = pd.read_csv(os.path.join(TIMESERIES_PATH, 'bwfl_wq_2024-12-20_to_2025-03-30.csv'), sep=';', low_memory=False)
# df = pd.read_csv(os.path.join(TIMESERIES_PATH, 'bwfl_wq_2021-01-04_to_2023-12-17_BW7.csv'), sep=';', low_memory=False)

df['datetime'] = pd.to_datetime(df['datetime'])
df = df[df['data_type'] == 'chlorine'].copy()
df['mean'] = pd.to_numeric(df['mean'], errors='coerce')


###### Step 2: Remove zero values ######
df['filtered_mean'] = df['mean'].copy()
df['is_outlier'] = False
df.loc[df['mean'] == 0, 'is_outlier'] = True
df.loc[df['mean'] == 0, 'filtered_mean'] = np.nan


# print(df.head(100))



###### Step 3: Identify low residuals (< 0.1 mg/L) ######
df['is_low_residual'] = (df['filtered_mean'] < 0.1) & df['filtered_mean'].notna()



###### Step 4: Create clean plots for each sensor ######
bwfl_ids = df['bwfl_id'].unique()

for bwfl_id in bwfl_ids:
    subset = df[df['bwfl_id'] == bwfl_id]
    
    # Skip if no data
    if subset['filtered_mean'].notna().sum() == 0:
        continue
    
    fig = go.Figure()
    
    # Normal points as lines (grey)
    fig.add_trace(
        go.Scatter(
            x=subset['datetime'],
            y=subset['filtered_mean'],
            mode='lines',
            line=dict(color='grey', width=2),
            name='Time series',
            showlegend=True,
            connectgaps=False  # leave gaps for NaNs
        )
    )
    
    # Low residual points (red dots)
    low = subset[subset['is_low_residual']]
    if len(low) > 0:
        fig.add_trace(
            go.Scatter(x=low['datetime'], y=low['filtered_mean'],
                       mode='markers', 
                       marker=dict(color='red', size=8),
                       name='Low residual',
                       showlegend=True)
        )

    fig.update_layout(
        title=dict(text=f'Chlorine Levels - {bwfl_id}', font=dict(size=24)),
        xaxis_title='',
        yaxis_title='Chlorine [mg/L]',
        height=500,  # Custom height
        width=1200,  # Custom width
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        legend=dict(
            title=None,
            font=dict(size=16),
            bgcolor='white',
            bordercolor='rgba(0,0,0,0)',
            borderwidth=0,
            x=0.85,
            y=1.05,
            xanchor='left',
            yanchor='top'
        ),
        xaxis=dict(
            range=['2024-03-01', '2024-08-31'],
            titlefont=dict(size=20),  # Axis label font
            tickfont=dict(size=18),   # Tick label font
            showgrid=False,  # No grid
            showline=True,
            linewidth=1,
            linecolor='black',
            ticks='outside',  # Ticks outside
            tickcolor='black',  # Black ticks
            tickwidth=1,
            ticklen=5
        ),
        yaxis=dict(
            range=[0,2],
            titlefont=dict(size=20),  # Axis label font
            tickfont=dict(size=18),   # Tick label font
            showgrid=False,  # No grid
            showline=True,
            linewidth=1,
            linecolor='black',
            ticks='outside',  # Ticks outside
            tickcolor='black',  # Black ticks
            tickwidth=1,
            ticklen=5
        )
    )
    
    fig.show()
    fig.write_html(f"chlorine_{bwfl_id}.html")







###### Step 5: Generate statistics ######
print("\n" + "="*60)
print("CHLORINE RESIDUAL ANALYSIS STATISTICS")
print("="*60)

total_measurements = df['filtered_mean'].notna().sum()
total_low = df['is_low_residual'].sum()
total_outliers = df['is_outlier'].sum()

print(f"\nOVERALL:")
print(f"  Measurements (filtered): {total_measurements}")
print(f"  Low residuals (<0.1): {total_low}")
print(f"  Percentage low: {total_low/total_measurements*100:.2f}%")
print(f"  Outliers removed: {total_outliers}")

for bwfl_id in df['bwfl_id'].unique():
    subset = df[df['bwfl_id'] == bwfl_id]
    filtered = subset['filtered_mean'].dropna()
    
    print(f"\n{bwfl_id}:")
    print(f"  Measurements: {len(filtered)}")
    print(f"  Low residuals: {subset['is_low_residual'].sum()}")
    print(f"  % Low: {subset['is_low_residual'].sum()/len(filtered)*100:.2f}%")
    print(f"  Mean: {filtered.mean():.3f} mg/L")
    print(f"  Min: {filtered.min():.3f} mg/L")
    print(f"  Max: {filtered.max():.3f} mg/L")
