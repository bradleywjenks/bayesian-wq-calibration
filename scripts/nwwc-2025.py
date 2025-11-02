"""
This script calculates monthly average chlorine and temperature from water quality data,
generates interactive dual-axis plots with box plots for chlorine, and produces summary statistics.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Wong color palette
wong_colors = [
    "rgb(89, 89, 89)",       # wong-black/grey
    "rgb(230, 159, 0)",      # wong-orange
    "rgb(86, 180, 233)",     # wong-skyblue
    "rgb(0, 158, 115)",      # wong-green
    "rgb(240, 228, 66)",     # wong-yellow
    "rgb(0, 114, 178)",      # wong-blue
    "rgb(213, 94, 0)",       # wong-vermillion
    "rgb(204, 121, 167)"     # wong-purple
]

TIMESERIES_PATH = "/Users/bradwjenks/Code/PhD/bayesian-wq-calibration/data/timeseries"

###### Step 1: Load and prepare data ######
# Select which CSV file to analyze (uncomment one):
# df = pd.read_csv(os.path.join(TIMESERIES_PATH, 'bwfl_wq_2021-01-01_to_2024-01-24_BW5.csv'), sep=';', low_memory=False)
df = pd.read_csv(os.path.join(TIMESERIES_PATH, 'bwfl_wq_2021-01-01_to_2024-01-24_BW6.csv'), sep=';', low_memory=False)
# df = pd.read_csv(os.path.join(TIMESERIES_PATH, 'bwfl_wq_2021-01-04_to_2023-12-17_BW7.csv'), sep=';', low_memory=False)

df['datetime'] = pd.to_datetime(df['datetime'])
df['mean'] = pd.to_numeric(df['mean'], errors='coerce')


###### Step 2: Separate chlorine and temperature data ######
chlorine_df = df[df['data_type'] == 'chlorine'].copy()
temp_df = df[df['data_type'] == 'temperature'].copy()

# Remove zero values from chlorine
chlorine_df['filtered_mean'] = chlorine_df['mean'].copy()
chlorine_df.loc[chlorine_df['mean'] == 0, 'filtered_mean'] = np.nan

temp_df['filtered_mean'] = temp_df['mean'].copy()
temp_df.loc[temp_df['mean'] == 0, 'filtered_mean'] = np.nan


###### Function to remove outliers using IQR method ######
def remove_outliers_iqr(data, column, iqr_multiplier=1.5):
    """Remove outliers using Interquartile Range (IQR) method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    # Mark outliers as NaN
    data.loc[(data[column] < lower_bound) | (data[column] > upper_bound), column] = np.nan
    
    return data


# Apply outlier removal for each bwfl_id
for bwfl_id in chlorine_df['bwfl_id'].unique():
    chlorine_df.loc[chlorine_df['bwfl_id'] == bwfl_id, 'filtered_mean'] = remove_outliers_iqr(
        chlorine_df[chlorine_df['bwfl_id'] == bwfl_id].copy(),
        'filtered_mean'
    )['filtered_mean'].values

for bwfl_id in temp_df['bwfl_id'].unique():
    temp_df.loc[temp_df['bwfl_id'] == bwfl_id, 'filtered_mean'] = remove_outliers_iqr(
        temp_df[temp_df['bwfl_id'] == bwfl_id].copy(),
        'filtered_mean'
    )['filtered_mean'].values


###### Step 3: Extract calendar month ######
chlorine_df['month'] = chlorine_df['datetime'].dt.month
temp_df['month'] = temp_df['datetime'].dt.month


###### Step 4: Calculate average values by calendar month (Jan-Dec) ######
monthly_chlorine = chlorine_df.groupby(['bwfl_id', 'month'])['filtered_mean'].agg(['mean', 'count']).reset_index()
monthly_chlorine.columns = ['bwfl_id', 'month', 'avg_chlorine', 'chlorine_count']

monthly_temp = temp_df.groupby(['bwfl_id', 'month'])['filtered_mean'].agg(['mean', 'count']).reset_index()
monthly_temp.columns = ['bwfl_id', 'month', 'avg_temperature', 'temp_count']

# Merge chlorine and temperature data
monthly_data = pd.merge(monthly_chlorine, monthly_temp, on=['bwfl_id', 'month'], how='outer')

# Keep month as the x-axis value (1-12)
print(monthly_data.head(20))


###### Step 5: Identify low residuals and create time series plots ######
chlorine_df['is_low_residual'] = (chlorine_df['filtered_mean'] < 0.1) & chlorine_df['filtered_mean'].notna()

# Create time series plots for each sensor
bwfl_ids_ts = chlorine_df['bwfl_id'].unique()

for bwfl_id in bwfl_ids_ts:
    subset_ts = chlorine_df[chlorine_df['bwfl_id'] == bwfl_id].sort_values('datetime')
    
    # Skip if no data
    if subset_ts['filtered_mean'].notna().sum() == 0:
        continue
    
    fig_ts = go.Figure()
    
    # Time series line (grey)
    fig_ts.add_trace(
        go.Scatter(
            x=subset_ts['datetime'],
            y=subset_ts['filtered_mean'],
            mode='lines',
            line=dict(color='grey', width=2),
            name='Time series',
            showlegend=True
        )
    )
    
    # Low residual points (red dots)
    low = subset_ts[subset_ts['is_low_residual']]
    if len(low) > 0:
        fig_ts.add_trace(
            go.Scatter(
                x=low['datetime'],
                y=low['filtered_mean'],
                mode='markers',
                marker=dict(color='red', size=8),
                name='Low residual',
                showlegend=True
            )
        )
    
    fig_ts.update_layout(
        height=500,
        width=1200,
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        legend=dict(
            title=None,
            font=dict(size=16),
            bgcolor='white',
            bordercolor='rgba(0,0,0,0)',
            borderwidth=0,
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top'
        ),
        xaxis=dict(
            titlefont=dict(size=20),
            tickfont=dict(size=18),
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            ticks='outside',
            tickcolor='black',
            tickwidth=1,
            ticklen=5
        ),
        yaxis=dict(
            range=[0, 1],
            title_text='Chlorine [mg/L]',
            titlefont=dict(size=20),
            tickfont=dict(size=18),
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            ticks='outside',
            tickcolor='black',
            tickwidth=1,
            ticklen=5
        )
    )
    
    fig_ts.show()
    fig_ts.write_html(f"timeseries_chlorine_{bwfl_id}.html")


###### Step 6: Create dual-axis plots with box plots and line plot for each sensor ######
bwfl_ids = monthly_data['bwfl_id'].unique()

# Create month names for display
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for bwfl_id in bwfl_ids:
    subset_monthly = monthly_data[monthly_data['bwfl_id'] == bwfl_id].sort_values('month')
    
    # Skip if no data
    if subset_monthly['avg_chlorine'].notna().sum() == 0:
        continue
    
    subset_monthly['month_name'] = subset_monthly['month'].map(lambda x: month_names[x-1])
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        specs=[[{"secondary_y": True}]]
    )
    
    # Add box plots for chlorine by month (using original chlorine_df data)
    subset_chlorine = chlorine_df[chlorine_df['bwfl_id'] == bwfl_id].copy()
    
    for month in range(1, 13):
        month_data = subset_chlorine[subset_chlorine['month'] == month]['filtered_mean'].dropna()
        
        if len(month_data) > 0:
            fig.add_trace(
                go.Box(
                    y=month_data,
                    name=month_names[month-1],
                    x=[month_names[month-1]] * len(month_data),  # Create x values matching data length
                    showlegend=False,
                    boxmean=False,
                    marker=dict(color=wong_colors[5], size=4),  # wong-blue
                    line=dict(color=wong_colors[5], width=2),
                    fillcolor='rgba(0, 114, 178, 0.3)',  # wong-blue with transparency
                    boxpoints=False  # Hide individual points
                ),
                secondary_y=False
            )
    
    # Add temperature trace (secondary y-axis) - monthly averages as line plot
    fig.add_trace(
        go.Scatter(
            x=subset_monthly['month_name'],
            y=subset_monthly['avg_temperature'],
            mode='lines+markers',
            line=dict(color=wong_colors[6], width=2),  # wong-vermillion
            marker=dict(
                size=10,
                color=wong_colors[6],
                line=dict(color='white', width=1)  # White outline
            ),
            name='Mean temperature',
            showlegend=True
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        width=1200,
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        legend=dict(
            title=None,
            font=dict(size=16),
            bgcolor='white',
            bordercolor='rgba(0,0,0,0)',
            borderwidth=0,
            x=1.05,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        xaxis=dict(
            titlefont=dict(size=20),
            tickfont=dict(size=18),
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            ticks='outside',
            tickcolor='black',
            tickwidth=1,
            ticklen=5
        )
    )
    
    # Update primary y-axis (Chlorine) - wong-blue
    fig.update_yaxes(
        title_text='Chlorine [mg/L]',
        titlefont=dict(size=20, color=wong_colors[5]),
        tickfont=dict(size=18, color=wong_colors[5]),
        showgrid=False,
        showline=True,
        linewidth=2,
        linecolor=wong_colors[5],
        ticks='outside',
        tickcolor=wong_colors[5],
        tickwidth=1,
        ticklen=5,
        secondary_y=False
    )
    
    # Update secondary y-axis (Temperature) - wong-vermillion
    fig.update_yaxes(
        title_text='Temperature [°C]',
        titlefont=dict(size=20, color=wong_colors[6]),
        tickfont=dict(size=18, color=wong_colors[6]),
        showgrid=False,
        showline=True,
        linewidth=2,
        linecolor=wong_colors[6],
        ticks='outside',
        tickcolor=wong_colors[6],
        tickwidth=1,
        ticklen=5,
        secondary_y=True
    )
    
    fig.show()
    fig.write_html(f"monthly_chlorine_temperature_{bwfl_id}.html")


###### Step 7: Generate statistics ######
print("\n" + "="*60)
print("LOW RESIDUAL ANALYSIS")
print("="*60)

# Calculate overall low residual statistics
total_measurements = chlorine_df['filtered_mean'].notna().sum()
total_low = chlorine_df['is_low_residual'].sum()
pct_low = (total_low / total_measurements * 100) if total_measurements > 0 else 0

print(f"\nOVERALL:")
print(f"  Total measurements: {total_measurements}")
print(f"  Low residuals (<0.1 mg/L): {total_low}")
print(f"  Percentage low: {pct_low:.2f}%")

# Per-sensor statistics
print(f"\nPER-SENSOR:")
for bwfl_id in chlorine_df['bwfl_id'].unique():
    subset = chlorine_df[chlorine_df['bwfl_id'] == bwfl_id]
    total = subset['filtered_mean'].notna().sum()
    low = subset['is_low_residual'].sum()
    pct = (low / total * 100) if total > 0 else 0
    
    print(f"\n  {bwfl_id}:")
    print(f"    Measurements: {total}")
    print(f"    Low residuals: {low}")
    print(f"    Percentage low: {pct:.2f}%")


print("\n" + "="*60)
print("MONTHLY CHLORINE AND TEMPERATURE ANALYSIS")
print("="*60)

for bwfl_id in monthly_data['bwfl_id'].unique():
    subset = monthly_data[monthly_data['bwfl_id'] == bwfl_id]
    
    print(f"\n{bwfl_id}:")
    print(f"  Months analyzed: {len(subset)}")
    print(f"\n  CHLORINE:")
    print(f"    Mean (monthly avg): {subset['avg_chlorine'].mean():.3f} mg/L")
    print(f"    Min (monthly avg): {subset['avg_chlorine'].min():.3f} mg/L")
    print(f"    Max (monthly avg): {subset['avg_chlorine'].max():.3f} mg/L")
    print(f"    Std Dev: {subset['avg_chlorine'].std():.3f} mg/L")
    print(f"\n  TEMPERATURE:")
    print(f"    Mean (monthly avg): {subset['avg_temperature'].mean():.3f} °C")
    print(f"    Min (monthly avg): {subset['avg_temperature'].min():.3f} °C")
    print(f"    Max (monthly avg): {subset['avg_temperature'].max():.3f} °C")
    print(f"    Std Dev: {subset['avg_temperature'].std():.3f} °C")
    
    # Calculate correlation
    valid_data = subset[['avg_chlorine', 'avg_temperature']].dropna()
    if len(valid_data) > 1:
        correlation = valid_data['avg_chlorine'].corr(valid_data['avg_temperature'])
        print(f"\n  CORRELATION:")
        print(f"    Chlorine vs Temperature: {correlation:.3f}")