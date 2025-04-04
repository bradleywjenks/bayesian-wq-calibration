"""
This script calibrates the Field Lab's water quality model using ensemble Kalman inversion (Stage 1 of the CES method). Specifically, bulk and wall decay coefficients are calibrated based on a selected grouping. The following steps are performed:
    1. load operational data
    2. split train/test datasets
    3. build epanet model
    4. set pipe grouping and define θ priors
    5. create forward model function, F <-- wn, θ, grouping
    6. create EKI calibration function: EKI <-- F, θ_prior
    7. enter text here...
"""


using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using DataFrames
using CSV
using Dates
using LinearAlgebra
using Distributions
using Statistics
using Colors
using Random
using Plots.PlotMeasures
using PyCall

pd = pyimport("pandas")
np = pyimport("numpy")
data = pyimport("bayesian_wq_calibration.data")
epanet = pyimport("bayesian_wq_calibration.epanet")

const TIMESERIES_PATH = "/Users/bradwjenks/Code/PhD/bayesian-wq-calibration/data/timeseries"
const EKP = EnsembleKalmanProcesses

wong_colors = [
    RGB(89/255, 89/255, 89/255),       # wong-black/grey
    RGB(230/255, 159/255, 0/255),      # wong-orange
    RGB(86/255, 180/255, 233/255),     # wong-skyblue
    RGB(0/255, 158/255, 115/255),      # wong-green
    RGB(240/255, 228/255, 66/255),     # wong-yellow
    RGB(0/255, 114/255, 178/255),      # wong-blue
    RGB(213/255, 94/255, 0/255),       # wong-vermillion
    RGB(204/255, 121/255, 167/255)     # wong-purple
]


########## MAIN SCRIPT ##########

### 1. load operational data ###
data_period = 18 # (Aug. 2024)
padded_period = lpad(data_period, 2, "0")

flow_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-flow.csv", DataFrame); flow_df.datetime = DateTime.(flow_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
pressure_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-pressure.csv", DataFrame); pressure_df.datetime = DateTime.(pressure_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
wq_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-wq.csv", DataFrame); wq_df.datetime = DateTime.(wq_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
cl_df = wq_df[wq_df.data_type .== "chlorine", :]

ylabel = "Chlorine [mg/L]" # "Flow [L/s]", "Pressure [m]", "Chlorine [mg/L]"
df = cl_df # cl_df, flow_df, pressure_df
plot_bwfl_data(df, ylabel)



### 2. split train/test datasets ###
unique_datetime = DateTime.(unique(df.datetime))
n_total = length(unique_datetime) # 15-minute time steps
range = 1:n_total
datetime = unique_datetime[total_range]

n_train = 3 * 24 * 4  # 2 train days (day 1 discarded in wq simulation)
range_train = 1:n_train
datetime_train = unique_datetime[range_train]

range_test = (n_train + 1):n_total
datetime_test = unique_datetime[range_test]

ylabel = "Chlorine [mg/L]" # "Flow [L/s]", "Pressure [m]", "Chlorine [mg/L]"
df_filter = filter(row -> row.datetime ∈ datetime_train, cl_df)
plot_bwfl_data(df_filter, ylabel)



### 3. build epanet model ###
demand_resolution = "wwmd"

θ_b = -0.7  # day^-1 (from bottle tests)
temp_train = mean(skipmissing(subset(wq_df, :data_type => ByRow(==("temperature")), :datetime => ByRow(in(datetime_train)))[:, :mean]))
θ_b_train = data.bulk_temp_adjust(θ_b, temp_train)

wn_train = epanet.build_model(
    df_2_pd(filter(row -> row.datetime ∈ train_datetime, flow_df)), 
    df_2_pd(filter(row -> row.datetime ∈ train_datetime, pressure_df)), 
    df_2_pd(filter(row -> row.datetime ∈ train_datetime, cl_df)),
    sim_type="chlorine",
    demand_resolution=demand_resolution,
    bulk_coeff=θ_b_train
)






########## FUNCTIONS ##########

function pd_2_df(df_pd)
    df= DataFrame()
    for col in df_pd.columns
        df[!, col] = getproperty(df_pd, col).values
    end
    return df
end

function df_2_pd(df)
    df_clean = mapcols(col -> coalesce.(col, np.nan), df)
    return pd.DataFrame(Dict(col => df_clean[!, col] for col in names(df_clean)))
end



function plot_bwfl_data(df, ylabel)

    bwfl_ids = unique(df.bwfl_id)

    plt = plot(xlabel="Datetime", ylabel=ylabel, legend=:outertopright, foreground_color_legend=nothing, size=(1000, 450), left_margin=8mm, bottom_margin=8mm, top_margin=8mm)

    for (i, name) in enumerate(bwfl_ids)
        color = wong_colors[mod1(i, length(wong_colors))]
        df_subset = df[df.bwfl_id .== name, :]
        plot!(plt, df_subset.datetime, df_subset.mean, label=name, lw=1.5, color=color)
    end

    return plt
end