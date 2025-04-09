"""
This script calibrates the Field Lab's water quality model using ensemble Kalman inversion (Stage 1 of the CES method). Specifically, bulk and wall decay coefficients are calibrated based on a selected grouping. The following steps are performed:
    1. load operational data (DONE)
    2. split train/test datasets (DONE)
    3. build epanet model (DONE)
    4. set pipe grouping and define θ priors (DONE)
    5. create forward model function, F <-- wn, θ, grouping (DONE)
    6. create EKI calibration function: EKI <-- F, θ_prior (DONE)
    7. results plotting
"""

using Revise
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
using Plots
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
plot_bwfl_data(df, ylabel, ymax=0.7)



### 2. split train/test datasets ###
unique_datetime = DateTime.(unique(df.datetime))
n_total = length(unique_datetime) # 15-minute time steps
range = 1:n_total
datetime = unique_datetime[range]

n_train = 3 * 24 * 4  # 2 train days (day 1 discarded in wq simulation)
range_train = 1:n_train
datetime_train = unique_datetime[range_train]

range_test = (n_train + 1):n_total
datetime_test = unique_datetime[range_test]

ylabel = "Chlorine [mg/L]" # "Flow [L/s]", "Pressure [m]", "Chlorine [mg/L]"
df_filter = filter(row -> row.datetime ∈ datetime_train, cl_df)
plot_bwfl_data(df_filter, ylabel, ymax=0.7)



### 3. build epanet model ###
demand_resolution = "wwmd"

θ_b = -0.7  # day^-1 (from bottle tests)
temp_train = mean(skipmissing(subset(wq_df, :data_type => ByRow(==("temperature")), :datetime => ByRow(in(datetime_train)))[:, :mean]))
θ_b_train = data.bulk_temp_adjust(θ_b, temp_train)

wn_train = epanet.build_model(
    df_2_pd(filter(row -> row.datetime ∈ datetime_train, flow_df)), 
    df_2_pd(filter(row -> row.datetime ∈ datetime_train, pressure_df)), 
    df_2_pd(filter(row -> row.datetime ∈ datetime_train, cl_df)),
    sim_type="chlorine",
    demand_resolution=demand_resolution,
    bulk_coeff=θ_b_train
)



### 4. set θ_w groupings and bounds ###
grouping = "material" # "single", "material", "material-age", "material-age-velocity"

θ_w_lb, θ_w_ub = if grouping == "single"
    ([-1.0], [0.0])  # G1: all pipes
elseif grouping == "material"
    ([-1.0, -0.15], [0.0, 0.0])  # G1: metallic, G2: cement + plastic
elseif grouping == "material-age"
    ([-1.0, -1.0, -0.15], [0.0, 0.0, 0.0])  # G1: metallic + > mean pipe age, G2: metallic + ≤ mean pipe age, G3: cement + plastic
elseif grouping == "material-age-velocity"
    ([-1.0, -1.0, -1.0, -1.0, -0.15], [0.0, 0.0, 0.0, 0.0, 0.0])  # G1: metallic + > mean pipe age + ≤ mean velocity, G2: metallic + > mean pipe age + > mean velocity, G3: metallic + ≤ mean pipe age + ≤ mean velocity, G4: metallic + ≤ mean pipe age + > mean velocity, G5: cement + plastic
else
    error("Unsupported grouping: $grouping")
end



### 5. create and test forward model F(θ) ###
θ_w = (θ_w_lb + θ_w_ub) ./ 2
θ = [θ_b_train; θ_w]
exclude_sensors = ["BW1", "BW4", "BW7"]
y = forward_model(wn_train, θ, grouping, datetime_train, exclude_sensors; sim_type="chlorine")



### 6. eki calibration ###
θ_init, θ_final = eki_calibration(θ_b_train, θ_w_lb, θ_w_ub, cl_df, wn_train, datetime_train, exclude_sensors, grouping;burn_in=24*4, δ_s=0.15, δ_b=0.025)



### 7. results plotting ###

# θ_b
histogram(θ_init[1, :], label="Initial ensemble", color=wong_colors[1])
histogram!(θ_final[1, :], label="Final ensemble", color=wong_colors[2])

# θ_w ...
histogram(θ_init[2, :], label="Initial ensemble", color=wong_colors[1])
histogram(θ_final[2, :], label="Final ensemble", color=wong_colors[3])









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



function plot_bwfl_data(df, ylabel; ymax=nothing)

    bwfl_ids = unique(df.bwfl_id)

    plt = plot(xlabel="Datetime", ylabel=ylabel, legend=:outertopright, foreground_color_legend=nothing, ylims=(0, ymax), size=(1000, 450), left_margin=8mm, bottom_margin=8mm, top_margin=8mm, xtickfont=14, ytickfont=14, xguidefont=16, yguidefont=16, legendfont=14)

    for (i, name) in enumerate(bwfl_ids)
        color = wong_colors[mod1(i, length(wong_colors))]
        df_subset = df[df.bwfl_id .== name, :]
        plot!(plt, df_subset.datetime, df_subset.mean, label=name, lw=1.5, color=color)
    end

    return plt
end


function forward_model(wn, θ, grouping, datetime, exclude_sensors; sim_type="flow", burn_in=96)

    θ_b = θ[1]
    θ_w = θ[2:end]
    wn = epanet.set_reaction_parameters(wn, grouping, θ_w, θ_b)
    y = epanet.epanet_simulator(wn, sim_type, datetime)
    
    if sim_type == "chlorine"
        bwfl_ids = Vector{String}(data.sensor_model_id("wq")["bwfl_id"].values)
        model_ids = Vector{String}(data.sensor_model_id("wq")["model_id"].values)
        sensor_model_id = [model_ids[i] for i in 1:length(bwfl_ids) if !(bwfl_ids[i] in exclude_sensors)]
        sensor_bwfl_id = [bwfl_ids[i] for i in 1:length(bwfl_ids) if !(bwfl_ids[i] in exclude_sensors)]
        y_df = pd_2_df(y.chlorine)
        y_df = y_df[:, Symbol.(sensor_model_id)]
        y_df = y_df[burn_in + 1:end, :]
        for (old_name, new_name) in zip(Symbol.(sensor_model_id), Symbol.(sensor_bwfl_id))
            rename!(y_df, old_name => new_name)
        end
        sensor_cols = Symbol.(sensor_bwfl_id)
        y = Float64[]
        for col in sensor_cols
            append!(y, y_df[:, col])
        end
    else
        @error("Unsupported simulation type: $sim_type")
    end

    return y

end


function eki_calibration(θ_b, θ_w_lb, θ_w_ub, cl_df, wn, datetime, exclude_sensors, grouping; burn_in=24*4, δ_s=0.1, δ_b=0.025, wall_prior="uniform", n_ensemble=100, n_iter=25)

    rng = Random.seed!(Random.GLOBAL_RNG)

    # parameter data
    θ_w_1 = (θ_w_lb + θ_w_ub) ./ 2
    θ_1 = [θ_b; θ_w_1]
    y_1 = forward_model(wn_train, θ_1, grouping, datetime, exclude_sensors; sim_type="chlorine", burn_in=burn_in)
    
    θ_n = length(θ_1)
    y_n = length(y_1)
    
    # sensor data + noise
    bwfl_ids = Vector{String}(data.sensor_model_id("wq")["bwfl_id"].values)
    sensor_bwfl_id = [bwfl_ids[i] for i in 1:length(bwfl_ids) if !(bwfl_ids[i] in exclude_sensors)]
    cl_df_filter = subset(cl_df, :bwfl_id => ByRow(in(sensor_bwfl_id)), :datetime => ByRow(in(datetime_train[burn_in + 1:end])))
    vals_df = unstack(cl_df_filter, :datetime, :bwfl_id, :mean)
    sensor_cols = Symbol.(sensor_bwfl_id)
    ȳ = Float64[]
    for col in sensor_cols
        append!(ȳ, vals_df[:, col])
    end
    missing_count = sum(ismissing.(ȳ))
    missing_mask = [!ismissing(val) ? 1 : 0 for val in ȳ]
    
    δ = ȳ .* δ_s
    Γ = δ.^2 .* I(y_n)
    Σ = MvNormal(zeros(y_n), Γ)
    
    # prior distributions
    prior = constrained_gaussian("θ_b", θ_b_train, abs(θ_b * δ_b), -Inf, 0.0)
    for i = 1:θ_n-1
        if wall_prior == "uniform"
            prior_wall = constrained_gaussian("θ_w_$i", (θ_w_lb[i] + θ_w_ub[i]) ./ 2, abs((θ_w_ub[i] - θ_w_lb[i]) ./ 3.333), θ_w_lb[i], θ_w_ub[i])
            # prior_wall = ParameterDistribution(Parameterized(Uniform(θ_w_lb[i], θ_w_ub[i])), bounded(θ_w_lb[i], θ_w_ub[i]), "θ_w_$i")
            # prior_wall = ParameterDistribution(Parameterized(Uniform(θ_w_lb[i], θ_w_ub[i])), no_constraint(), "θ_w_$i")
            prior = combine_distributions([prior, prior_wall])
        else
            @error "$wall_prior distribution not available."
        end
    end
    
    # plot(prior)
    
    # set up EKI
    ensemble_0 = EKP.construct_initial_ensemble(rng, prior, n_ensemble)
    
    # create EKI process
    process = EKP.EnsembleKalmanProcess(
        ensemble_0,
        ȳ,
        Γ,
        Inversion(),
        scheduler = DataMisfitController(on_terminate="stop");
        rng=rng,
    )
    
    # run EKI iterations
    for i in 1:n_iter
        println("Iteration $i/$n_iter")
        θ_i = get_ϕ_final(prior, process)
        g_ens = hcat([forward_model(wn, θ_i[:, j], grouping, datetime, exclude_sensors; sim_type="chlorine", burn_in=burn_in) for j in 1:n_ensemble]...)
        status = EKP.update_ensemble!(process, g_ens)
        if status == true
            println("Converged at iteration $i")
            break
        end
    end
    
    # get results
    θ_final = get_ϕ_final(prior, process)

    u_init = get_u_prior(process)
    θ_init = transform_unconstrained_to_constrained(prior, u_init)
    
    return θ_init, θ_final

end
