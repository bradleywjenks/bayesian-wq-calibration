"""
This script calibrates the Field Lab's water quality model using ensemble Kalman inversion (Stage 1 of the CES method). Specifically, bulk and wall decay coefficients are calibrated based on a selected grouping. The following steps are performed:
    1. load operational data (DONE)
    2. split train/test datasets (DONE)
    3. build epanet model (DONE)
    4. set pipe grouping and define θ priors (DONE)
    5. create forward model function, F <-- wn, θ, grouping (DONE)
    6. create EKI calibration function: EKI <-- F, θ_prior (DONE)
    7. results plotting (DONE)
    8. save θ, ȳ, and L data (DONE)
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
using PGFPlotsX
using PyCall
using JLD2
using LaTeXStrings

# pgfplotsx()
# PGFPlotsX.latexengine!(PGFPlotsX.PDFLATEX)
# ENV["PGFPLOTSX_BACKEND"] = "pdf"

pd = pyimport("pandas")
np = pyimport("numpy")
data = pyimport("bayesian_wq_calibration.data")
epanet = pyimport("bayesian_wq_calibration.epanet")

const TIMESERIES_PATH = "/Users/bradwjenks/Code/PhD/bayesian-wq-calibration/data/timeseries"
const RESULTS_PATH = "/Users/bradwjenks/Code/PhD/bayesian-wq-calibration/results/"
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
p1a = plot_bwfl_data(df, ylabel, ymax=0.7)



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
p1b = plot_bwfl_data(df_filter, ylabel, ymax=0.7)



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
grouping = "material-age" # "single", "material", "material-age", "material-age-velocity"

θ_w_lb, θ_w_ub = if grouping == "single"
    ([-1.0], [0.0])  # G1: all pipes
elseif grouping == "material"
    ([-1.0, -0.5], [0.0, 0.0])  # G1: metallic, G2: cement + plastic
elseif grouping == "material-age"
    ([-1.0, -1.0, -0.5], [0.0, 0.0, 0.0])  # G1: metallic + > mean pipe age, G2: metallic + ≤ mean pipe age, G3: cement + plastic
elseif grouping == "material-age-velocity"
    ([-1.0, -1.0, -1.0, -1.0, -0.5], [0.0, 0.0, 0.0, 0.0, 0.0])  # G1: metallic + > mean pipe age + ≤ mean velocity, G2: metallic + > mean pipe age + > mean velocity, G3: metallic + ≤ mean pipe age + ≤ mean velocity, G4: metallic + ≤ mean pipe age + > mean velocity, G5: cement + plastic
else
    error("Unsupported grouping: $grouping")
end



### 5. create and test forward model F(θ) ###
θ_w = (θ_w_lb + θ_w_ub) ./ 2
θ = [θ_b_train; θ_w]
exclude_sensors = ["BW1", "BW4", "BW7"]
burn_in = 24 * 4
y = forward_model(wn_train, θ, grouping, datetime_train, exclude_sensors; sim_type="chlorine", burn_in=burn_in)



### 6. eki calibration ###

# get sensor data
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

# set noise
δ_s = 0.1
δ_b = 0.025

# eki calibration
θ_init, θ_final, stats = run_eki_calibration(θ_b_train, θ_w_lb, θ_w_ub, wn_train, datetime_train, exclude_sensors, grouping, ȳ; burn_in=burn_in, δ_s=δ_s, δ_b=δ_b)



### 7. results plotting ###
p2a, p2b, p2c = plot_eki_progress(stats; save_tex=true)
p3 = plot_parameter_distribution(θ_init, θ_final, 1, 1; save_tex=true)



### 8. save θ, ȳ, and L data ###
eki_results = summarize_eki_results(θ_final, wn_train, datetime_train, exclude_sensors, grouping, ȳ, δ_s; sim_type="chlorine", burn_in=24*4, save_results=true)



### 9. run test data ###
# insert code here...







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

    p = plot(xlabel="", ylabel=ylabel, legend=:outertopright, foreground_color_legend=nothing, ylims=(0, ymax), size=(950, 500), left_margin=8mm, bottom_margin=8mm, top_margin=8mm, xtickfont=14, ytickfont=14, xguidefont=16, yguidefont=16, legendfont=12, grid=false)

    for (i, name) in enumerate(bwfl_ids)
        color = wong_colors[mod1(i, length(wong_colors))]
        df_subset = df[df.bwfl_id .== name, :]
        plot!(p, df_subset.datetime, df_subset.mean, label=name, lw=1.5, color=color)
    end

    return p
end


function forward_model(wn, θ, grouping, datetime, exclude_sensors; sim_type="chlorine", burn_in=96)

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


function run_eki_calibration(θ_b, θ_w_lb, θ_w_ub, wn, datetime, exclude_sensors, grouping, ȳ; burn_in=24*4, δ_s=0.1, δ_b=0.025, wall_prior="uniform", n_ensemble=100, n_iter=10)


    bwfl_ids = Vector{String}(data.sensor_model_id("wq")["bwfl_id"].values)
    sensor_bwfl_id = [bwfl_ids[i] for i in 1:length(bwfl_ids) if !(bwfl_ids[i] in exclude_sensors)]
    n_sensor = length(sensor_bwfl_id)

    n_time = length(datetime[burn_in + 1:end])

    # rng = Random.seed!(Random.GLOBAL_RNG)
    rng = Random.seed!(42)

    # parameter data
    θ_w_1 = (θ_w_lb + θ_w_ub) ./ 2
    θ_1 = [θ_b; θ_w_1]
    y_1 = forward_model(wn_train, θ_1, grouping, datetime, exclude_sensors; sim_type="chlorine", burn_in=burn_in)
    
    θ_n = length(θ_1)
    y_n = length(y_1)
    
    # sensor noise
    # δ = max.((ȳ .* δ_s), 0.05)
    δ = max.((ȳ .* δ_s), 0.025)
    Γ = δ.^2 .* I(y_n)
    Σ = MvNormal(zeros(y_n), Γ)

    missing_mask = [!ismissing(val) ? 1 : 0 for val in ȳ]
    valid_indices = findall(x -> x == 1, missing_mask) 
    
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
        # scheduler = DataMisfitController(on_terminate="continue_fixed");
        scheduler = DataMisfitController(on_terminate="stop");
        rng=rng,
    )
    
    # run EKI iterations
    stats = Dict{Int, Dict{String, Any}}()
    k = 0
    for i in 1:n_iter

        println("Iteration $i/$n_iter")

        # get parameters
        stats[i] = Dict{String, Any}()
        θ_i = get_ϕ_final(prior, process)
        stats[i]["θ_mean"] = mean(θ_i, dims=2)
        stats[i]["θ_sd"] = std(θ_i, dims=2)

        # get model predictions
        g_ens = hcat([forward_model(wn, θ_i[:, j], grouping, datetime, exclude_sensors; sim_type="chlorine", burn_in=burn_in) for j in 1:n_ensemble]...)

        # compute loss function
        mean_residuals = zeros(n_ensemble, n_sensor)
        loss = Float64[]

        for j in 1:n_ensemble
            residual_j = g_ens[:, j] - ȳ
            mean_residuals[j, :] = mean(reshape(abs.(residual_j), n_time, n_sensor), dims=1)
            if length(valid_indices) > 0
                valid_residual_j = residual_j[valid_indices]
                valid_Γ = Γ[valid_indices, valid_indices]
                append!(loss, dot(valid_residual_j, valid_Γ \ valid_residual_j))
            else
                append!(loss, NaN)
            end
        end
        stats[i]["residual_mean"] = mean(mean_residuals, dims=1)
        stats[i]["loss_mean"] = mean(loss)
        stats[i]["loss_sd"] = std(loss)

        status = EKP.update_ensemble!(process, g_ens)
        # println("Status: $status")
        if status == true
            k += 1
            if k > 1
                println("Converged at iteration $i")
                break
            end
        end
    end
    
    # get results
    θ_final = get_ϕ_final(prior, process)
    u_init = get_u_prior(process)
    θ_init = transform_unconstrained_to_constrained(prior, u_init)
    
    return θ_init, θ_final, stats

end




function summarize_eki_results(θ_final, wn, datetime, exclude_sensors, grouping, ȳ, δ_s; sim_type="chlorine", burn_in=24*4, save_results=true)

    output_path = RESULTS_PATH * "/wq/eki_calibration/"

    eki_results = Dict{Int, Dict{String, Any}}()

    n_ensemble = size(θ_final, 2)

    # compute forward model and loss function for each θ sample
    for m in 1:n_ensemble

        eki_results[m] = Dict{String, Any}()
        θ_m = θ_final[:, m]
        
        eki_results[m]["θ"] = θ_m
        
        y_m = forward_model(wn, θ_m, grouping, datetime, exclude_sensors; sim_type="chlorine", burn_in=burn_in)
            
        n_sensors = length(sensor_bwfl_id)
        n_timesteps = length(datetime[burn_in + 1:end])
        if length(y_m) != n_sensors * n_timesteps
            error("Length of y ($(length(y_m))) doesn't match expected size ($(n_sensors * n_timesteps))")
        end
        y_df = DataFrame()
        y_df.datetime = repeat(datetime[burn_in + 1:end], outer=n_sensors)
        y_df.bwfl_id = repeat(sensor_bwfl_id, inner=n_timesteps)
        y_df.value = y_m
        y_df = unstack(y_df, :datetime, :bwfl_id, :value)

        eki_results[m]["y_df"] = y_df
        
        # compute negative log-likelihood
        residual = y_m - ȳ
        # δ = max.((ȳ .* δ_s), 0.05)
        δ = max.((ȳ .* δ_s), 0.025)
        Γ = (δ).^2 .* I(length(y_m))
        
        missing_mask = [!ismissing(val) ? 1 : 0 for val in ȳ]
        valid_indices = findall(x -> x == 1, missing_mask) 
        
        loss = 0
        if length(valid_indices) > 0
            valid_residual = residual[valid_indices]
            valid_Γ = Γ[valid_indices, valid_indices]
            loss = dot(valid_residual, valid_Γ \ valid_residual)
        else
            loss = NaN
        end
        eki_results[m]["loss"] = loss
        
    end

        if save_results
            filename = joinpath(output_path, "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s)).jld2")
            JLD2.save(filename, "eki_results", eki_results)
        end

        return eki_results

end


function plot_parameter_distribution(θ_initial, θ_final, param_1, param_2; save_tex=false, filename=nothing)

    output_path = RESULTS_PATH * "/wq/eki_calibration/"

    if param_1 == 1
        label_1 = L"\theta_b"
    else
        label_1 = L"\theta_w_{%$(param_1-1)}"
    end
    if param_2 == 1
        label_2 = L"\theta_b"
    else
        label_2 = L"\theta_w_{%$(param_2-1)}"
    end

    all_vals_1 = [θ_initial[param_1, :]; θ_final[param_1, :]]
    x_min = floor(minimum(all_vals_1) * 20) / 20
    x_max = ceil(maximum(all_vals_1) * 20) / 20
    x_min = fix_negative_zero(x_min)
    x_max = fix_negative_zero(x_max)

    all_vals_2 = [θ_initial[param_2, :]; θ_final[param_2, :]]
    y_min = floor(minimum(all_vals_2) * 20) / 20
    y_max = ceil(maximum(all_vals_2) * 20) / 20
    y_min = fix_negative_zero(y_min)
    y_max = fix_negative_zero(y_max)

    bin_edges = LinRange(x_min, x_max, 31)
    
    if param_1 == param_2
        # histogram
        p = histogram(θ_initial[param_1, :], bins=bin_edges, alpha=0.65, label="Initial", color=wong_colors[1], xlabel=label_1, ylabel="Frequency", legend=:topleft, linecolor=:transparent, xlims=(x_min, x_max), size=(525, 400), left_margin=2mm, right_margin=8mm, bottom_margin=2mm, top_margin=2mm, xtickfont=14, ytickfont=14, xguidefont=18, yguidefont=18, legendfont=14, foreground_color_legend=nothing, grid=false)
        histogram!(p, θ_final[param_1, :], bins=bin_edges, alpha=0.85, label="Final", color=wong_colors[2], linecolor=:transparent)

    else
        # scatter plot
        p = scatter(θ_initial[param_1, :], θ_initial[param_2, :], markersize=5, alpha=0.65, label="Initial", color=wong_colors[1], xlabel=label_1, ylabel=label_2, legend=:bottomleft, markerstrokewidth=0, xlims=(x_min, x_max), ylims=(y_min, y_max), size=(525, 400), left_margin=2mm, right_margin=8mm, bottom_margin=2mm, top_margin=2mm, xtickfont=14, ytickfont=14, xguidefont=18, yguidefont=18, legendfont=14, foreground_color_legend=nothing, grid=false)
        scatter!(p, θ_final[param_1, :], θ_final[param_2, :], markersize=5, alpha=0.85, label="Final", color=wong_colors[2], markerstrokewidth=0)
    
    end
    
    if save_tex
        if isnothing(filename)
            if param_1 == param_2
                param_name = param_1 == 1 ? "θb" : "θw_$(param_1-1)"
                filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_hist_$(param_name).tex"
            else
                param1_name = param_1 == 1 ? "θb" : "θw_$(param_1-1)"
                param2_name = param_2 == 1 ? "θb" : "θw_$(param_2-1)"
                filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_scatter_$(param1_name)_v_$(param2_name).tex"
            end
        end
        
        try
            pgfplotsx()
            
            if param_1 == param_2

                # histogram
                p_pgf = histogram(θ_initial[param_1, :], bins=bin_edges, alpha=0.65, label="Initial", color=wong_colors[1], xlabel=label_1, ylabel="Frequency", legend=:topleft, linecolor=:transparent, xlims=(x_min, x_max), size=(525, 400), left_margin=2mm, right_margin=8mm, bottom_margin=2mm, top_margin=2mm, xtickfont=14, ytickfont=14, xguidefont=18, yguidefont=18, legendfont=14, foreground_color_legend=nothing, grid=false)
                histogram!(p_pgf, θ_final[param_1, :], bins=bin_edges, alpha=0.85, label="Final", color=wong_colors[2], linecolor=:transparent)

            else

                # scatter plot
                p_pgf = scatter(θ_initial[param_1, :], θ_initial[param_2, :], markersize=5, alpha=0.65, label="Initial", color=wong_colors[1], xlabel=label_1, ylabel=label_2, legend=:bottomleft, markerstrokewidth=0, xlims=(x_min, x_max), ylims=(y_min, y_max), size=(525, 400), left_margin=2mm, right_margin=8mm, bottom_margin=2mm, top_margin=2mm, xtickfont=14, ytickfont=14, xguidefont=18, yguidefont=18, legendfont=14, foreground_color_legend=nothing, grid=false)
                scatter!(p_pgf, θ_final[param_1, :], θ_final[param_2, :], markersize=5, alpha=0.85, label="Final", color=wong_colors[2], markerstrokewidth=0)

            end
            
            savefig(p_pgf, output_path * filename)
            gr()
            println("Plot saved as $filename")

        catch e
            @warn "Failed to save as TEX file. Make sure PGFPlotsX is installed: $(e)"
            gr()

        end
    end

    return p
end


function plot_eki_progress(stats; save_tex=false, filename=nothing)

    output_path = RESULTS_PATH * "/wq/eki_calibration/"

    # get eki stats
    n = length(stats)
    θ_mean = vcat([stats[i]["θ_mean"]' for i in 1:n]...)
    θ_sd = vcat([stats[i]["θ_sd"]' for i in 1:n]...)
    residual_mean = vcat([stats[i]["residual_mean"] for i in 1:n]...)
    println(residual_mean)
    loss_mean = vcat([stats[i]["loss_mean"]' for i in 1:n]...)
    loss_sd = vcat([stats[i]["loss_sd"]' for i in 1:n]...)
    
    # get plot labels
    n_sensor = size(θ_mean, 2)
    θ_labels = String[]
    for i in 1:n_sensor
        if i == 1
            push!(θ_labels, L"\theta_b")
        else
            push!(θ_labels, L"\theta_w_{%$(i-1)}")
        end
    end

    bwfl_ids = Vector{String}(data.sensor_model_id("wq")["bwfl_id"].values)
    sensor_bwfl_id = [bwfl_ids[i] for i in 1:length(bwfl_ids) if !(bwfl_ids[i] in exclude_sensors)]

    # make plots
    p1 = plot(1:size(θ_sd, 1), θ_sd, color=reshape(wong_colors[1:size(θ_sd, 2)], 1, :), linewidth=2, label=reshape(θ_labels, 1, :), xlabel="Iteration", ylabel="Ensemble SD", legend=:topright, size=(525, 400), left_margin=2mm, right_margin=8mm, bottom_margin=2mm, top_margin=2mm, xtickfont=14, ytickfont=14, xguidefont=18, yguidefont=18, legendfont=14, foreground_color_legend=nothing, grid=false)
    display(p1)

    p2 = plot(1:size(residual_mean, 1), residual_mean, color=reshape(wong_colors[1:size(residual_mean, 2)], 1, :), linewidth=2, label=reshape(sensor_bwfl_id, 1, :), xlabel="Iteration", ylabel="Mean residual [mg/L]", legend=:topright, size=(525, 400), left_margin=2mm, right_margin=8mm, bottom_margin=2mm, top_margin=2mm, xtickfont=14, ytickfont=14, xguidefont=18, yguidefont=18, legendfont=14, foreground_color_legend=nothing, grid=false)
    display(p2)

    p3 = plot(1:size(loss_mean, 1), loss_mean, color=wong_colors[1], linewidth=2, label="", xlabel="Iteration", ylabel="Loss", legend=:topright, size=(525, 400), left_margin=2mm, right_margin=8mm, bottom_margin=2mm, top_margin=2mm, xtickfont=14, ytickfont=14, xguidefont=18, yguidefont=18, legendfont=14, foreground_color_legend=nothing, grid=false)
    display(p3)

    if save_tex
        if isnothing(filename)
            filename_p1 = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_eki_progress_1.tex"
            filename_p2 = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_eki_progress_2.tex"
            filename_p3 = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_eki_progress_3.tex"
        end
        try
            pgfplotsx()

            p1_pfg = plot(1:size(θ_sd, 1), θ_sd, color=reshape(wong_colors[1:size(θ_sd, 2)], 1, :), linewidth=2, label=reshape(θ_labels, 1, :), xlabel="Iteration", ylabel="Ensemble SD", legend=:topright, size=(525, 400), left_margin=2mm, right_margin=8mm, bottom_margin=2mm, top_margin=2mm, xtickfont=14, ytickfont=14, xguidefont=18, yguidefont=18, legendfont=14, foreground_color_legend=nothing, grid=false)
        
            p2_pfg = plot(1:size(residual_mean, 1), residual_mean, color=reshape(wong_colors[1:size(residual_mean, 2)], 1, :), linewidth=2, label=reshape(sensor_bwfl_id, 1, :), xlabel="Iteration", ylabel="Mean residual [mg/L]", legend=:topright, size=(525, 400), left_margin=2mm, right_margin=8mm, bottom_margin=2mm, top_margin=2mm, xtickfont=14, ytickfont=14, xguidefont=18, yguidefont=18, legendfont=14, foreground_color_legend=nothing, grid=false)
        
            p3_pfg = plot(1:size(loss_mean, 1), loss_mean, color=wong_colors[1], linewidth=2, label="", xlabel="Iteration", ylabel="Loss", legend=:topright, size=(525, 400), left_margin=2mm, right_margin=8mm, bottom_margin=2mm, top_margin=2mm, xtickfont=14, ytickfont=14, xguidefont=18, yguidefont=18, legendfont=14, foreground_color_legend=nothing, grid=false)

            savefig(p1_pfg, output_path * filename_p1)
            savefig(p2_pfg, output_path * filename_p2)
            savefig(p3_pfg, output_path * filename_p3)
            gr()
            println("Plots saved as $filename_p1, $filename_p2, $filename_p3")
        catch e
            @warn "Failed to save as TEX file. Make sure PGFPlotsX is installed: $(e)"
            gr()
        end
    end

    return p1, p2, p3

end


function fix_negative_zero(x)
    return x == -0.0 ? 0.0 : x
end