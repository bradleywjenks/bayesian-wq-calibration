"""
this script creates a gaussian process (gp) emulator based on the ensemble from the eki calibration.
the following steps are performed:
    1. load parameter-output pairs from eki calibration
    2. extract parameter-output pairs from eki ensemble
    3. train gp models for selected sensors
    4. validate gp model performance
    5. visualize results
"""

using Revise
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
using JLD2
using LaTeXStrings
using ScikitLearn

@sk_import preprocessing: StandardScaler
@sk_import gaussian_process: GaussianProcessRegressor
@sk_import gaussian_process.kernels: (RBF, Matern, RationalQuadratic, ConstantKernel)
@sk_import metrics: (mean_squared_error, mean_absolute_error, r2_score)
@sk_import multioutput: MultiOutputRegressor

const TIMESERIES_PATH = "/Users/bradwjenks/Code/PhD/bayesian-wq-calibration/data/timeseries"
const RESULTS_PATH = "/Users/bradwjenks/Code/PhD/bayesian-wq-calibration/results/"

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

### 1. load eki calibration results ###
data_period = 18 # (aug. 2024)
grouping = "material" # "single", "material", "material-age", "material-age-velocity"
δ_s = 0.2
δ_b = 0.025

eki_results_path = RESULTS_PATH * "/wq/eki_calibration/"
eki_filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s)).jld2"
eki_results = JLD2.load(eki_results_path * eki_filename, "eki_results")

bwfl_ids = [string(col) for col in propertynames(eki_results[1]["y_df"]) if col != :datetime]
selected_sensor = bwfl_ids[2]



### 2. extract parameter-output pairs from eki ensemble ###
n_ensemble = length(eki_results)
θ_samples = hcat([eki_results[i]["θ"] for i in 1:n_ensemble]...)'
y = get_sensor_y(eki_results, selected_sensor)



### 3. train gp models for selected sensor ###
n_samples = size(θ_samples, 1)
train_ratio = 0.8
n_train = round(Int, train_ratio * n_samples)

rng = Random.seed!(42)
train_idx = randperm(n_samples)[1:n_train]
test_idx = setdiff(1:n_samples, train_idx)

x_train = θ_samples[train_idx, :]
y_train = y[train_idx, :]
x_test = θ_samples[test_idx, :]
y_test = y[test_idx, :]

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

kernel_type = "RBF" # "RBF", "Matern", "RationalQuadratic"

kernel = if kernel_type == "RBF"
    ConstantKernel() * RBF()
elseif kernel_type == "Matern"
    ConstantKernel() * Matern(nu=1.5)
elseif kernel_type == "RationalQuadratic"
    ConstantKernel() * RationalQuadratic(alpha=1.0)
else
    @error "invalid kernel type: $kernel_type"
end

# train gp model
base_model = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    random_state=42
)
gp_model = MultiOutputRegressor(base_model)
gp_model.fit(x_train_scaled, y_train)

# save gp model
output_path = RESULTS_PATH * "/wq/gp_models/"
filename = joinpath(output_path, "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_sensor_$(selected_sensor).jld2")
model_save = Dict(
    "gp_model" => gp_model,
    "scaler" => scaler
)
JLD2.save(filename, "gp_model", model_save)

constant_values, length_scale = gp_model_stats(gp_model)




### 4. validate gp model performance ###
y_pred = gp_model.predict(x_test_scaled)
y_std = zeros(size(y_test))
for i in 1:length(test_idx)
    for t in 1:size(y_test, 2)
        estimator = gp_model.estimators_[t]
        _, std = estimator.predict(x_test_scaled[i:i, :], return_std=true)
        y_std[i, t] = std[1]
    end
end

rmse = sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
maxae = maximum(abs.(y_test - y_pred))
r2 = 1 - sum((y_test - y_pred) .^ 2) / sum((y_test .- mean(y_test)) .^ 2)



### 5. visualize results ###
p1 = plot_parity(y_test, y_pred)

if !isempty(test_idx)
    p2 = plot_time_series(y_test, y_pred, y_std, n_samples=5)
end

p3 = plot_error_histogram(y_test, y_pred)







########## FUNCTIONS ##########

function get_sensor_y(eki_results, sensor_id)
    n_ensemble = length(eki_results)
    
    y_df = eki_results[1]["y_df"]
    n_timesteps = size(y_df, 1)
    
    outputs = zeros(n_ensemble, n_timesteps)
    
    for i in 1:n_ensemble
        y_df = eki_results[i]["y_df"]
        if Symbol(sensor_id) in propertynames(y_df)
            outputs[i, :] = y_df[!, Symbol(sensor_id)]
        else
            @warn "sensor $sensor_id not found in ensemble member $i"
        end
    end
    
    return outputs
end



function gp_model_stats(gp_model)
    
    constant_values = Float64[]
    length_scales = Float64[]
    
    for estimator in gp_model.estimators_
        kernel_params = estimator.kernel_.get_params()
        push!(constant_values, kernel_params["k1__constant_value"])
    
        length_scale = kernel_params["k2__length_scale"]
        if length(length_scale) > 1
            append!(length_scales, length_scale)
        else
            push!(length_scales, length_scale)
        end
    end
    
    # print statistics
    println("\nKernel parameter statistics:")
    println("Constant value (k1): mean = $(round(mean(constant_values), digits=4)), " *
            "std = $(round(std(constant_values), digits=4)), " *
            "min = $(round(minimum(constant_values), digits=4)), " *
            "max = $(round(maximum(constant_values), digits=4))")
    println("Length scale (k2): mean = $(round(mean(length_scales), digits=4)), " *
            "std = $(round(std(length_scales), digits=4)), " *
            "min = $(round(minimum(length_scales), digits=4)), " *
            "max = $(round(maximum(length_scales), digits=4))")
    
    return constant_values, length_scales
end



function plot_parity(y_true, y_pred)

    y_true_flat = vec(y_true)
    y_pred_flat = vec(y_pred)

    max_val = ceil(max(maximum(y_true_flat), maximum(y_pred_flat)) * 10) / 10

    p = scatter(y_true_flat, y_pred_flat,
        xlabel="EPANET",
        ylabel="GP",
        title=selected_sensor,
        legend=false,
        markersize=6,
        markercolor=wong_colors[3],
        markerstrokewidth=0,
        markeralpha=0.6,
        # aspect_ratio=:equal,
        size=(450, 400),
        left_margin=6mm,
        right_margin=6mm,
        bottom_margin=6mm,
        top_margin=6mm,
        xtickfont=14,
        ytickfont=14,
        xguidefont=16,
        yguidefont=16,
        titlefont=16,
        grid=false,
        xlims=(0, max_val),
        ylims=(0, max_val),
        )
    
    plot!(p, [0.0, max_val], [0.0, max_val],
          linestyle=:dash,
          color=:black,
          linewidth=2)
    
    return p
end


function plot_time_series(y_true, y_pred, y_std; n_samples=3)
    n_to_plot = min(n_samples, size(y_true, 1))
    sample_indices = 1:n_to_plot
    
    p = plot(
        xlabel="Time step",
        ylabel="Chlorine [mg/L]",
        title=selected_sensor,
        size=(750, 400),
        left_margin=6mm,
        right_margin=6mm,
        bottom_margin=6mm,
        top_margin=6mm,
        xtickfont=14,
        ytickfont=14,
        xguidefont=16,
        yguidefont=16,
        titlefont=16,
        grid=false,
        legend=:outertopright,
        legendfont=14,
        foreground_color_legend=nothing
    )
    
    for (i, idx) in enumerate(sample_indices)
        color_idx = mod1(i, length(wong_colors))
        plot!(p, 1:size(y_true, 2), y_true[idx, :],
            label="EPANET-$(idx)",
            linewidth=2,
            color=wong_colors[color_idx]
        )
        plot!(p, 1:size(y_pred, 2), y_pred[idx, :],
            ribbon=1.96*y_std[idx, :],
            label="GP-$(idx)",
            linewidth=2,
            linestyle=:dash,
            color=wong_colors[color_idx],
            fillalpha=0.15
        )
    end
    
    return p
end

function plot_error_histogram(y_true, y_pred)
    errors = vec(y_true - y_pred)
    
    p = histogram(errors,
                 bins=30,
                 xlabel="Error [mg/L]",
                 ylabel="Frequency",
                 title="$selected_sensor: Error Distribution",
                 legend=false,
                 alpha=0.7,
                 color=wong_colors[3],
                 size=(525, 400),
                 left_margin=2mm,
                 right_margin=8mm,
                 bottom_margin=2mm,
                 top_margin=2mm,
                 xtickfont=14,
                 ytickfont=14,
                 xguidefont=16,
                 yguidefont=16)
    
    return p
end