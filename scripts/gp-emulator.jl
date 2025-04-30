"""
This script creates a gaussian process (gp) emulator based on the final ensemble from the eki calibration stage. The following steps are performed:
    1. load parameter-output pairs from eki calibration (Done)
    2. extract parameter-output pairs from eki ensemble (Done)
    3. train and test gp model for selected sensors using k-folds cross validation (DONE)
    4. test on expanded θ samples (DONE)
    5. visualize results (DONE)
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
using PyCall

py"""
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
"""

pd = pyimport("pandas")
np = pyimport("numpy")
pickle = pyimport("pickle")
data = pyimport("bayesian_wq_calibration.data")
epanet = pyimport("bayesian_wq_calibration.epanet")


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

### 1. load eki calibration results and operational data ###
data_period = 18 # (aug. 2024)
padded_period = lpad(data_period, 2, "0")
grouping = "material" # "single", "material", "material-age", "material-age-velocity"
δ_s = 0.2
δ_b = 0.025

# eki results
eki_results_path = RESULTS_PATH * "/wq/eki_calibration/"
eki_filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s)).jld2"
eki_results = JLD2.load(eki_results_path * eki_filename, "eki_results")

bwfl_ids = [string(col) for col in propertynames(eki_results[1]["y_df"]) if col != :datetime]
selected_sensor = bwfl_ids[6]

# operational data
flow_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-flow.csv", DataFrame); flow_df.datetime = DateTime.(flow_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
pressure_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-pressure.csv", DataFrame); pressure_df.datetime = DateTime.(pressure_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
wq_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-wq.csv", DataFrame); wq_df.datetime = DateTime.(wq_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
cl_df = wq_df[wq_df.data_type .== "chlorine", :]

unique_datetime = DateTime.(unique(cl_df.datetime))
dataset_size = length(eki_results[1]["y_df"].datetime)
datetime_select = unique_datetime[1:dataset_size + (24 * 4)] # train data size + 24-hour burn-in



### 2. extract parameter-output pairs from eki ensemble ###
n_ensemble = length(eki_results)
θ_samples = hcat([eki_results[i]["θ"] for i in 1:n_ensemble]...)'
y = get_sensor_y(eki_results, selected_sensor)



### 3. train and test gp model for selected sensor ###
x_valid_results = nothing
x_valid_results = gp_model_x_valid(θ_samples, y, grouping, selected_sensor; kernel_type="RBF", save=true, cv_folds=5)



### 4. test GP on expanded θ samples ###
wn = epanet.build_model(
    df_2_pd(filter(row -> row.datetime ∈ datetime_select, flow_df)), 
    df_2_pd(filter(row -> row.datetime ∈ datetime_select, pressure_df)), 
    df_2_pd(filter(row -> row.datetime ∈ datetime_select, cl_df)),
    sim_type="chlorine"
)
exclude_sensors = ["BW1", "BW4", "BW7"]
burn_in = 24 * 4
n_expand = 100
lhs_dist = "uniform"

test_results = test_expanded_θ(x_valid_results["gp_model"], x_valid_results["scaler"], n_expand, θ_samples, y, grouping, selected_sensor, wn, datetime_select; exclude_sensors=exclude_sensors, burn_in=burn_in, lhs_dist=lhs_dist)



### 5. visualize results ###
x_test = test_results["x_test"]
y_test = test_results["y_test"]
y_pred_μ = test_results["y_pred_μ"]
y_pred_σ = test_results["y_pred_σ"]

filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_$(selected_sensor)"

θ_n = 3
save_tex = true
begin
    p0 = histogram(x_test[:, θ_n], alpha=0.5, color=wong_colors[2], label="Extended θ", xlabel="θ", ylabel="Frequency", grid=false, size=(750, 400), left_margin=4mm, right_margin=8mm, bottom_margin=4mm, top_margin=4mm, xtickfont=12, ytickfont=12, xguidefont=14, yguidefont=14, legend=:outertopright, legendfont=12, foreground_color_legend=nothing)
    p0 = histogram!(θ_samples[:, θ_n], alpha=0.5, color=wong_colors[3], label="EKI θ")
end
p1 = plot_parity(y_test, y_pred_μ, filename, save_tex=save_tex)
p2 = plot_time_series(y_test, y_pred_μ, y_pred_σ, filename, n_samples=3, save_tex=save_tex)
p3 = plot_error_histogram(y_test, y_pred_μ, filename, save_tex=save_tex)







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



function plot_parity(y_true, y_pred, filename; save_tex=false)

    y_true_flat = vec(y_true)
    y_pred_flat = vec(y_pred)

    max_val = ceil(max(maximum(y_true_flat), maximum(y_pred_flat)) * 10) / 10

    p = scatter(y_true_flat, y_pred_flat,
        xlabel="EPANET",
        ylabel="GP",
        title=selected_sensor,
        legend=false,
        markersize=5,
        markercolor=wong_colors[3],
        markerstrokewidth=0,
        markeralpha=0.7,
        # aspect_ratio=:equal,
        size=(450, 400),
        left_margin=6mm,
        right_margin=6mm,
        bottom_margin=6mm,
        top_margin=6mm,
        xtickfont=12,
        ytickfont=12,
        xguidefont=14,
        yguidefont=14,
        titlefont=14,
        grid=false,
        xlims=(0, max_val),
        ylims=(0, max_val),
    )
    
    plot!(p, [0.0, max_val], [0.0, max_val],
          linestyle=:dash,
          color=:black,
          linewidth=2)
    
    if save_tex
        try
            pgfplotsx()
            
            p_pgf = scatter(y_true_flat, y_pred_flat,
                xlabel="EPANET",
                ylabel="GP",
                title=selected_sensor,
                legend=false,
                markersize=5,
                markercolor=wong_colors[3],
                markerstrokewidth=0,
                markeralpha=0.7,
                size=(450, 400),
                left_margin=6mm,
                right_margin=6mm,
                bottom_margin=6mm,
                top_margin=6mm,
                xtickfont=12,
                ytickfont=12,
                xguidefont=14,
                yguidefont=14,
                titlefont=14,
                grid=false,
                xlims=(0, max_val),
                ylims=(0, max_val),
            )
            
            plot!(p_pgf, [0.0, max_val], [0.0, max_val],
                  linestyle=:dash,
                  color=:black,
                  linewidth=2)
            
            output_path = RESULTS_PATH * "/wq/gp_models/"
            savefig(p_pgf, output_path * filename * "_parity.tex")
            gr()
            println("Plot saved as $filename")
            
        catch e
            @warn "Failed to save as TEX file. Make sure PGFPlotsX is installed: $(e)"
            gr()
        end
    end
    
    return p
end



function plot_time_series(y_true, y_pred, y_std, filename; n_samples=3, save_tex=false)

    n_to_plot = min(n_samples, size(y_true, 1))
    sample_indices = 1:n_to_plot
    
    p = plot(
        xlabel="Time step",
        ylabel="Chlorine [mg/L]",
        title=selected_sensor,
        size=(800, 400),
        left_margin=6mm,
        right_margin=6mm,
        bottom_margin=6mm,
        top_margin=6mm,
        xtickfont=12,
        ytickfont=12,
        xguidefont=14,
        yguidefont=14,
        titlefont=14,
        grid=false,
        legend=:outertopright,
        legendfont=12,
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
    
    if save_tex       
        try
            pgfplotsx()
            
            p_pgf = plot(
                xlabel="Time step",
                ylabel="Chlorine [mg/L]",
                title=selected_sensor,
                size=(800, 400),
                left_margin=6mm,
                right_margin=6mm,
                bottom_margin=6mm,
                top_margin=6mm,
                xtickfont=12,
                ytickfont=12,
                xguidefont=14,
                yguidefont=14,
                titlefont=14,
                grid=false,
                legend=:outertopright,
                legendfont=12,
                foreground_color_legend=nothing
            )
            
            for (i, idx) in enumerate(sample_indices)
                color_idx = mod1(i, length(wong_colors))
                plot!(p_pgf, 1:size(y_true, 2), y_true[idx, :],
                    label="EPANET-$(idx)",
                    linewidth=2,
                    color=wong_colors[color_idx]
                )
                plot!(p_pgf, 1:size(y_pred, 2), y_pred[idx, :],
                    ribbon=1.96*y_std[idx, :],
                    label="GP-$(idx)",
                    linewidth=2,
                    linestyle=:dash,
                    color=wong_colors[color_idx],
                    fillalpha=0.15
                )
            end
            
            output_path = RESULTS_PATH * "/wq/gp_models/"
            savefig(p_pgf, output_path * filename * "_timeseries.tex")
            gr()
            println("Plot saved as $filename")
            
        catch e
            @warn "Failed to save as TEX file. Make sure PGFPlotsX is installed: $(e)"
            gr()
        end
    end
    
    return p
end



function plot_error_histogram(y_true, y_pred, filename; save_tex=false)

    errors = vec(y_true - y_pred)
    
    p = histogram(errors,
        bins=30,
        xlabel="Error [mg/L]",
        ylabel="Frequency",
        title=selected_sensor,
        legend=false,
        alpha=0.7,
        color=wong_colors[3],
        size=(525, 400),
        left_margin=2mm,
        right_margin=8mm,
        bottom_margin=2mm,
        top_margin=2mm,
        xtickfont=12,
        ytickfont=12,
        xguidefont=14,
        yguidefont=14
    )
    
    if save_tex 
        try
            pgfplotsx()
            
            p_pgf = histogram(errors,
                bins=30,
                xlabel="Error [mg/L]",
                ylabel="Frequency",
                title=selected_sensor,
                legend=false,
                alpha=0.7,
                color=wong_colors[3],
                size=(525, 400),
                left_margin=2mm,
                right_margin=8mm,
                bottom_margin=2mm,
                top_margin=2mm,
                xtickfont=12,
                ytickfont=12,
                xguidefont=14,
                yguidefont=14
            )
            
            output_path = RESULTS_PATH * "/wq/gp_models/"
            savefig(p_pgf, output_path * filename * "_error_hist.tex")
            gr()
            println("Plot saved as $filename")
            
        catch e
            @warn "Failed to save as TEX file. Make sure PGFPlotsX is installed: $(e)"
            gr()
        end
    end
    
    return p
end




function gp_model_x_valid(θ_samples, y, grouping, selected_sensor; kernel_type="RBF", save=true, cv_folds=5)

    n_samples = size(θ_samples, 1)
    n_params = size(θ_samples, 2)
    n_outputs = size(y, 2)

    # prepare containers for cross-validation results
    all_models = []
    all_scalers = []
    all_train_stats = []
    all_test_stats = []
    all_preds = []
    all_stds = []
    all_test_x = []
    all_test_y = []

    # generate indices for cross-validation
    Random.seed!(42)
    shuffled_indices = randperm(n_samples)
    fold_size = floor(Int, n_samples / cv_folds)

    # perform cross-validation
    for fold in 1:cv_folds

        println("Processing fold $fold of $cv_folds")

        # determine test indices for this fold
        start_idx = (fold - 1) * fold_size + 1
        end_idx = min(fold * fold_size, n_samples)
        test_indices = shuffled_indices[start_idx:end_idx]
        train_indices = setdiff(shuffled_indices, test_indices)

        # split data
        x_train = θ_samples[train_indices, :]
        y_train = y[train_indices, :]
        x_test = θ_samples[test_indices, :]
        y_test = y[test_indices, :]

        # standardize features
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # select kernel
        kernel = if kernel_type == "RBF"
            ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=ones(n_params), length_scale_bounds=(1e-3, 1e3))
        elseif kernel_type == "Matern"
            ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * Matern(length_scale=ones(n_params), nu=1.5)
        else
            @error "invalid kernel type: $kernel_type"
        end

        # train GP model
        base_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=42,
            normalize_y=true
        )
        gp_model = MultiOutputRegressor(base_model)
        gp_model.fit(x_train_scaled, y_train)

        # train set performance
        y_train_pred = gp_model.predict(x_train_scaled)
        train_stats = Dict(
            "rmse" => sqrt(mean_squared_error(y_train, y_train_pred)),
            "mae" => mean_absolute_error(y_train, y_train_pred),
            "maxae" => maximum(abs.(y_train - y_train_pred)),
            "r2" => r2_score(y_train, y_train_pred)
        )

        # test set performance
        y_test_pred = gp_model.predict(x_test_scaled)
        y_test_std = zeros(size(y_test))

        for i in 1:size(x_test, 1)
            for t in 1:n_outputs
                estimator = gp_model.estimators_[t]
                _, std = estimator.predict(x_test_scaled[i:i, :], return_std=true)
                y_test_std[i, t] = std[1]
            end
        end

        test_stats = Dict(
            "rmse" => sqrt(mean_squared_error(y_test, y_test_pred)),
            "mae" => mean_absolute_error(y_test, y_test_pred),
            "maxae" => maximum(abs.(y_test - y_test_pred)),
            "r2" => r2_score(y_test, y_test_pred)
        )

        # store results for this fold
        push!(all_models, gp_model)
        push!(all_scalers, scaler)
        push!(all_train_stats, train_stats)
        push!(all_test_stats, test_stats)
        push!(all_preds, y_test_pred)
        push!(all_stds, y_test_std)
        push!(all_test_x, x_test)
        push!(all_test_y, y_test)
    end

    # summarize cross-validation results
    println("\nCross-validation summary ($cv_folds folds):")

    # training stats
    train_rmse = [stats["rmse"] for stats in all_train_stats]
    train_mae = [stats["mae"] for stats in all_train_stats]
    train_r2 = [stats["r2"] for stats in all_train_stats]

    # test stats
    test_rmse = [stats["rmse"] for stats in all_test_stats]
    test_mae = [stats["mae"] for stats in all_test_stats]
    test_r2 = [stats["r2"] for stats in all_test_stats]

    println("Test RMSE: mean = $(round(mean(test_rmse), digits=4)), std = $(round(std(test_rmse), digits=4))")
    println("Test MAE: mean = $(round(mean(test_mae), digits=4)), std = $(round(std(test_mae), digits=4))")
    println("Test R2: mean = $(round(mean(test_r2), digits=4)), std = $(round(std(test_r2), digits=4))")

    # find best model based on test rmse
    best_idx = argmin([stats["rmse"] for stats in all_test_stats])
    best_model = all_models[best_idx]
    best_scaler = all_scalers[best_idx]
    best_test_stats = all_test_stats[best_idx]
    best_y_pred = all_preds[best_idx]
    best_y_std = all_stds[best_idx]
    best_test_x = all_test_x[best_idx]
    best_test_y = all_test_y[best_idx]

    println("\nBest model (fold $(best_idx)):")
    println("Test RMSE: $(round(best_test_stats["rmse"], digits=4))")
    println("Test MAE: $(round(best_test_stats["mae"], digits=4))")
    println("Test R2: $(round(best_test_stats["r2"], digits=4))")

    # calculate kernel parameters for the best model
    kernel_params = []
    for estimator in best_model.estimators_
        push!(kernel_params, estimator.kernel_.get_params())
    end

    # create comprehensive results dictionary
    results_dict = Dict(
        "gp_model" => best_model,
        "scaler" => best_scaler,
        "y_pred_μ" => best_y_pred,
        "y_pred_σ" => best_y_std,
        "x_test" => best_test_x,
        "y_test" => best_test_y,
        "cv_results" => Dict(
            "models" => all_models,
            "scalers" => all_scalers,
            "train_stats" => all_train_stats,
            "test_stats" => all_test_stats,
            "test_x" => all_test_x,
            "test_y" => all_test_y,
            "best_idx" => best_idx
        ),
        "kernel_params" => kernel_params
    )

    # save results
    if save
        save_gp_model(results_dict["gp_model"], results_dict["scaler"], results_dict)
    end

    return results_dict
end



function test_expanded_θ(gp_model, scaler, n_expand, θ_samples, y, grouping, selected_sensor, wn, datetime_select; exclude_sensors=exclude_sensors, burn_in=burn_in, lhs_dist="uniform")

    n_output = size(y, 2)
    θ_means = vec(mean(θ_samples, dims=1))
    θ_stds = vec(std(θ_samples, dims=1))

    # sample extended θ
    x_test = latin_hypercube_sampling(θ_means, θ_stds, n_expand, dist_type=lhs_dist)
    x_test_scaled = scaler.transform(x_test)

    # run forward model
    y_test = zeros(n_expand, n_output)
    for i in 1:n_expand
        y_df = forward_model(wn, x_test[i, :], grouping, datetime_select, exclude_sensors; sim_type="chlorine", burn_in=burn_in)
        y_test[i,:] = y_df[!, Symbol(selected_sensor)]
    end
    
    # run GP model
    y_pred_μ = gp_model.predict(x_test_scaled)
    y_pred_σ = zeros(size(y_test))

    for i in 1:size(x_test, 1)
        for t in 1:n_output
            estimator = gp_model.estimators_[t]
            _, std = estimator.predict(x_test_scaled[i:i, :], return_std=true)
            y_pred_σ[i, t] = std[1]
        end
    end

    results_dict = Dict(
        "x_test" => x_test,
        "y_test" => y_test,
        "y_pred_μ" => y_pred_μ,
        "y_pred_σ" => y_pred_σ,
        "metrics" => Dict(
            "rmse" => sqrt(mean_squared_error(y_test, y_pred_μ)),
            "mae" => mean_absolute_error(y_test, y_pred_μ),
            "maxae" => maximum(abs.(y_test - y_pred_μ)),
            "r2" => r2_score(y_test, y_pred_μ)
        )
    )

    return results_dict


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

    return y_df

end




function latin_hypercube_sampling(θ_means, θ_stds, n_expand; dist_type="uniform")

    n_params = length(θ_means)
    result = zeros(n_expand, n_params)

    for j in 1:n_params

        ε = 1e-10
        u = [(i - 1 + rand() * (1 - 2ε)) / n_expand + ε for i in 1:n_expand]
        shuffle!(u)

        if dist_type == "normal"
            d = Normal(θ_means[j], θ_stds[j])
            samples = quantile.(d, u)
        elseif dist_type == "uniform"
            a = θ_means[j] - 3 * θ_stds[j]
            b = θ_means[j] + 3 * θ_stds[j]
            samples = a .+ u .* (b - a)
        else
            error("Unsupported distribution type: \"$dist_type\". Use \"normal\" or \"uniform\".")
        end

        result[:, j] = clamp.(samples, -Inf, -1e-5)
    end

    return result
end



function save_gp_model(model, scaler, results_dict)

    output_path = RESULTS_PATH * "wq/gp_models/"
    base_filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_$(selected_sensor)"

    # pickle files
    open(output_path * base_filename * "_model.pkl", "w") do f
        pickle.dump(model, f)
    end
    open(output_path * base_filename * "_scaler.pkl", "w") do f
        pickle.dump(scaler, f)
    end
    
    # JLD2 files
    JLD2.save(output_path * base_filename * ".jld2", "gp_results", results_dict)

    println("Saved GP model for $selected_sensor using pickle and JLD2.")

end


