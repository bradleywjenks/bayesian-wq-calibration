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
using LatinHypercubeSampling
using Surrogates
using MLJ
using PyCall
using ProgressMeter


pd = pyimport("pandas")
np = pyimport("numpy")
data = pyimport("bayesian_wq_calibration.data")
epanet = pyimport("bayesian_wq_calibration.epanet")

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


# NB: MUST RUN FUNCTIONS BLOCK BEFORE EXECUTING MAIN SCRIPT #


########## MAIN SCRIPT ##########

### 1. load eki calibration results and operational data ###
data_period = 18 # (aug. 2024)
padded_period = lpad(data_period, 2, "0")
grouping = "single" # "single", "material", "material-age", "material-age-velocity"
δ_s = 0.2 # 0.1, 0.2
δ_b = 0.05

# eki results
eki_results_path = RESULTS_PATH * "/wq/eki_calibration/"
eki_filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s)).jld2"
eki_results = JLD2.load(eki_results_path * eki_filename, "eki_results")


# operational data
flow_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-flow.csv", DataFrame); flow_df.datetime = DateTime.(flow_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
pressure_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-pressure.csv", DataFrame); pressure_df.datetime = DateTime.(pressure_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
wq_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-wq.csv", DataFrame); wq_df.datetime = DateTime.(wq_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
cl_df = wq_df[wq_df.data_type .== "chlorine", :]

unique_datetime = DateTime.(unique(cl_df.datetime))
dataset_size = length(eki_results[1]["y_df"].datetime)
datetime_select = unique_datetime[1:dataset_size + (24 * 4)] # train data size + 24-hour burn-in



### 2. extract parameter-output pairs from eki ensemble ###

# create additional training data
wn = epanet.build_model(
    df_2_pd(filter(row -> row.datetime ∈ datetime_select, flow_df)), 
    df_2_pd(filter(row -> row.datetime ∈ datetime_select, pressure_df)), 
    df_2_pd(filter(row -> row.datetime ∈ datetime_select, cl_df)),
    sim_type="chlorine"
)
exclude_sensors = ["BW1", "BW4", "BW7"]
burn_in = 24 * 4
n_output = length(datetime_select) - burn_in
bwfl_ids = [string(col) for col in propertynames(eki_results[1]["y_df"]) if col != :datetime]
n_ensemble = length(eki_results)
θ_samples = hcat([eki_results[i]["θ"] for i in 1:n_ensemble]...)'


# # Filter outliers only on the lower bound using IQR method
# param_data = θ_samples[:, 4]
# Q1, Q3 = quantile(param_data, [0.25, 0.75])
# IQR = Q3 - Q1
# filtered_data = param_data[param_data .>= Q1 - 1.5*IQR]
# μ, σ = mean(filtered_data), std(filtered_data)
# fitted_dist = Truncated(Normal(μ, σ), -Inf, 0.0)
# x_range = range(minimum(filtered_data), 0, length=100)
# histogram(filtered_data, normalize=:pdf, alpha=0.6, label="Filtered Data")
# plot!(x_range, pdf.(fitted_dist, x_range), linewidth=2, label="Fitted Truncated Normal")



# get final eki ensemble data
sensor = bwfl_ids[1]
y = get_sensor_y(eki_results, sensor)


### 3. train and test gp model for selected sensor ###
x_valid_results = nothing
x_valid_results =  gp_model_x_valid(θ_samples, y, grouping, sensor; save=true, cv_folds=5)



### 4. test GP on new θ samples ###
n_test = 10
test_results = test_gp_model(x_valid_results["gp_model"], x_valid_results["scaler"], n_test, θ_samples, y, grouping, sensor, wn, datetime_select; exclude_sensors=exclude_sensors, burn_in=burn_in, save_results=true)



### 5. visualize results ###
x_test = test_results["x_test"]
y_test = test_results["y_test"]
y_pred_μ = test_results["y_pred_μ"]
filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_$(sensor)"

save_tex = true
p1 = plot_parity(y_test, y_pred_μ, filename, save_tex=save_tex)
p2 = plot_error_histogram(y_test, y_pred_μ, filename, save_tex=save_tex)







########## FUNCTIONS ##########

begin

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
            title=sensor,
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
                    title=sensor,
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
            title=sensor,
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
                    title=sensor,
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
            title=sensor,
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
                    title=sensor,
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




    function gp_model_x_valid(θ_samples, y, grouping, sensor; save=true, cv_folds=5)

        n_samples = size(θ_samples, 1)
        n_params = size(θ_samples, 2)
        n_outputs = size(y, 2)

        all_models = Vector{Vector{Any}}()
        all_scalers = Vector{Machine{Standardizer}}()
        all_train_stats = Vector{Dict{String, Float64}}()
        all_test_stats = Vector{Dict{String, Float64}}()
        all_preds = Vector{Matrix{Float64}}()
        all_test_x = Vector{Matrix{Float64}}()
        all_test_y = Vector{Matrix{Float64}}()

        Random.seed!(42)
        shuffled_indices = randperm(n_samples)
        fold_size = floor(Int, n_samples / cv_folds)

        # Perform cross-validation
        for fold in 1:cv_folds

            println("Processing fold $fold of $cv_folds")

            # determine train/test indices for this fold
            start_idx = (fold - 1) * fold_size + 1
            end_idx = min(fold * fold_size, n_samples)
            test_indices = shuffled_indices[start_idx:end_idx]
            train_indices = setdiff(shuffled_indices, test_indices)

            x_train = θ_samples[train_indices, :]
            y_train = y[train_indices, :]
            x_test = θ_samples[test_indices, :]
            y_test = y[test_indices, :]

            # standardize features using MLJ's Standardizer
            x_train_df = DataFrame(x_train, :auto)
            scaler_model = Standardizer()
            scaler = machine(scaler_model, x_train_df)
            fit!(scaler)

            x_train_scaled = Matrix(MLJ.transform(scaler, x_train_df))
            x_test_scaled = Matrix(MLJ.transform(scaler, DataFrame(x_test, :auto)))

            x_train_scaled_vec = [x_train_scaled[i, :] for i in 1:size(x_train_scaled, 1)]
            x_test_scaled_vec = [x_test_scaled[i, :] for i in 1:size(x_test_scaled, 1)]

            lower_bounds = vec(minimum(x_train_scaled, dims=1))
            upper_bounds = vec(maximum(x_train_scaled, dims=1))

            # train GP model for each output dimension
            gp_surrogates = Vector{Any}(undef, n_outputs)
            println("Training $(n_outputs) GP emulators...")
            for t in 1:n_outputs
                y_output_t = y_train[:, t]
                # surrogate = RadialBasis(x_train_scaled_vec, y_output_t, lower_bounds, upper_bounds)
                surrogate = Kriging(x_train_scaled_vec, y_output_t, lower_bounds, upper_bounds)
                gp_surrogates[t] = surrogate
            end
            println("Training complete for fold $fold.")

            # predict train and test datasets using helper function
            y_train_pred = predict_multi_output(gp_surrogates, x_train_scaled_vec, n_outputs)
            train_stats = calculate_metrics(y_train, y_train_pred)
            y_test_pred = predict_multi_output(gp_surrogates, x_test_scaled_vec, n_outputs)
            test_stats = calculate_metrics(y_test, y_test_pred)

            # store results for this fold
            push!(all_models, gp_surrogates)
            push!(all_scalers, scaler)
            push!(all_train_stats, train_stats)
            push!(all_test_stats, test_stats)
            push!(all_preds, y_test_pred)
            push!(all_test_x, x_test)
            push!(all_test_y, y_test)
        end

        # summarize cross-validation results
        println("\nCross-validation summary ($cv_folds folds):")
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
        best_test_x = all_test_x[best_idx]
        best_test_y = all_test_y[best_idx]

        # create comprehensive results dictionary
        results_dict = Dict(
            "gp_model" => best_model,
            "scaler" => best_scaler,
            "y_pred_μ" => best_y_pred,
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
            )
        )

        # save GP emulators
        if save
            output_path = joinpath(RESULTS_PATH, "wq/gp_models")
            filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_$(sensor)_gp.jld2"
            filepath = joinpath(output_path, filename)
            model_to_save = (
                surrogates = best_model,
                scaler = best_scaler
            )
            jldsave(filepath; model_to_save...)
            println("Best GP model saved to: $filepath")
        end

        return results_dict
    end


    function predict_multi_output(gp_surrogates, x_scaled_vec, n_outputs)
        n_samples_pred = length(x_scaled_vec)
        y_pred = zeros(n_samples_pred, n_outputs)
        for i in 1:n_samples_pred
            x_sample_scaled = x_scaled_vec[i]
            for t in 1:n_outputs
                y_pred[i, t] = gp_surrogates[t](x_sample_scaled)
            end
        end
        return y_pred
    end


    function generate_expanded_training_data(n_samples, θ_samples_eki, n_output, grouping, wn, datetime_select, bwfl_ids; exclude_sensors=nothing, burn_in=96)

        n_params = size(θ_samples_eki, 2)
        θ_lbs = [quantile(θ_samples_eki[:, j], 0.05) for j in 1:n_params]
        θ_ubs = [quantile(θ_samples_eki[:, j], 0.95) for j in 1:n_params]

        x_expanded_samples = zeros(n_samples, n_params)
        y_expanded_outputs = Matrix{Float64}(undef, n_samples, n_output)

        # param 1: Sample directly from EKI ensemble distribution
        x_expanded_samples[:, 1] = rand(θ_samples_eki[:, 1], n_samples)

        expanded_min_bounds = zeros(n_params - 1)
        expanded_max_bounds = zeros(n_params - 1)

        for j in 2:n_params
            lt_raw = θ_lbs[j]
            ut_raw = θ_ubs[j]
            lt_r = floor(lt_raw / 0.05) * 0.05
            ut_r = ceil(ut_raw / 0.05) * 0.05
            expanded_min_bounds[j-1] = abs(lt_raw - lt_r) <= 0.025 ? lt_r - 0.05 : lt_r
            expanded_max_bounds[j-1] = min(abs(ut_raw - ut_r) <= 0.025 ? ut_r + 0.05 : ut_r, 0.0)
        end

        for j in 1:(n_params - 1)
            println("  Param $(j+1): [$(round(expanded_min_bounds[j], digits=4)), $(round(expanded_max_bounds[j], digits=4))]")
        end

        ε = 1e-10
        for j in 2:n_params
            u = [(i - 1 + rand() * (1 - 2ε)) / n_samples + ε for i in 1:n_samples]
            shuffle!(u)
            samples = expanded_min_bounds[j-1] .+ u .* (expanded_max_bounds[j-1] - expanded_min_bounds[j-1])
            x_expanded_samples[:, j] = samples
        end

        y_expanded_outputs_dict = Dict{String, Matrix{Float64}}()
        for s_id in bwfl_ids
            y_expanded_outputs_dict[s_id] = Matrix{Float64}(undef, n_samples, n_output)
        end

        p = Progress(n_samples, 1, "Generating samples: ")
        for i in 1:n_samples
            y_df_sample = forward_model(wn, x_expanded_samples[i, :], grouping, datetime_select, exclude_sensors; sim_type="chlorine", burn_in=burn_in)
            
            for s_id in bwfl_ids
                y_expanded_outputs_dict[s_id][i,:] = y_df_sample[!, Symbol(s_id)]
            end
            next!(p)
        end
        finish!(p)

        return x_expanded_samples, y_expanded_outputs_dict
    end



    function test_gp_model(gp_model, scaler, n_test, θ_samples, y, grouping, sensor, wn, datetime_select; exclude_sensors=nothing, burn_in=nothing, save_percent=100, decimal_places=4, tolerance=0.01, save_results=true)

        n_output = size(y, 2)
        n_params = size(θ_samples, 2)
        x_test = zeros(n_test, n_params)
        
        # Sample from truncated normal distributions for each parameter
        for j in 1:n_params
            # Filter outliers using IQR method
            param_data = θ_samples[:, j]
            Q1, Q3 = quantile(param_data, [0.25, 0.75])
            IQR = Q3 - Q1
            filtered_data = param_data[param_data .>= Q1 - 1.5*IQR]
            
            # Fit normal to filtered data, then truncate
            μ, σ = mean(filtered_data), std(filtered_data)
            fitted_dist = Truncated(Normal(μ, σ), -Inf, 0.0)
            
            # Draw n_test samples
            x_test[:, j] = rand(fitted_dist, n_test)
        end
        
        x_test_scaled_matrix = Matrix(MLJ.transform(scaler, DataFrame(x_test, :auto)))
        x_test_scaled_vec = [x_test_scaled_matrix[i, :] for i in 1:size(x_test_scaled_matrix, 1)]

        y_test_full = Matrix{Union{Missing, Float64}}(undef, n_test, n_output)
        for i in 1:n_test
            y_df = forward_model(wn, x_test[i, :], grouping, datetime_select, exclude_sensors; sim_type="chlorine", burn_in=burn_in)
            if Symbol(sensor) ∈ propertynames(y_df)
                 y_test_full[i,:] = y_df[!, Symbol(sensor)]
            else
                @warn "Sensor $(sensor) not found in y_df for sample $i. Marking this row as missing."
                y_test_full[i,:] .= missing 
            end
        end

        y_pred_μ_full = predict_multi_output(gp_model, x_test_scaled_vec, n_output)

        valid_mask = .!ismissing.(y_test_full)
        valid_indices_flat = findall(valid_mask)

        y_test_valid = y_test_full[valid_indices_flat]
        y_pred_μ_valid = y_pred_μ_full[valid_indices_flat]

        # compute metrics
        abs_diff_valid = abs.(y_test_valid .- y_pred_μ_valid)
        within_tolerance = if !isempty(abs_diff_valid)
            count(abs_diff_valid .<= tolerance) / length(abs_diff_valid) * 100
        else
            0.0
        end
        metrics = Dict(
            "rmse" => round(sqrt(mean(abs2.(y_test_valid .- y_pred_μ_valid))), digits=decimal_places),
            "mae" => round(mean(abs.(y_test_valid .- y_pred_μ_valid)), digits=decimal_places),
            "maxae" => round(maximum(abs.(y_test_valid .- y_pred_μ_valid)), digits=decimal_places),
            "r2" => round(r2_score(reshape(y_test_valid, :, 1), reshape(y_pred_μ_valid, :, 1)), digits=decimal_places),
            "within_$(tolerance)_mg_L" => round(within_tolerance, digits=decimal_places)
        )

        if save_percent < 100
            n_save = round(Int, n_test * save_percent / 100)
            save_indices = randperm(n_test)[1:n_save]

            results_dict = Dict(
                "x_test" => x_test[save_indices, :],
                "y_test" => y_test_full[save_indices, :],
                "y_pred_μ" => y_pred_μ_full[save_indices, :],
                "metrics" => metrics,
                "save_percent" => save_percent
            )
        else
            results_dict = Dict(
                "x_test" => x_test,
                "y_test" => y_test_full,
                "y_pred_μ" => y_pred_μ_full,
                "metrics" => metrics,
                "save_percent" => 100
            )
        end

        if save_results
            save_validation_results(results_dict, data_period, grouping, δ_b, δ_s, sensor)
        end

        return results_dict
    end


    function save_validation_results(results_dict, data_period, grouping, δ_b, δ_s, sensor)
        output_path = RESULTS_PATH * "wq/gp_models/"
        base_filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_$(sensor)_validation"
        
        JLD2.save(output_path * base_filename * ".jld2", "test_results", results_dict)
        
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
        base_filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_$(sensor)"

        # pickle files
        open(output_path * base_filename * "_model.pkl", "w") do f
            pickle.dump(model, f)
        end
        open(output_path * base_filename * "_scaler.pkl", "w") do f
            pickle.dump(scaler, f)
        end
        
        # JLD2 files
        JLD2.save(output_path * base_filename * ".jld2", "gp_results", results_dict)

        println("Saved GP model for $sensor using pickle and JLD2.")

    end

    function calculate_metrics(y_true, y_pred)
        Dict(
            "rmse" => sqrt(mean(abs2.(y_true .- y_pred))),
            "mae" => mean(abs.(y_true .- y_pred)),
            "maxae" => maximum(abs.(y_true .- y_pred)),
            "r2" => r2_score(y_true, y_pred)
        )
    end


    function r2_score(y_true, y_pred)

        if size(y_true) != size(y_pred)
            error("Dimensions of y_true and y_pred must match for R2 score calculation.")
        end

        ss_res = sum(abs2.(y_true .- y_pred))
        ss_tot = sum(abs2.(y_true .- mean(y_true, dims=1)))

        if ss_tot ≈ 0.0
            return 1.0
        end

        return 1.0 - (ss_res / ss_tot)
    end

end


