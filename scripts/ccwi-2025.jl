"""
This script quantifies disinfectant decay parameter uncertainty using Markov chain Monte Carlo (MCMC) sampling. GP emulators are first trained to reduce the computational cost of the EPANET forward model. This script is a simplified version of the the calibrate-emulate-sample methodology, and is presented at the 2025 CCWI conference.
The script is divided into the following steps:
    1. load operational data and build epanet model (DONE)
    2. set pipe grouping and define θ priors (DONE)
    3. build GP emulators (DONE)
    4. run mcmc sampling and plot results (DONE)
"""

using Revise
using DataFrames
using CSV
using Dates
using Distributed
using LinearAlgebra
using Distributions
using Statistics
using Colors
using Random
using Plots
using Plots.PlotMeasures
using PyCall
using JLD2
using LatinHypercubeSampling
using ProgressMeter
using Printf
using Surrogates
using MLJ


addprocs(4)
@everywhere using Distributions, Random, ProgressMeter, PyCall, DataFrames, LinearAlgebra, MLJ, Surrogates

pd = pyimport("pandas")
np = pyimport("numpy")
pickle = pyimport("pickle")
io = pyimport("io")
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
        else
            @error("Unsupported simulation type: $sim_type")
        end

        return y_df

    end



    function generate_gp_data_pairs(n_gp_pairs, priors, param_order, datetime_train, wn_train, grouping, exclude_sensors; force_regenerate=false)
        output_dir = joinpath(RESULTS_PATH, "ccwi-2025")
        file_name = "gp_data_pairs.jld2"
        file_path = joinpath(output_dir, file_name)

        if isfile(file_path) && !force_regenerate
            local dataset_pairs
            JLD2.@load file_path dataset_pairs
            return dataset_pairs
        end

        n_params = length(param_order)
        param_bounds = Vector{Tuple{Float64, Float64}}(undef, n_params)
        for (i, p_name) in enumerate(param_order)
            dist = priors[p_name]
            if isa(dist, Normal)
                param_bounds[i] = (mean(dist) - 3 * std(dist), mean(dist) + 3 * std(dist))
            elseif isa(dist, Uniform)
                param_bounds[i] = (minimum(dist), maximum(dist))
            else
                error("Unsupported distribution type for LHS: $(typeof(dist)) for parameter $(p_name).")
            end
        end

        scaled_lhs_samples = scaleLHC(randomLHC(n_gp_pairs, n_params), param_bounds)

        p = Progress(n_gp_pairs, 1, "Generating samples: ")
        dataset_pairs = Dict{Int, Dict{Symbol, Any}}()
        for i in 1:n_gp_pairs
            θ_sample = scaled_lhs_samples[i, :]
            y_df = forward_model(wn_train, θ_sample, grouping, datetime_train, exclude_sensors; sim_type="chlorine", burn_in=(1 * 24 * 4))
            dataset_pairs[i] = Dict(:θ_sample => θ_sample, :y_df => y_df)
            next!(p)
        end

        @save file_path dataset_pairs
        println("GP data pairs successfully generated and saved to: ", file_path)

        return dataset_pairs
    end



    function train_gp_emulators(dataset_pairs, bwfl_ids, train_indices; force_retrain=false)

        all_thetas = [pair[:θ_sample] for pair in values(dataset_pairs)]
        X_full = hcat(all_thetas...)'

        for id ∈ bwfl_ids
            output_path = joinpath(RESULTS_PATH, "ccwi-2025")
            mkpath(output_path)
            filepath = joinpath(output_path, "$(id)_gp_model.jld2")

            if isfile(filepath) && !force_retrain
                println("Julia surrogate for $id already exists. Skipping training.")
                continue
            end

            first_pair = first(values(dataset_pairs))
            n_timesteps = size(first_pair[:y_df], 1)
            n_total_samples = length(dataset_pairs)
            Y_full = zeros(n_total_samples, n_timesteps)

            for (i, pair) in enumerate(values(dataset_pairs))
                y_df_sample = pair[:y_df]
                if Symbol(id) ∈ propertynames(y_df_sample)
                    Y_full[i, :] = y_df_sample[!, Symbol(id)]
                else
                    @warn "Sensor $(id) not found in y_df for sample $i. Using zeros."
                end
            end

            X_train = X_full[train_indices, :]
            Y_train = Y_full[train_indices, :]
            X_train_table = DataFrame(X_train, :auto)

            scaler_model = Standardizer()
            scaler = machine(scaler_model, X_train_table)
            fit!(scaler)

            X_train_scaled_table = MLJ.transform(scaler, X_train_table)
            X_train_scaled_mat = Matrix(X_train_scaled_table)
            X_train_scaled_vec = [X_train_scaled_mat[i, :] for i in 1:size(X_train_scaled_mat, 1)]

            gp_surrogates = Vector{Any}(undef, n_timesteps)
            println("\nTraining GP surrogates for ID: $id")

            lower_bounds = vec(minimum(X_train_scaled_mat, dims=1))
            upper_bounds = vec(maximum(X_train_scaled_mat, dims=1))

            for t in 1:n_timesteps
                y_timestep = Y_train[:, t]
                surrogate = Kriging(X_train_scaled_vec, y_timestep, lower_bounds, upper_bounds)
                gp_surrogates[t] = surrogate
            end

            model_to_save = (
                surrogates = gp_surrogates,
                scaler = scaler
            )
            
            jldsave(filepath; model_to_save...)
   
        end
    end




    function test_gp_emulators(dataset_pairs, bwfl_ids, test_indices; tolerance=0.025)

        all_thetas = [pair[:θ_sample] for pair in values(dataset_pairs)]
        X_full = hcat(all_thetas...)'
        X_test = X_full[test_indices, :]
        X_test_table = DataFrame(X_test, :auto)

        n_test_samples = size(X_test, 1)
        first_pair = first(values(dataset_pairs))
        n_timesteps = size(first_pair[:y_df], 1)

        results_table = DataFrame(
            bwfl_id = String[],
            rmse = Float64[],
            mae = Float64[],
            max_ae = Float64[],
            percent_within_tol = Float64[]
        )
        
        test_predictions = Dict{String, Dict{Symbol, Vector{Float64}}}()
        for id ∈ bwfl_ids
            output_path = joinpath(RESULTS_PATH, "ccwi-2025")
            filepath = joinpath(output_path, "$(id)_gp_model.jld2")

            if !isfile(filepath)
                @warn "Surrogate file for $id not found at $filepath. Skipping."
                continue
            end

            gp_model = JLD2.load(filepath)
            gp_surrogates = gp_model["surrogates"]
            scaler = gp_model["scaler"]

            Y_test = zeros(n_test_samples, n_timesteps)
            for (i, test_idx) in enumerate(test_indices)     
                y_df_sample = dataset_pairs[test_idx][:y_df]
                if Symbol(id) ∈ propertynames(y_df_sample)
                    Y_test[i, :] = y_df_sample[!, Symbol(id)]
                else
                    @warn "Sensor $(id) not found in y_df for sample $(test_idx). Using zeros for true values."
                end
            end

            X_test_scaled_table = MLJ.transform(scaler, X_test_table)
            X_test_scaled_mat = Matrix(X_test_scaled_table)
            X_test_scaled_vec = [X_test_scaled_mat[i, :] for i in 1:n_test_samples]

            y_pred_μ = zeros(n_test_samples, n_timesteps)
            y_pred_σ = zeros(n_test_samples, n_timesteps)

            for t in 1:n_timesteps
                surrogate = gp_surrogates[t]
                y_pred_μ[:, t] = surrogate.(X_test_scaled_vec)
                y_pred_σ[:, t] = std_error_at_point.(Ref(surrogate), X_test_scaled_vec)
            end
            
            abs_diff = abs.(Y_test - y_pred_μ)
            rmse_val = sqrt(mean((Y_test - y_pred_μ).^2))
            mae_val = mean(abs_diff)
            max_ae_val = maximum(abs_diff)
            percent_within_tolerance = count(abs_diff .<= tolerance) / length(abs_diff) * 100

            push!(results_table, (
                bwfl_id = id,
                rmse = rmse_val,
                mae = mae_val,
                max_ae = max_ae_val,
                percent_within_tol = percent_within_tolerance
            ))

            test_predictions[id] = Dict(
                :y_test => vec(Y_test),
                :y_pred_μ => vec(y_pred_μ),
                :y_pred_σ => vec(y_pred_σ)
            )

            # max_val = max(maximum(Y_test), maximum(y_pred_μ)) * 1.05
            max_val = 0.8
            p = scatter(vec(Y_test), vec(y_pred_μ),
                xlabel="EPANET (True)",
                ylabel="GP Surrogate (Predicted)",
                title="Sensor: $(id)",
                legend=false,
                markersize=3,
                markeralpha=0.6,
                # markercolor=wong_colors[6], 
                markerstrokewidth=0,
                grid=false,
                xlims=(0, max_val),
                ylims=(0, max_val),
                aspect_ratio=:equal
            )
            
            plot!(p, [0.0, max_val], [0.0, max_val],
                linestyle=:dash,
                color=:black,
                linewidth=1.5,
                label="1:1 Line"
            )
            display(p)
        end

        println("\nPerformance Summary of Julia GP Emulators:")
        println(results_table)

        return test_predictions
    end



    function load_gp_models(bwfl_ids)
        # gp_dict = Dict{String, Dict{String, PyObject}}()
        gp_dict = Dict{String, Dict{String, Any}}()
        gp_model_path = RESULTS_PATH * "ccwi-2025/"
        
        for id ∈ bwfl_ids
            output_path = joinpath(RESULTS_PATH, "ccwi-2025")
            filepath = joinpath(output_path, "$(id)_gp_model.jld2")
            gp_model = JLD2.load(filepath)
            surrogates = gp_model["surrogates"]
            scaler = gp_model["scaler"]

            gp_dict[id] = Dict("gp_model" => surrogates, "gp_scaler" => scaler)
        end

        #     model_file = gp_model_path * "$(id)_gp_model.pkl"
        #     scaler_file = gp_model_path * "$(id)_gp_scaler.pkl"
        #     gp_model = nothing
        #     gp_scaler = nothing
        #     try
        #         model_bytes = read(model_file)
        #         scaler_bytes = read(scaler_file)
        #         gp_model = pickle.load(io.BytesIO(model_bytes))
        #         gp_scaler = pickle.load(io.BytesIO(scaler_bytes))
                
        #         gp_dict[id] = Dict("gp_model" => gp_model, "gp_scaler" => gp_scaler)
        #     catch e
        #         @warn "Error loading GP model files for sensor $id: $e. Skipping this sensor."
        #     end
        # end

        return gp_dict
    end


    @everywhere function get_gp_prediction(gp_dict, id, θ)

        surrogates = gp_dict[id]["gp_model"]
        scaler = gp_dict[id]["gp_scaler"]
        n_timesteps = length(surrogates)

        X = hcat([θ]...)'
        X_table = DataFrame(X, :auto)
        X_scaled_table = MLJ.transform(scaler, X_table)
        X_scaled_mat = Matrix(X_scaled_table)
        X_scaled_vec = vec(X_scaled_mat)

        y_pred_μ = zeros(n_timesteps)
        y_pred_σ = zeros(n_timesteps)

        for t in 1:n_timesteps
            surrogate = surrogates[t]
            y_pred_μ[t] = surrogate(X_scaled_vec)
            y_pred_σ[t] = std_error_at_point(surrogate, X_scaled_vec)
        end

        return y_pred_μ, y_pred_σ
    end


    @everywhere function log_prior(θ, priors_dict)
        lp = 0.0

        bulk_prior = priors_dict[:B]
        if θ[1] > 0
            return -Inf
        end
        lp += logpdf(bulk_prior, θ[1])

        wall_priors_keys = [:G1, :G2, :G3]
        for (i, p_key) in enumerate(wall_priors_keys)
            wall_prior = priors_dict[p_key]
            if isa(wall_prior, Uniform)
                lb = minimum(wall_prior)
                ub = maximum(wall_prior)
                if θ[i+1] < lb || θ[i+1] > ub
                    return -Inf
                end
                lp += -log(ub - lb)
            else
                @error "Unsupported prior type for wall decay: $(typeof(wall_prior))"
            end
        end
        return lp
    end



    @everywhere function log_likelihood(y_obs_i, gp_dict, id, θ, ϵ)
        y_pred_μ, y_pred_σ = get_gp_prediction(gp_dict, id, θ)

        y_obs_vec = collect(y_obs_i)
        not_missing_indices = findall(!ismissing, y_obs_vec)

        y_obs_filtered = y_obs_vec[not_missing_indices]
        y_pred_μ_filtered = y_pred_μ[not_missing_indices]
        y_pred_σ_filtered = y_pred_σ[not_missing_indices]

        if isempty(not_missing_indices)
            return -Inf
        end
        residual = y_obs_vec[not_missing_indices] - y_pred_μ[not_missing_indices]
        # Δ = ϵ^2 .+ y_pred_σ[not_missing_indices].^2
        Δ = ϵ^2
        # return -0.5 * sum(residual.^2 ./ Δ) - 0.5 * sum(log.(Δ))
        return -0.5 * sum(residual.^2 ./ Δ)
    end




    @everywhere function log_posterior(y_obs, gp_dict, priors_dict, bwfl_ids, θ, ϵ)
        lp = log_prior(θ, priors_dict)
        if lp == -Inf
            return lp
        end

        ll = 0.0
        for id ∈ bwfl_ids
            y_obs_i = subset(y_obs, :bwfl_id => ByRow(==(id)))[!, :mean]
            ll += log_likelihood(y_obs_i, gp_dict, id, θ, ϵ)
        end
        return lp + ll
    end




    function run_mcmc(y_obs, gp_dict, priors_dict, θ_init; n_samples=50000, f_b=0.2, λ=0.1, parallel=false, ϵ=0.1)

        n_chains = length(θ_init)
        n_params = length(θ_init[1])
        burn_in = round(Int, f_b * n_samples)

        @everywhere function run_single_chain(y_obs, gp_dict, bwfl_ids, priors_dict, θ_init, chain_id, ϵ, λ, n_params, n_samples, burn_in)

            rng = MersenneTwister()
            θ_current = copy(θ_init)
            logp_current = log_posterior(y_obs, gp_dict, priors_dict, bwfl_ids, θ_current, ϵ)
            chain_samples = zeros(n_samples, n_params)
            chain_logps = zeros(n_samples)
            accepts = 0

            # fix proposal distribution
            base_proposal_stds = [std(priors_dict[:B]); λ .* abs.(θ_init[2:end])]
            proposal_cov = Diagonal(base_proposal_stds .^ 2)
            proposal_dist = MvNormal(zeros(n_params), proposal_cov)

            progress_interval = round(Int, n_samples / 10)
            println("Chain $chain_id: $(round(Int, 100 * 0 / n_samples))% complete")

            for i in 1:n_samples
                # proposal_stds = [std(priors_dict[:B]); λ .* abs.(θ_current[2:end])]
                # proposal_dist = MvNormal(zeros(n_params), Diagonal(proposal_stds.^2))
                θ_proposal = θ_current .+ rand(rng, proposal_dist)
                θ_proposal = min.(θ_proposal, 0.0)
                logp_proposal = log_posterior(y_obs, gp_dict, priors_dict, bwfl_ids, θ_proposal, ϵ)

                if log(rand(rng)) < (logp_proposal - logp_current)
                    θ_current = copy(θ_proposal)
                    logp_current = copy(logp_proposal)
                    accepts += 1
                end
                chain_samples[i, :] = copy(θ_current)
                chain_logps[i] = copy(logp_current)

                if i % progress_interval == 0
                    println("Chain $chain_id: $(round(Int, 100 * i / n_samples))% complete")
                end
            end

            return (
                samples = chain_samples[burn_in+1:end, :],
                log_posteriors = chain_logps[burn_in+1:end],
                accepts = accepts
            )
        end

        println("Running $n_chains chains of $n_samples samples each...")

        results = parallel ? pmap(c -> run_single_chain(y_obs, gp_dict, bwfl_ids, priors_dict, θ_init[c], c, ϵ, λ, n_params, n_samples, burn_in), 1:n_chains) : map(c -> run_single_chain(y_obs, gp_dict, bwfl_ids, priors_dict, θ_init[c], c, ϵ, λ, n_params, n_samples, burn_in), 1:n_chains)

        samples = Array{Float64}(undef, n_samples - burn_in, n_params, n_chains)
        log_posteriors = Array{Float64}(undef, n_samples - burn_in, n_chains)
        accepts = zeros(Int, n_chains)

        for (chain, result) in enumerate(results)
            samples[:, :, chain] = result.samples
            log_posteriors[:, chain] = result.log_posteriors
            accepts[chain] = result.accepts
        end

        return Dict(
            "samples" => samples,
            "log_posteriors" => log_posteriors,
            "accepts" => accepts,
            "acceptance_rates" => accepts ./ n_samples,
        )
    end


    function plot_mcmc_param_histogram(samples_for_param, prior_dist, label, x_limits, bin_width, color)
        first_bin_edge = floor(x_limits[1] / bin_width) * bin_width
        bin_edges = first_bin_edge:bin_width:(x_limits[2] + bin_width)

        if label == "G3 [m/d]"
            x_tick_start = floor(x_limits[1] / 0.025) * 0.025
            x_tick_end = ceil(x_limits[2] / 0.025) * 0.025
            x_ticks_values = x_tick_start:0.025:x_tick_end
        else
            x_tick_start = floor(x_limits[1] / 0.05) * 0.05
            x_tick_end = ceil(x_limits[2] / 0.05) * 0.05
            x_ticks_values = x_tick_start:0.05:x_tick_end
        end


        p = histogram(samples_for_param,
            bins=bin_edges,
            xlabel=label,
            xticks=x_ticks_values,
            xlims=x_limits,
            xguidefontsize=18,
            yguidefontsize=18,
            xtickfont=16,
            ytickfont=16,
            legendfontsize=16,
            legend=(0.025, 0.95),
            label="posterior",
            normalize=:none,
            fillcolor=color,
            linecolor=:white,
            linewidth=0.5,
            grid=false,
            size=(500, 400),
            left_margin=2mm,
            bottom_margin=4mm,
            right_margin=6mm,
            top_margin=5mm,
            yaxis=false,
            ticklength=0mm,
            xformatter = x -> Printf.@sprintf("%g", x),
        )
        
        scaling_factor = length(samples_for_param) * bin_width

        if isa(prior_dist, Normal)
            x_vals_prior = range(x_limits[1], x_limits[2], length=200)
            pdf_vals = pdf.(prior_dist, x_vals_prior)
            scaled_pdf_vals = pdf_vals .* scaling_factor

            plot!(p, x_vals_prior, scaled_pdf_vals,
                line=(:dash, 2, "black"),
                label="prior",
                foreground_color_legend=nothing
            )
        elseif isa(prior_dist, Uniform)
            lower_bound = minimum(prior_dist)
            upper_bound = maximum(prior_dist)
            prior_pdf_val = 1 / (upper_bound - lower_bound)
            scaled_prior_val = prior_pdf_val * scaling_factor
            plot!(p, [lower_bound, upper_bound], [scaled_prior_val, scaled_prior_val],
                line=(:dash, 2, "black"),
                label="prior",
                foreground_color_legend=nothing
            )
        else
            @warn "Unsupported prior distribution type for plotting: $(typeof(prior_dist))"
        end

        return p
    end

end






########## MAIN SCRIPT ##########

### 1. load operational data and build epanet model ###

data_period = 18 # (Aug. 2024)
padded_period = lpad(data_period, 2, "0")
flow_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-flow.csv", DataFrame); flow_df.datetime = DateTime.(flow_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
pressure_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-pressure.csv", DataFrame); pressure_df.datetime = DateTime.(pressure_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
wq_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-wq.csv", DataFrame); wq_df.datetime = DateTime.(wq_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
cl_df = wq_df[wq_df.data_type .== "chlorine", :]
temp_df = wq_df[wq_df.data_type .== "temperature", :]

ylabel = "Temperature [°C]" # "Flow [L/s]", "Pressure [m]", "Chlorine [mg/L]", "Temperature [°C]"
df = temp_df # cl_df, flow_df, pressure_df
p1 = plot_bwfl_data(df, ylabel, ymax=30)

datetime = DateTime.(unique(flow_df.datetime)) 
n_total = length(datetime) 

n_train = 4 * 24 * 4 # 4 days for training; 1-day burn-in period for wq stabilization
datetime_train = datetime[1:((1 * 24 * 4) + n_train)]
wn_train = epanet.build_model(
    df_2_pd(filter(row -> row.datetime ∈ datetime_train, flow_df)), 
    df_2_pd(filter(row -> row.datetime ∈ datetime_train, pressure_df)), 
    df_2_pd(filter(row -> row.datetime ∈ datetime_train, cl_df)),
    sim_type="chlorine"
)
cl_df_train = filter(row -> row.datetime ∈ datetime_train, cl_df)

datetime_test = datetime[n_train+1:end]
wn_test = epanet.build_model(
    df_2_pd(filter(row -> row.datetime ∈ datetime_test, flow_df)), 
    df_2_pd(filter(row -> row.datetime ∈ datetime_test, pressure_df)), 
    df_2_pd(filter(row -> row.datetime ∈ datetime_test, cl_df)),
    sim_type="chlorine"
)
cl_df_test = filter(row -> row.datetime ∈ datetime_test, cl_df)





### 2. set pipe grouping and define θ priors ###

# bulk decay
θ_b = -0.7  # day^-1 (from bottle tests)
temp_train = mean(skipmissing(subset(wq_df, :data_type => ByRow(==("temperature")), :datetime => ByRow(in(datetime_train)))[:, :mean]))
θ_b_train = data.bulk_temp_adjust(θ_b, temp_train)
temp_test = mean(skipmissing(subset(wq_df, :data_type => ByRow(==("temperature")), :datetime => ByRow(in(datetime_test)))[:, :mean]))
θ_b_test = data.bulk_temp_adjust(θ_b, temp_test)

σ_b = 0.025 * abs(θ_b_train)
prior_θ_b_train = Normal(θ_b_train, σ_b)
prior_θ_b_test = Normal(θ_b_test, σ_b)

# wall decay --> G1: metallic + > mean pipe age, G2: metallic + ≤ mean pipe age, G3: cement + plastic
grouping = "material-age"
θ_w_lb = [-0.25, -0.25, -0.1]
θ_w_ub = [0.0, 0.0, 0.0]
prior_θ_w_groups = map( (lb, ub) -> Uniform(lb, ub), θ_w_lb, θ_w_ub)

# combined priors
priors_train = Dict(
    :B => prior_θ_b_train,
    :G1 => prior_θ_w_groups[1],
    :G2 => prior_θ_w_groups[2],
    :G3 => prior_θ_w_groups[3]
)
priors_test = Dict(
    :B => prior_θ_b_test,
    :G1 => prior_θ_w_groups[1],
    :G2 => prior_θ_w_groups[2],
    :G3 => prior_θ_w_groups[3]
)





### 3. build GP emulators ###

# create (θ, y) pairs using latin hypercube sampling
exclude_sensors = ["BW1", "BW4", "BW7", "BW5_2"]
param_order = [:B, :G1, :G2, :G3]
n_gp_pairs = 500
force_regenerate = false
dataset_pairs_cpu_time = @elapsed begin
    dataset_pairs = generate_gp_data_pairs(n_gp_pairs, priors_train, param_order, datetime_train, wn_train, grouping, exclude_sensors; force_regenerate=force_regenerate)
end




# train GP emulators
Random.seed!(42)
train_ratio = 0.8
train_indices = randperm(n_gp_pairs)[1:floor(Int, n_gp_pairs * train_ratio)]
test_indices = setdiff(1:n_gp_pairs, train_indices)
bwfl_ids = filter(id -> !(id ∈ exclude_sensors), unique(cl_df_train.bwfl_id))
force_retrain = false

gp_train_cpu_time = @elapsed begin
    train_gp_emulators(dataset_pairs, bwfl_ids, train_indices; force_retrain=force_retrain)
end
test_data = test_gp_emulators(dataset_pairs, bwfl_ids, test_indices; tolerance=0.05)




### 4. run mcmc sampling algorithm ###

gp_dict = load_gp_models(bwfl_ids)
y_obs = filter(row -> row.datetime ∈ datetime_train[1*24*4+1:end], cl_df_train)
ϵ = 0.2
λ = 0.1
f_b = 0.2
n_chains = 4
n_samples = 10000
parallel = true
# θ_init = [[rand(priors_train[:B]); rand(priors_train[:G1]); rand(priors_train[:G2]); rand(priors_train[:G3])] for _ in 1:n_chains]
θ_guess = [θ_b_train, -0.15, -0.15, -0.01]
θ_init = [θ_guess .+ 0.0001 .* randn(length(θ_guess)) for _ in 1:n_chains]

mcmc_results = run_mcmc(y_obs, gp_dict, priors_train, θ_init; n_samples=n_samples, f_b=f_b, λ=λ, parallel=parallel, ϵ=ϵ)

# Export each parameter's MCMC samples to CSV
chain = 1
for i in 1:length(θ_guess)
    param_samples = mcmc_results["samples"][:, i, chain]
    param_name = ["B", "G1", "G2", "G3"][i]
    output_file = joinpath(RESULTS_PATH, "ccwi-2025", "mcmc_samples_$(param_name).csv")
    CSV.write(output_file, DataFrame("$(param_name)" => param_samples))
end



# temp plotting
plot(mcmc_results["log_posteriors"], xlabel="Sample Index", ylabel="Log Posterior")
plot(mcmc_results["samples"][:, 1, :], xlabel="Sample Index", ylabel="Bulk Decay Coefficient (B)")
plot(mcmc_results["samples"][:, 2, :], xlabel="Sample Index", ylabel="Wall Decay Coefficient (G1)")
plot(mcmc_results["samples"][:, 3, :], xlabel="Sample Index", ylabel="Wall Decay Coefficient (G2)")
plot(mcmc_results["samples"][:, 4, :], xlabel="Sample Index", ylabel="Wall Decay Coefficient (G3)")



# bulk decay (B)
b_samples = CSV.read(RESULTS_PATH * "ccwi-2025/mcmc_samples_B.csv", DataFrame)[:, :B]
b_label = "B [1/d]"
b_xlims = (-0.8, -0.5)
b_bin_width = 0.013
plot_b = plot_mcmc_param_histogram(b_samples, priors_train[:B], b_label, b_xlims, b_bin_width, wong_colors[6])

# wall decay (G1)
g1_samples = CSV.read(RESULTS_PATH * "ccwi-2025/mcmc_samples_G1.csv", DataFrame)[:, :G1]
g1_label = "G1 [m/d]"
g1_xlims = (-0.25001, 0.0)
g1_bin_width = 0.01
plot_g1 = plot_mcmc_param_histogram(g1_samples, priors_train[:G1], g1_label, g1_xlims, g1_bin_width, wong_colors[6])

# wall decay (G2)
g2_samples = CSV.read(RESULTS_PATH * "ccwi-2025/mcmc_samples_G2.csv", DataFrame)[:, :G2]
g2_label = "G2 [m/d]"
g2_xlims = (-0.250001, 0.0)
g2_bin_width = 0.01
plot_g2 = plot_mcmc_param_histogram(g2_samples, priors_train[:G2], g2_label, g2_xlims, g2_bin_width, wong_colors[6])

# wall decay (G3)
g3_samples = CSV.read(RESULTS_PATH * "ccwi-2025/mcmc_samples_G3.csv", DataFrame)[:, :G3]
g3_label = "G3 [m/d]"
g3_xlims = (-0.1001, 0.0)
g3_bin_width = 0.004
plot_g3 = plot_mcmc_param_histogram(g3_samples, priors_train[:G3], g3_label, g3_xlims, g3_bin_width, wong_colors[6])


# make time series plots
θ_mean = [mean(b_samples), mean(g1_samples), mean(g2_samples), mean(g3_samples)]
θ_2_5 = [quantile(b_samples, 0.025), quantile(g1_samples, 0.025), quantile(g2_samples, 0.025), quantile(g3_samples, 0.025)]
θ_97_5 = [quantile(b_samples, 0.975), quantile(g1_samples, 0.975), quantile(g2_samples, 0.975), quantile(g3_samples, 0.975)]

θ_lb_prior = [θ_b_train - 3 * σ_b, -0.25, -0.25, -0.1]
θ_ub_prior = [θ_b_train + 3 * σ_b, 0.0, 0.0, 0.0]

y_df_mean = forward_model(wn_test, θ_mean, grouping, datetime_test, exclude_sensors; sim_type="chlorine", burn_in=(1 * 24 * 4))
y_df_2_5 = forward_model(wn_test, θ_2_5, grouping, datetime_test, exclude_sensors; sim_type="chlorine", burn_in=(1 * 24 * 4))
y_df_97_5 = forward_model(wn_test, θ_97_5, grouping, datetime_test, exclude_sensors; sim_type="chlorine", burn_in=(1 * 24 * 4))
y_df_lb_prior = forward_model(wn_test, θ_lb_prior, grouping, datetime_test, exclude_sensors; sim_type="chlorine", burn_in=(1 * 24 * 4))
y_df_ub_prior = forward_model(wn_test, θ_ub_prior, grouping, datetime_test, exclude_sensors; sim_type="chlorine", burn_in=(1 * 24 * 4))


sensor_id = "BW12"

begin
    x_indices = 1:length(datetime_test[1*24*4+1:end])
    tick_positions = 0:48:length(x_indices)
    tick_labels = Int.(tick_positions ./ 4)

    observed_data = filter(row -> row.bwfl_id == sensor_id, cl_df_test).mean[1*24*4+1:end]
    plot(x_indices, observed_data, label="obs.", color="black", linewidth=1.5, xlabel="Hour", ylabel="Chlorine [mg/L]", size=(500, 350), left_margin=8mm, right_margin=8mm, bottom_margin=8mm, top_margin=8mm, xtickfont=14, ytickfont=14, xguidefont=16, yguidefont=16,legendfont=14,grid=false, legend=(0.55, 1.0), ylims=(0, 1.0), yticks=(0:0.25:1.0), xticks=(tick_positions, tick_labels), legend_foreground_color=nothing, foreground_color_legend=nothing)

    lower_bound_posterior = y_df_2_5[!, sensor_id]
    upper_bound_posterior = y_df_97_5[!, sensor_id]
    mean_sim_posterior = y_df_mean[!, sensor_id]
    lower_bound_prior = y_df_lb_prior[!, sensor_id]
    upper_bound_prior = y_df_ub_prior[!, sensor_id]

    plot!(x_indices, lower_bound_prior, fillrange = upper_bound_prior, fillcolor = wong_colors[1], fillalpha = 0.075, linewidth = 0, linealpha = 0, label = "sim. (prior)")
    plot!(x_indices, upper_bound_posterior, fillrange = lower_bound_posterior, fillcolor = wong_colors[2], fillalpha = 0.3, linewidth = 0, linealpha = 0, label = "sim. (posterior)")
    plot!(x_indices, mean_sim_posterior, label="", color=wong_colors[2], linewidth=1.5)
end


