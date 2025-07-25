"""
This script performs MCMC sampling using the GP emulator trained from the ensemble Kalman inversion calibration. The following steps are performed:
    1. load eki results, GP model, and operational data (DONE)
    2. define (a) prior, (b) likelihood, and (c) posterior functions (DONE)
    3. create Metropolis-Hastings MCMC algorithm (DONE)
    4. run MCMC to sample from posterior distribution (DONE)
    5. results plotting (DONE)
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
using Surrogates
using MLJ
using PyCall
using Printf
using KernelDensity
using ProgressMeter
using Distributed
addprocs(4)
@everywhere using Distributions, Random, ProgressMeter, PyCall, DataFrames, MLJ, Surrogates, JLD2, LinearAlgebra

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

### 1. load eki results, GP model, operational data, and other MCMC parameters ###
data_period = 18 # (aug. 2024)
padded_period = lpad(data_period, 2, "0")
grouping = "material" # "single", "material", "material-age", "material-age-velocity"
δ_s = 0.2 # 0.1, 0.2
δ_b = 0.05

# eki results
eki_results_path = RESULTS_PATH * "wq/eki_calibration/"
eki_filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s)).jld2"
eki_results = JLD2.load(eki_results_path * eki_filename, "eki_results")
bwfl_ids = [string(col) for col in propertynames(eki_results[1]["y_df"]) if col != :datetime]

# gp models
gp_models_path = RESULTS_PATH * "wq/gp_models/"
gp_filename_base = gp_models_path * "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))"
gp_dict = Dict{String, Dict{String, Any}}()
for sensor_id ∈ bwfl_ids
    model_file = gp_filename_base * "_$(sensor_id)_gp.jld2"
    gp_files = JLD2.load(model_file)
    gp_dict[sensor_id] = Dict(
        "gp_model" => gp_files["surrogates"],
        "scaler" => gp_files["scaler"]
    )
end

# operational data
wq_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-wq.csv", DataFrame); wq_df.datetime = DateTime.(wq_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
y_df = subset(wq_df, 
    :data_type => ByRow(==("chlorine")), 
    :datetime => ByRow(in(Set(eki_results[1]["y_df"].datetime)))
)

# bulk parameter
n_ensemble = length(eki_results)
θ_samples = hcat([eki_results[i]["θ"] for i in 1:n_ensemble]...)'
mcmc_lb, mcmc_ub = calculate_mcmc_bounds(θ_samples)
θ_b = mean(θ_samples[:, 1])
bulk_std = abs(θ_b * δ_b)

# wall parameters
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





### 2a. define prior function ###

@everywhere begin
    function log_prior(θ)
        lp = 0.0

        if θ[1] > 0
            return -Inf
        end

        bulk_mean = $(θ_b)
        bulk_std = abs($(θ_b) * $(δ_b))
        bulk_prior = Normal(bulk_mean, bulk_std)
        lp += logpdf(bulk_prior, θ[1])

        for (i, (lb, ub)) in enumerate(zip($(θ_w_lb), $(θ_w_ub)))

            if θ[i+1] < $(mcmc_lb)[i+1] || θ[i+1] > $(mcmc_ub)[i+1]
                return -Inf
            end

            lp += -log(ub - lb)
        end
        return lp
    end
end





### 2b. define likelihood function ###

@everywhere begin
    function log_likelihood(θ, gp_model, scaler, y_obs, valid_indices)

        n_outputs = length(y_obs)

        # scale input data
        X = hcat([θ]...)'
        X_table = DataFrame(X, :auto)
        X_scaled_table = MLJ.transform(scaler, X_table)
        X_scaled_mat = Matrix(X_scaled_table)
        X_scaled_vec = vec(X_scaled_mat)

        # get gp predictions
        y_pred_μ = zeros(n_outputs)
        # y_pred_σ = zeros(n_outputs)
        for t in 1:n_outputs
            surrogate = gp_model[t]
            y_pred_μ[t] = surrogate(X_scaled_vec)
            # y_pred_σ[t] = std_error_at_point(surrogate, X_scaled_vec)
        end
        residual = y_obs[valid_indices] - y_pred_μ[valid_indices]
        # variance = $(δ_s).^2 .+ y_pred_σ[valid_indices].^2
        # δ = max.((y_obs[valid_indices] .* $(δ_s)), 0.025)
        # δ = max.((y_obs[valid_indices] .* $(δ_s)), 0.05)
        variance = $(δ_s).^2
        
        return -0.5 * sum((residual.^2) ./ variance)
    end
end




### 2c. define posterior function ###

@everywhere begin
    function log_posterior(θ)
        lp = log_prior(θ)
        if lp < -1e4
            return lp
        end
        ll = 0.0
        for sensor ∈ $(bwfl_ids)
            if haskey($(gp_dict), sensor)
                gp_model = $(gp_dict)[sensor]["gp_model"]
                scaler = $(gp_dict)[sensor]["scaler"]
                y_obs = subset($(y_df), :bwfl_id => ByRow(==(sensor)))[!, :mean]
                missing_mask = [!ismissing(val) ? 1 : 0 for val in y_obs]
                valid_indices = findall(x -> x == 1, missing_mask)
                ll += log_likelihood(θ, gp_model, scaler, y_obs, valid_indices)
            end
        end
        return lp + ll
    end
end




### 3. create metropolis-hastings MCMC sampler ###

function run_mcmc(θ_init_list, θ_samples; n_samples=50000, burn_in_frac=0.2, scaling_factors=nothing, parallel=false)

    if isnothing(scaling_factors)
        scaling_factors = 0.1 * ones(n_params)
    end
    S = Diagonal(scaling_factors)

    n_chains = length(θ_init_list)
    n_params = length(θ_init_list[1])
    burn_in = round(Int, burn_in_frac * n_samples)
    proposal_cov = cov(θ_samples)
    scaled_cov = S * proposal_cov * S
    proposal_dist = MvNormal(zeros(n_params), scaled_cov)

    function run_single_chain(θ_init, chain_id)

        rng = MersenneTwister()
        θ_current = copy(θ_init)
        logp_current = log_posterior(θ_current)
        chain_samples = zeros(n_samples, n_params)
        chain_logps = zeros(n_samples)
        accepts = 0

        for i in 1:n_samples
            θ_proposal = θ_current .+ rand(rng, proposal_dist)
            θ_proposal = min.(θ_proposal, 0.0)
            logp_proposal = log_posterior(θ_proposal)
            if log(rand(rng)) < (logp_proposal - logp_current)
                θ_current = θ_proposal
                logp_current = logp_proposal
                accepts += 1
            end
            chain_samples[i, :] = θ_current
            chain_logps[i] = logp_current

            if i % (n_samples ÷ 10) == 0
                println("Chain $chain_id: $(round(Int, 100 * i / n_samples))% complete")
                flush(stdout)
            end
        end

        return (
            samples = chain_samples[burn_in+1:end, :],
            log_posteriors = chain_logps[burn_in+1:end],
            accepts = accepts
        )
    end

    println("Running $n_chains chains of $n_samples samples each (burn-in: $burn_in)")

    results = parallel ? pmap(c -> run_single_chain(θ_init_list[c], c), 1:n_chains) :
                         map(c -> run_single_chain(θ_init_list[c], c), 1:n_chains)

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
        "burn_in" => burn_in,
        "n_samples_per_chain" => n_samples
    )
end





### 4. run MCMC algorithm ###

begin
    scaling_factors = [0.2, 0.4, 0.4]
    parallel = true
    n_samples = 100000
    θ_samples = hcat([eki_results[i]["θ"] for i in 1:n_ensemble]...)'
    θ_init = vec(mean(θ_samples, dims=1))
    θ_init = [-0.66, -0.075, -0.055]
    θ_init_list = [θ_init .+ 0.0001 .* randn(length(θ_init)) for _ in 1:4]
    for i in 1:length(θ_init_list)
        θ_init_list[i] = [val > 0 ? 0.0 : val for val in θ_init_list[i]]
    end
end

mcmc_results = run_mcmc(θ_init_list, θ_samples; n_samples=n_samples, scaling_factors=scaling_factors, parallel=parallel)

println(mcmc_results["acceptance_rates"])
r_hat = assess_mcmc_convergence(mcmc_results)




### 5. results plotting ###
param_1 = 1
param_2 = 1
save_tex = false
contours = false
thin = 20

p1a = plot_trace(mcmc_results, param_1, save_tex=save_tex, thin=thin)
if param_1 == 1
    p1b = plot_cdf(mcmc_results, param_1; θ_b=θ_b, δ_b=δ_b, show_prior=true, save_tex=save_tex)
    display(p1b)
    p2a = plot_param_dist(mcmc_results, param_1, param_2, save_tex=save_tex, contours=contours, θ_b=θ_b, δ_b=δ_b, show_prior=true)
else
    p1b = plot_cdf(mcmc_results, param_1; θ_b=θ_b, δ_b=δ_b, show_prior=true, θ_w_lb=θ_w_lb[param_1-1], θ_w_ub=θ_w_ub[param_1-1], save_tex=save_tex)
    display(p1b)
    p2a = plot_param_dist(mcmc_results, param_1, param_2, save_tex=save_tex, contours=contours, θ_b=θ_b, δ_b=δ_b, show_prior=true, θ_w_lb=θ_w_lb[param_1-1], θ_w_ub=θ_w_ub[param_1-1])
end

export_mcmc_samples(mcmc_results, param_1, thin_to=5000)

 





########## FUNCTIONS ##########

begin

    function assess_mcmc_convergence(mcmc_results)

        chains = mcmc_results["samples"]
        n_samples, n_params, n_chains = size(chains)
        
        if n_chains < 2
            println("Cannot calculate Gelman-Rubin statistic: need at least 2 chains")
            return nothing
        end
        
        r_hat = zeros(n_params)
        
        for p in 1:n_params

            param_chains = chains[:, p, :]
            chain_vars = [var(param_chains[:, c]) for c in 1:n_chains]
            W = mean(chain_vars)
            
            chain_means = [mean(param_chains[:, c]) for c in 1:n_chains]
            B = n_samples * var(chain_means)
            V = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
            r_hat[p] = sqrt(V / W)

        end
        
        println("Gelman-Rubin convergence assessment...")
        for p in 1:n_params
            parameter_name = "Parameter $p"
            if haskey(mcmc_results, "parameter_names")
                parameter_name = mcmc_results["parameter_names"][p]
            end
            
            if r_hat[p] < 1.01
                println("$parameter_name: R̂ = $(round(r_hat[p], digits=3)) (excellent convergence)")
            elseif r_hat[p] < 1.05
                println("$parameter_name: R̂ = $(round(r_hat[p], digits=3)) (good convergence)")
            elseif r_hat[p] < 1.1
                println("$parameter_name: R̂ = $(round(r_hat[p], digits=3)) (acceptable convergence)")
            else
                println("$parameter_name: R̂ = $(round(r_hat[p], digits=3)) (poor convergence - consider running chains longer)")
            end
        end
        
        max_r_hat = maximum(r_hat)
        println("\nWorst R̂ value: $(round(max_r_hat, digits=3))")
        if max_r_hat < 1.1
            println("All parameters appear to have converged (R̂ < 1.1)")
        else
            println("Some parameters have not converged (R̂ > 1.1)")
        end
        
        return r_hat
    end


    function plot_trace(mcmc_results, param_idx; param_names=nothing, save_tex=false, filename=nothing, thin=1)
        samples = mcmc_results["samples"]
        n_samples, n_params, n_chains = size(samples)
        
        if param_idx < 1 || param_idx > n_params
            error("Parameter index must be between 1 and $n_params")
        end
        
        # Validate thinning parameter
        if thin < 1
            error("Thinning factor must be a positive integer")
        end
        
        # Create thinned indices for samples
        thinned_indices = 1:thin:n_samples
        
        # Create reindexed values (1 to length of thinned indices)
        reindexed_values = 1:length(thinned_indices)
        
        colors = wong_colors[[1, 4, 3, 6]]
        alphas = [0.75, 0.75, 0.75, 0.75]  
        labels = ["chain 1", "chain 2", "chain 3", "chain 4"]
        
        # plot attributes
        plot_kwargs = (left_margin=4mm, right_margin=4mm, bottom_margin=4mm, top_margin=2mm, 
                    xtickfont=12, ytickfont=12, xguidefont=14, yguidefont=14, titlefont=14, 
                    grid=false, legendfont=12, foreground_color_legend=nothing, 
                    size=(600, 400), xformatter=x -> @sprintf("%.0f", x), legend=false)
        
        if isnothing(param_names)
            name = param_idx == 1 ? L"\theta_b" : L"\theta_w_{%$(param_idx-1)}"
        else
            name = param_names[param_idx]
        end
        
        # trace plot - using reindexed values for the regular plot
        plt = plot(xlabel="Sample", ylabel=name, legend=:outertopright; plot_kwargs...)
        for c in 1:n_chains
            # Use reindexed values instead of actual indices
            plot!(plt, reindexed_values, samples[thinned_indices, param_idx, c], 
                color=colors[c], alpha=alphas[c], label=labels[c], linewidth=1.5)
        end

        # save as TeX file
        if save_tex
            if isnothing(filename)
                param_name = param_idx == 1 ? "θb" : "θw_$(param_idx-1)"
                filename = "mcmc_trace_$(param_name).tex"
            end
            
            try
                pgfplotsx()
                
                # Calculate appropriate tick positions for reindexed data
                n_thinned = length(reindexed_values)
                tick_interval = ceil(Int, n_thinned / 5)  # Create ~5 ticks
                tick_positions = 0:tick_interval:n_thinned
                
                # Scale factor for display
                scale_factor = 1000  # To display as thousands
                
                p_pgf = plot(
                    xlabel="Sample", 
                    ylabel=name, 
                    legend=:outertopright, 
                    left_margin=4mm, 
                    right_margin=4mm, 
                    bottom_margin=4mm, 
                    top_margin=2mm,
                    xticks=tick_positions,  # Use calculated tick positions
                    xformatter=x -> @sprintf("%.1f", x/scale_factor),  # Format as thousands
                    xtickfont=12, 
                    ytickfont=12, 
                    xguidefont=14, 
                    yguidefont=14, 
                    titlefont=14, 
                    grid=false, 
                    legendfont=12, 
                    foreground_color_legend=nothing, 
                    size=(600, 400)
                )
                
                # Add notation for thousands
                plot!(p_pgf, xlabel="Sample ⋅ 10³")
                
                for c in 1:n_chains
                    # Use the reindexed values for PGFPlots output
                    plot!(p_pgf, reindexed_values, samples[thinned_indices, param_idx, c], 
                        color=colors[c], alpha=alphas[c], label=labels[c], linewidth=1.5)
                end
                
                output_path = RESULTS_PATH * "wq/posteriors/"
                savefig(p_pgf, output_path * filename)
                gr()
                println("Plot saved as $filename")
                
            catch e
                @warn "Failed to save as TEX file. Make sure PGFPlotsX is installed: $(e)"
                gr()
            end
        end

        return plt
    end


    function plot_cdf(mcmc_results, param_idx; param_names=nothing, θ_b=0.5, δ_b=0.1, show_prior=true, θ_w_lb=nothing, θ_w_ub=nothing, save_tex=false, filename=nothing)

        samples = mcmc_results["samples"]
        n_samples, n_params, n_chains = size(samples)

        if param_idx < 1 || param_idx > n_params
            error("Parameter index must be between 1 and $n_params")
        end

        colors = wong_colors[[1, 4, 3, 6]]
        alphas = [0.75, 0.75, 0.75, 0.75] 
        labels = ["chain 1", "chain 2", "chain 3", "chain 4"]

        # plot attributes
        plot_kwargs = (left_margin=4mm, right_margin=4mm, bottom_margin=4mm, top_margin=2mm, 
            xtickfont=12, ytickfont=12, xguidefont=14, yguidefont=14, titlefont=14, 
            grid=false, legendfont=12, foreground_color_legend=nothing, 
            size=(600, 400), xformatter=x -> @sprintf("%.2f", x), 
            yformatter=y -> @sprintf("%.1f", y))

        if isnothing(param_names)
            name = param_idx == 1 ? L"\theta_b" : L"\theta_w_{%$(param_idx-1)}"
        else
            name = param_names[param_idx]
        end

        # cdf plot
        plt = plot(xlabel=name, ylabel="Cumulative density", legend=:outertopright; plot_kwargs...)

        for c in 1:n_chains
            sorted_samples = sort(samples[:, param_idx, c])
            cdf_values = collect(1:length(sorted_samples)) ./ length(sorted_samples)
            plot!(plt, sorted_samples, cdf_values, color=colors[c], alpha=alphas[c], label=labels[c], linewidth=1.5) 
        end

        # add prior
        if show_prior
            all_samples = vcat([samples[:, param_idx, c] for c in 1:n_chains]...)
            min_x = minimum(all_samples)
            max_x = maximum(all_samples)
            x_prior = range(min_x, max_x, length=500)

            if param_idx == 1
                sd = abs(δ_b * θ_b)
                prior_dist = Normal(θ_b, sd)
                prior_cdf = cdf.(prior_dist, x_prior)
            else
                if isnothing(θ_w_lb) || isnothing(θ_w_ub)
                    θ_w_lb = min_x
                    θ_w_ub = max_x
                end
                prior_dist = Uniform(θ_w_lb, θ_w_ub)
                prior_cdf = cdf.(prior_dist, x_prior)
            end

            plot!(plt, x_prior, prior_cdf, color=:black, linestyle=:dash, linewidth=1.5, label="prior", alpha=1.0)
        end

        # save as TeX file
        if save_tex
            if isnothing(filename)
                param_name = param_idx == 1 ? "θb" : "θw_$(param_idx-1)"
                filename = "mcmc_cdf_$(param_name).tex"
            end

            try
                pgfplotsx()

                p_pgf = plot(xlabel=name, ylabel="Cumulative density", legend=:outertopright,
                        left_margin=4mm, right_margin=4mm, bottom_margin=4mm, top_margin=2mm, 
                        xtickfont=12, ytickfont=12, xguidefont=14, yguidefont=14, titlefont=14, 
                        grid=false, legendfont=12, foreground_color_legend=nothing, 
                        size=(600, 400), xformatter=x -> @sprintf("%.2f", x), 
                        yformatter=y -> @sprintf("%.1f", y))

                for c in 1:n_chains
                    # Original samples
                    sorted_samples = sort(samples[:, param_idx, c])
                    cdf_values = collect(1:length(sorted_samples)) ./ length(sorted_samples)
                    
                    # Downsample to approximately 100 points
                    # Create approximately 100 evenly spaced CDF values
                    target_cdf_values = range(0.01, 1.0, length=100)
                    downsampled_x = Float64[]
                    downsampled_y = Float64[]
                    
                    # Interpolate the samples to get values at these CDF points
                    for target_cdf in target_cdf_values
                        # Find closest index
                        idx = findmin(abs.(cdf_values .- target_cdf))[2]
                        push!(downsampled_x, sorted_samples[idx])
                        push!(downsampled_y, cdf_values[idx])
                    end
                    
                    # Plot downsampled data
                    plot!(p_pgf, downsampled_x, downsampled_y, color=colors[c], 
                        alpha=alphas[c], label=labels[c], linewidth=1.5) 
                end

                # add prior - also with 100 points
                if show_prior
                    all_samples = vcat([samples[:, param_idx, c] for c in 1:n_chains]...)
                    min_x = minimum(all_samples)
                    max_x = maximum(all_samples)
                    x_prior = range(min_x, max_x, length=100)  # Reduced to 100 points

                    if param_idx == 1
                        sd = abs(δ_b * θ_b)
                        prior_dist = Normal(θ_b, sd)
                        prior_cdf = cdf.(prior_dist, x_prior)
                    else
                        if isnothing(θ_w_lb) || isnothing(θ_w_ub)
                            θ_w_lb = min_x
                            θ_w_ub = max_x
                        end
                        prior_dist = Uniform(θ_w_lb, θ_w_ub)
                        prior_cdf = cdf.(prior_dist, x_prior)
                    end

                    plot!(p_pgf, x_prior, prior_cdf, color=:black, linestyle=:dash, 
                        linewidth=1.5, label="prior", alpha=1.0)
                end

                output_path = RESULTS_PATH * "wq/posteriors/"
                savefig(p_pgf, output_path * filename)
                gr()
                println("Plot saved as $filename")

            catch e
                @warn "Failed to save as TEX file. Make sure PGFPlotsX is installed: $(e)"
                gr()
            end
        end

        return plt
    end



    function plot_param_dist(mcmc_results, param_1, param_2; save_tex=false, filename=nothing, contours=false, use_pdf=true, θ_b=0.5, δ_b=0.1, show_prior=true, θ_w_lb=nothing, θ_w_ub=nothing, show_y_axis=false)

        samples = mcmc_results["samples"]
        n_samples, n_params, n_chains = size(samples)

        if param_1 < 1 || param_1 > n_params || param_2 < 1 || param_2 > n_params
            error("Parameter indices must be between 1 and $n_params")
        end

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

        color = wong_colors[6]

        # combine samples from all chains
        samples_1 = samples[:, param_1, 1]
        x_min = floor(minimum(samples_1) * 20) / 20
        x_max = ceil(maximum(samples_1) * 20) / 20
        x_min = iszero(x_min) && signbit(x_min) ? 0.0 : x_min  
        x_max = iszero(x_max) && signbit(x_max) ? 0.0 : x_max  

        samples_2 = samples[:, param_2, 1]
        y_min = floor(minimum(samples_2) * 20) / 20
        y_max = ceil(maximum(samples_2) * 20) / 20
        y_min = iszero(y_min) && signbit(y_min) ? 0.0 : y_min  
        y_max = iszero(y_max) && signbit(y_max) ? 0.0 : y_max  

        # Define plot attributes
        plot_kwargs = (
            size=(525, 400), right_margin=8mm, bottom_margin=2mm, top_margin=2mm, 
            xtickfont=12, ytickfont=12, xguidefont=14, yguidefont=14, legendfont=12, 
            foreground_color_legend=nothing, grid=false
        )

        if param_1 == param_2
            bin_edges = LinRange(x_min, x_max, 31)
            normalize_option = use_pdf ? :pdf : false

            # Use empty string for y-label if not showing y-axis
            y_label = show_y_axis ? (use_pdf ? "Density" : "Frequency") : ""

            # Create histogram plot
            p = histogram(
                samples_1, bins=bin_edges, color=color, alpha=0.75, 
                xlabel=label_1, ylabel=y_label, label="posterior", 
                legend=true, linecolor=:transparent, xlims=(x_min, x_max),
                normalize=normalize_option, left_margin=4mm; plot_kwargs...
            )

            # Hide y-axis ticks and numbers if show_y_axis is false
            if !show_y_axis
                plot!(p, yticks=nothing, yaxis=false)
            end

            if show_prior
                x_prior = range(x_min, x_max, length=200)

                if param_1 == 1
                    sd = abs(δ_b * θ_b)
                    prior_dist = Normal(θ_b, sd)
                    prior_pdf = pdf.(prior_dist, x_prior)

                    if !use_pdf
                        max_hist_height = maximum(fit(Histogram, samples_1, bin_edges).weights)
                        scaling_factor = max_hist_height / maximum(prior_pdf)
                        prior_pdf = prior_pdf .* scaling_factor
                    end
                    plot!(p, x_prior, prior_pdf, 
                    color=:black, 
                    linestyle=:dash, 
                    linewidth=1.5, 
                    label="prior", 
                    alpha=1.0)
                else
                    if isnothing(θ_w_lb) || isnothing(θ_w_ub)
                        θ_w_lb = x_min
                        θ_w_ub = x_max
                    end

                    prior_dist = Uniform(θ_w_lb, θ_w_ub)
                    prior_pdf = pdf.(prior_dist, x_prior)
                    plot!(p, x_prior, prior_pdf, color=:black, linestyle=:dash, linewidth=1.5, label="prior", alpha=1.0)
                end
            end
        else
            # For scatter plots, we maintain the y-axis since it represents a different parameter
            p = scatter(
                samples_1, samples_2, markersize=4, alpha=0.6, label=false, 
                color=color, markerstrokewidth=0, xlabel=label_1, ylabel=label_2, 
                xlims=(x_min, x_max), ylims=(y_min, y_max), left_margin=4mm; plot_kwargs...
            )

            if contours
                try
                    k = kde((samples_1, samples_2))
                    x_grid = range(x_min, x_max, length=100)
                    y_grid = range(y_min, y_max, length=100)
                    z = [pdf(k, x, y) for y in y_grid, x in x_grid]
                    contour!(
                    p, x_grid, y_grid, z, color=:white, linewidth=1.25, linestyle=:solid, 
                    label=false, zaxis=false
                    )
                catch e
                    @warn "Failed to create contour plot: $e"
                end
            end
        end

        # Handle TEX file saving with modified y-axis
        if save_tex
            if isnothing(filename)
                if param_1 == param_2
                    param_name = param_1 == 1 ? "θb" : "θw_$(param_1-1)"
                    filename = "mcmc_hist_$(param_name).tex"
                else
                    param1_name = param_1 == 1 ? "θb" : "θw_$(param_1-1)"
                    param2_name = param_2 == 1 ? "θb" : "θw_$(param_2-1)"
                    filename = "mcmc_scatter_$(param1_name)_v_$(param2_name).tex"
                end
            end

            try
                pgfplotsx()

                if param_1 == param_2
                    # histogram for TEX
                    y_label = show_y_axis ? (use_pdf ? "Density" : "Frequency") : ""
                    normalize_option = use_pdf ? :pdf : false

                    p_pgf = histogram(
                        samples_1, bins=bin_edges, 
                        color=color, alpha=0.75, 
                        xlabel=label_1, ylabel=y_label, 
                        legend=false, linecolor=:transparent, 
                        xlims=(x_min, x_max),
                        normalize=normalize_option,
                        size=(525, 400), 
                        right_margin=8mm, bottom_margin=2mm, top_margin=2mm,
                        left_margin=4mm,
                        xtickfont=12, ytickfont=12, xguidefont=14, yguidefont=14,
                        legendfont=12, foreground_color_legend=nothing, grid=false
                    )

                    # Apply the same y-axis visibility settings for the PGF plot
                    if !show_y_axis
                        plot!(p_pgf, yticks=nothing)  # Remove y-axis ticks and numbers
                        plot!(p_pgf, yaxis=false)     # Hide the y-axis line
                    end

                    if show_prior
                        x_prior = range(x_min, x_max, length=200)
                        
                        if param_1 == 1
                            sd = abs(δ_b * θ_b)
                            prior_dist = Normal(θ_b, sd)
                            prior_pdf = pdf.(prior_dist, x_prior)
                            if !use_pdf
                                max_hist_height = maximum(fit(Histogram, samples_1, bin_edges).weights)
                                scaling_factor = max_hist_height / maximum(prior_pdf)
                                prior_pdf = prior_pdf .* scaling_factor
                            end
                        else
                            if isnothing(θ_w_lb) || isnothing(θ_w_ub)
                                θ_w_lb = x_min
                                θ_w_ub = x_max
                            end
                            prior_dist = Uniform(θ_w_lb, θ_w_ub)
                            prior_pdf = pdf.(prior_dist, x_prior)
                        end
                        
                        plot!(p_pgf, x_prior, prior_pdf, color=:black, linestyle=:dash, linewidth=1.5, label=false, alpha=1.0)
                    end
                else
                    # scatter plot for TEX
                    p_pgf = scatter(
                    samples_1, samples_2, 
                    markersize=4, alpha=0.6, label=false, 
                    color=color, markerstrokewidth=0,
                    xlabel=label_1, ylabel=label_2, 
                    xlims=(x_min, x_max), ylims=(y_min, y_max),
                    size=(525, 400), 
                    right_margin=8mm, bottom_margin=2mm, top_margin=2mm,
                    left_margin=4mm,
                    xtickfont=12, ytickfont=12, xguidefont=14, yguidefont=14,
                    legendfont=12, foreground_color_legend=nothing, grid=false
                    )
                end

                output_path = RESULTS_PATH * "wq/posteriors/"
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




    function export_mcmc_samples(mcmc_results, param_idx; filename=nothing, use_chains=:all, thin_to=2000)

        # get all samples for the requested parameter across chains
        samples = mcmc_results["samples"]
        n_samples, n_params, n_chains = size(samples)

        # verify parameter index
        if param_idx < 1 || param_idx > n_params
            error("parameter index must be between 1 and $n_params")
        end

        # create parameter name for the file header
        param_name = param_idx == 1 ? "bulk" : "G$(param_idx-1)"

        # create default filename if not provided
        if isnothing(filename)
            filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_mcmc_samples_$(param_name).csv"
        end

        if use_chains == :all
            # combine all chains for this parameter
            all_samples = Vector{Float64}(undef, n_samples * n_chains)

            for chain in 1:n_chains
                start_idx = (chain - 1) * n_samples + 1
                end_idx = chain * n_samples
                all_samples[start_idx:end_idx] = samples[:, param_idx, chain]
            end

            # apply thinning if needed
            if thin_to > 0 && thin_to < length(all_samples)
                thin_interval = max(1, floor(Int, length(all_samples) / thin_to))
                thin_indices = 1:thin_interval:length(all_samples)

                if length(thin_indices) > thin_to
                thin_indices = thin_indices[1:thin_to]
                end

                data_to_export = all_samples[thin_indices]
            else
                data_to_export = all_samples
            end
        else
            # use specific chain
            if use_chains < 1 || use_chains > n_chains
                error("chain index must be between 1 and $n_chains")
            end

            chain_samples = samples[:, param_idx, use_chains]

            # apply thinning if needed
            if thin_to > 0 && thin_to < n_samples
                thin_interval = max(1, floor(Int, n_samples / thin_to))
                thin_indices = 1:thin_interval:n_samples

                if length(thin_indices) > thin_to
                    thin_indices = thin_indices[1:thin_to]
                end

                data_to_export = chain_samples[thin_indices]
            else
                data_to_export = chain_samples
            end
        end

        output_path = RESULTS_PATH * "wq/posteriors/"
        df = DataFrame()
        df[!, param_name] = data_to_export
        CSV.write(output_path * filename, df)


    end



    function calculate_mcmc_bounds(eki_samples; ub_cap=0.0)

        n_params = size(eki_samples, 2)
        eki_lbs = [quantile(eki_samples[:, j], 0.025) for j in 1:n_params]
        eki_ubs = [quantile(eki_samples[:, j], 0.975) for j in 1:n_params]
        mcmc_lb = zeros(n_params)
        mcmc_ub = zeros(n_params)

        interval = 0.05
        push_thresh = 0.025

        for j in 1:n_params
            lt_raw = eki_lbs[j]
            ut_raw = eki_ubs[j]
            
            lt_r = floor(lt_raw / interval) * interval
            ut_r = ceil(ut_raw / interval) * interval

            lt_adjusted = abs(lt_raw - lt_r) <= push_thresh ? lt_r - interval : lt_r
            ut_adjusted = abs(ut_raw - ut_r) <= push_thresh ? ut_r + interval : ut_r
            
            mcmc_lb[j] = lt_adjusted
            mcmc_ub[j] = min(ut_adjusted, ub_cap)

            if mcmc_lb[j] >= mcmc_ub[j]
                mcmc_lb[j] = mcmc_ub[j] - interval
            end
        end

        return mcmc_lb, mcmc_ub
    end


    
end