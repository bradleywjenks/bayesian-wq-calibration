"""
This script performs MCMC sampling using the GP emulator trained from the ensemble Kalman inversion calibration. The following steps are performed:
    1. load eki results, GP model, and operational data (DONE)
    2. define (a) prior, (b) likelihood, and (c) posterior functions (DONE)
    3. create Metropolis-Hastings MCMC algorithm (DONE)
    4. run MCMC to sample from posterior distribution (DONE)
    5. results plotting 
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
using ProgressMeter
using Distributed
addprocs(4)
@everywhere using Distributions, Random, ProgressMeter, PyCall, DataFrames 

pd = pyimport("pandas")
np = pyimport("numpy")
pickle = pyimport("pickle")
builtins = pyimport("builtins")
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

### 1. load eki results, GP model, operational data, and other MCMC parameters ###
data_period = 18 # (aug. 2024)
padded_period = lpad(data_period, 2, "0")
grouping = "material" # "single", "material", "material-age", "material-age-velocity"
δ_s = 0.2
δ_b = 0.025

# eki results
eki_results_path = RESULTS_PATH * "wq/eki_calibration/"
eki_filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s)).jld2"
eki_results = JLD2.load(eki_results_path * eki_filename, "eki_results")
bwfl_ids = [string(col) for col in propertynames(eki_results[1]["y_df"]) if col != :datetime]

# gp models
gp_models_path = RESULTS_PATH * "wq/gp_models/"
gp_filename_base = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))"
gp_dict = Dict{String, Dict{String, Any}}()
for sensor_id ∈ bwfl_ids
    base_filename = gp_filename_base * "_$(sensor_id)"
    model_file = gp_models_path * base_filename * "_model.pkl"
    scaler_file = gp_models_path * base_filename * "_scaler.pkl"
    
    if isfile(model_file) && isfile(scaler_file)
        py_model_file = builtins.open(model_file, "rb")
        gp_model = pickle.load(py_model_file)
        py_model_file.close()
        
        py_scaler_file = builtins.open(scaler_file, "rb")
        scaler = pickle.load(py_scaler_file)
        py_scaler_file.close()
        
        gp_dict[sensor_id] = Dict(
            "gp_model" => gp_model,
            "scaler" => scaler
        )
    else
        println("Missing GP model files for sensor $sensor_id")
    end
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
θ_b = mean(θ_samples[:, 1])

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
        if θ[1] > 0 || θ[1] < -Inf
            return -1e5
        end
        bulk_mean = $(θ_b)
        bulk_std = abs($(θ_b) * $(δ_b))
        bulk_prior = Normal(bulk_mean, bulk_std)
        lp += logpdf(bulk_prior, θ[1])
        for (i, (lb, ub)) in enumerate(zip($(θ_w_lb), $(θ_w_ub)))
            if θ[i+1] < lb || θ[i+1] > ub
                return -1e5
            end
            lp += -log(ub - lb)
        end
        return lp
    end
end




### 2b. define likelihood function ###

@everywhere begin
    function log_likelihood(θ, gp_model, scaler, y_obs)
        θ_scaled = scaler.transform(reshape(θ, 1, length(θ)))
        y_pred = vec(gp_model.predict(θ_scaled))
        residual = y_obs - y_pred
        variance = (y_obs .* $(δ_s)).^2
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
                ll += log_likelihood(θ, gp_model, scaler, y_obs)
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
    scaling_factors = [0.25, 0.5, 0.5]
    parallel = true
    n_samples = 50000
    θ_samples = hcat([eki_results[i]["θ"] for i in 1:n_ensemble]...)'
    θ_init = vec(mean(θ_samples, dims=1))
    θ_init_list = [θ_init .+ 0.01 .* randn(length(θ_init)) for _ in 1:4]
end

mcmc_results = run_mcmc(θ_init_list, θ_samples; n_samples=n_samples, scaling_factors=scaling_factors, parallel=parallel)

chain = 1
histogram(mcmc_results["samples"][:, 1, chain], bins=50, xlabel="θ_b", ylabel="Frequency", label="Chain $(chain)")
histogram(mcmc_results["samples"][:, 2, chain], bins=50, xlabel="θ_w1", ylabel="Frequency", label="Chain $(chain)")
histogram(mcmc_results["samples"][:, 3, chain], bins=50, xlabel="θ_w2", ylabel="Frequency", label="Chain $(chain)")
plot(mcmc_results["log_posteriors"][:, chain], label="Chain $(chain)", xlabel="Sample", ylabel="Log Posterior")

println(mcmc_results["acceptance_rates"])

r_hat = assess_mcmc_convergence(mcmc_results)







########## OTHER FUNCTIONS ##########

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