"""
MCMC sampling using Turing.jl for GP emulator-based Bayesian calibration.
This script performs MCMC sampling using the GP emulator trained from the ensemble Kalman inversion calibration.

Steps:
1. Load EKI results, GP models, and operational data
2. Define Bayesian model using Turing.jl with priors and likelihood
3. Run MCMC sampling using NUTS sampler
4. Analyze and visualize results
"""

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
using JLD2
using LaTeXStrings
using Surrogates
using MLJ
using Printf

# Turing.jl for MCMC sampling
using Turing
using StatsPlots

# Optional: Add worker processes for distributed computing if needed
# using Distributed
# addprocs(4)
# @everywhere begin
#     using Turing, Distributions, LinearAlgebra, MLJ, DataFrames
#     using Surrogates, JLD2
# end


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

### 1. Load EKI results, GP models, operational data ###
data_period = 18 # (aug. 2024)
padded_period = lpad(data_period, 2, "0")
grouping = "material-age" # "single", "material", "material-age", "material-age-velocity"
δ_s = 0.05 # 0.05, 0.1, 0.2
δ_b = 0.05

# EKI results
eki_results_path = RESULTS_PATH * "wq/eki_calibration/"
eki_filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s)).jld2"
eki_results = JLD2.load(eki_results_path * eki_filename, "eki_results")
bwfl_ids = [string(col) for col in propertynames(eki_results[1]["y_df"]) if col != :datetime]

# GP models
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

# Operational data
wq_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-wq.csv", DataFrame)
wq_df.datetime = DateTime.(wq_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
y_df = subset(wq_df, 
    :data_type => ByRow(==("chlorine")), 
    :datetime => ByRow(in(Set(eki_results[1]["y_df"].datetime)))
)

# Parameter setup from EKI results
n_ensemble = length(eki_results)
θ_samples = hcat([eki_results[i]["θ"] for i in 1:n_ensemble]...)'
θ_b_train = mean(θ_samples[:, 1])
bulk_std = abs(θ_b_train * δ_b)

# Wall parameter priors from EKI ensemble (filtered for outliers, truncated at 0)
wall_priors = []
for j in 2:(size(θ_samples, 2))
    param_data = θ_samples[:, j]
    Q1, Q3 = quantile(param_data, [0.25, 0.75])
    IQR = Q3 - Q1
    filtered_data = param_data[param_data .>= Q1 - 1.5*IQR]
    
    μ, σ = mean(filtered_data), std(filtered_data)
    fitted_dist = Truncated(Normal(μ, σ), -Inf, 0.0)
    push!(wall_priors, fitted_dist)
end

n_wall_params = length(wall_priors)
n_params = 1 + n_wall_params

println("Number of parameters: $n_params (1 bulk + $n_wall_params wall)")
println("Bulk parameter prior: Normal($θ_b_train, $bulk_std)")
println("Wall parameter priors: $(length(wall_priors)) truncated normal distributions")

### 2. Define Turing.jl Bayesian Model ###

@model function water_quality_model(observations, sensor_data, gp_models, scalers, θ_b_mean, θ_b_std, wall_priors, σ_obs)
    
    # Bulk parameter prior (normal distribution)
    θ_b ~ Normal(θ_b_mean, θ_b_std)
    
    # Wall parameter priors (truncated normal distributions)
    θ_w = Vector{Float64}(undef, length(wall_priors))
    for i in 1:length(wall_priors)
        θ_w[i] ~ wall_priors[i]
    end
    
    # Combine parameters
    θ = vcat([θ_b], θ_w)
    
    # Likelihood calculation for each sensor
    for (sensor_id, sensor_obs) in observations
        if haskey(gp_models, sensor_id)
            gp_model = gp_models[sensor_id]
            scaler = scalers[sensor_id]
            valid_indices = sensor_data[sensor_id]["valid_indices"]
            
            if !isempty(valid_indices)
                # Scale input parameters
                X = reshape(θ, 1, :)
                X_table = DataFrame(X, :auto)
                X_scaled_table = MLJ.transform(scaler, X_table)
                X_scaled_mat = Matrix(X_scaled_table)
                X_scaled_vec = vec(X_scaled_mat)
                
                # Get GP predictions for valid time points
                y_pred = Vector{eltype(θ)}(undef, length(valid_indices))
                for (j, t_idx) in enumerate(valid_indices)
                    surrogate = gp_model[t_idx]
                    y_pred[j] = surrogate(X_scaled_vec)
                end
                
                # Add likelihood contribution using log probability
                sensor_obs_valid = sensor_obs[valid_indices]
                for (j, obs_val) in enumerate(sensor_obs_valid)
                    Turing.@addlogprob! logpdf(Normal(y_pred[j], σ_obs), obs_val)
                end
            end
        end
    end
end

### 3. Prepare data for Turing model ###

# Prepare observations and sensor data
observations = Dict{String, Vector{Float64}}()
sensor_data_dict = Dict{String, Dict{String, Any}}()

for sensor_id in bwfl_ids
    if haskey(gp_dict, sensor_id)
        # Get observations for this sensor
        sensor_df = subset(y_df, :bwfl_id => ByRow(==(sensor_id)))
        y_obs = sensor_df[!, :mean]
        
        # Find valid (non-missing) observations
        missing_mask = [!ismissing(val) for val in y_obs]
        valid_indices = findall(missing_mask)
        
        if !isempty(valid_indices)
            # Convert to Float64 array, replacing missing with NaN (though we won't use invalid indices)
            y_obs_clean = Float64[]
            for val in y_obs
                push!(y_obs_clean, ismissing(val) ? NaN : Float64(val))
            end
            
            observations[sensor_id] = y_obs_clean
            sensor_data_dict[sensor_id] = Dict("valid_indices" => valid_indices)
        end
    end
end

# Extract GP models and scalers
gp_models_dict = Dict(sensor_id => gp_dict[sensor_id]["gp_model"] for sensor_id in keys(observations))
scalers_dict = Dict(sensor_id => gp_dict[sensor_id]["scaler"] for sensor_id in keys(observations))



### 4. Run MCMC sampling with Turing.jl ###

# Create the model instance
model = water_quality_model(
    observations, 
    sensor_data_dict,
    gp_models_dict, 
    scalers_dict,
    θ_b_train, 
    bulk_std, 
    wall_priors, 
    δ_s
)

# MCMC sampling parameters
n_samples = 10000
n_chains = 4
target_accept = 0.65

# assign sampler
sampler_name = "nuts"

if sampler_name == "nuts"
    sampler = NUTS(target_accept)
elseif sampler_name == "mh"
    sampler = MH()
end

# run MCMC sampling with threading
chains = Turing.sample(model, sampler, MCMCThreads(), n_samples, n_chains; progress=true, drop_warmup=true)





### 5. Results plotting ###

# plots
p_trace = plot(chains)
p_density = density(chains)
p_corner = corner(chains, compact=true)
p_autocor = autocorplot(chains)



### 6. Save results ###

output_dir = RESULTS_PATH * "wq/posteriors/"

# Save summary statistics and quantiles as JLD2
chain_quantiles = quantile(chains)
summary_stats = summarystats(chains)

# Extract acceptance rates for each chain
chains_df = DataFrame(chains)
acceptance_rates = Dict{Int, Float64}()

for chain_num in 1:n_chains
    chain_data = subset(chains_df, :chain => ByRow(==(chain_num)))
    chain_accept_rate = mean(chain_data[!, :acceptance_rate])
    acceptance_rates[chain_num] = chain_accept_rate
end

summary_filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_mcmc_summary_$(sampler_name).jld2"
JLD2.save(output_dir * summary_filename, 
    "summary_stats", summary_stats, 
    "quantiles", chain_quantiles,
    "acceptance_rates", acceptance_rates)

# extract parameter samples
chains_df = DataFrame(chains)
param_columns = filter(name -> startswith(string(name), "θ"), names(chains_df))

for param_col in param_columns 
    if param_col == "θ_b"
        param_name = "B"
    else
        group_match = match(r"θ_w\[(\d+)\]", string(param_col))
        param_name = group_match !== nothing ? "G$(group_match.captures[1])" : string(param_col)
    end
    param_df = DataFrame()
    param_df[!, :chain] = chains_df[!, :chain]
    param_df[!, Symbol(param_name)] = chains_df[!, param_col]
    param_filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_mcmc_samples_$(param_name).csv"
    CSV.write(output_dir * param_filename, param_df)
end

### 7. Load summary statistics and quantiles ###

# δ_s = 0.1
summary_filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_mcmc_summary_$(sampler_name).jld2"

loaded_data = JLD2.load(output_dir * summary_filename)
loaded_summary = loaded_data["summary_stats"]
loaded_quantiles = loaded_data["quantiles"]
loaded_acceptance_rates = loaded_data["acceptance_rates"]

# Plot posterior histograms with priors
plots_list = []

# Plot bulk parameter (B)
bulk_csv = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_mcmc_samples_B.csv"
bulk_data = CSV.read(output_dir * bulk_csv, DataFrame)
bulk_samples = bulk_data[!, :B]

p_bulk = histogram(bulk_samples, bins=50, alpha=0.7, color=wong_colors[6], 
                   label="Posterior", title="θ_b (Bulk Parameter)", 
                   xlabel="Value", ylabel="Density", normalize=:pdf)

# Add prior (Normal distribution)
bulk_prior = Normal(θ_b_train, bulk_std)
x_range = range(minimum(bulk_samples) - 0.1*std(bulk_samples), 
                maximum(bulk_samples) + 0.1*std(bulk_samples), length=200)
plot!(p_bulk, x_range, pdf.(bulk_prior, x_range), 
      line=(:black, :dash, 2), label="Prior")

push!(plots_list, p_bulk)

# Plot wall parameters (G1, G2, etc.)
for (i, wall_prior) in enumerate(wall_priors)
    param_csv = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_mcmc_samples_G$(i).csv"
    param_data = CSV.read(output_dir * param_csv, DataFrame)
    param_samples = param_data[!, Symbol("G$(i)")]
    
    p_wall = histogram(param_samples, bins=50, alpha=0.7, color=wong_colors[6], 
                       label="Posterior", title="θ_w[$(i)] (Wall Parameter G$(i))", 
                       xlabel="Value", ylabel="Density", normalize=:pdf)
    
    # Add prior (Truncated Normal distribution)
    x_range = range(minimum(param_samples) - 0.1*std(param_samples), 
                    maximum(param_samples) + 0.1*std(param_samples), length=200)
    plot!(p_wall, x_range, pdf.(wall_prior, x_range), 
          line=(:black, :dash, 2), label="Prior")
    
    push!(plots_list, p_wall)
end

# Create combined plot
combined_plot = plot(plots_list..., layout=(length(plots_list), 1), size=(800, 300*length(plots_list)))
display(combined_plot)


