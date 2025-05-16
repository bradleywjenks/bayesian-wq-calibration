using CSV
using DataFrames
using Statistics
using Printf

const RESULTS_PATH = "/Users/bradwjenks/Code/PhD/bayesian-wq-calibration/results/"
data_period = 18
grouping = "material-age"
δ_b = 0.025
δ_s = 0.05

# determine parameter names based on grouping
param_names = if grouping == "material-age"
    ["bulk", "G1", "G2", "G3"]
elseif grouping == "material"
    ["bulk", "G1", "G2"]
else
    error("Unknown grouping: $grouping")
end

for param_name in param_names
    filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_mcmc_samples_$(param_name).csv"
    filepath = RESULTS_PATH * "wq/posteriors/" * filename
    
    # load the MCMC samples
    samples_df = CSV.read(filepath, DataFrame)
    samples = samples_df[:, 1]
    
    # compute statistics
    mean_val = mean(samples)
    std_val = std(samples)
    quantile_025 = quantile(samples, 0.025)
    quantile_975 = quantile(samples, 0.975)
    ci_width = quantile_975 - quantile_025
    
    # print results
    @printf("%-10s | %9.6f | %9.6f | %10.6f | %10.6f | %11.6f\n",
            param_name, mean_val, std_val, quantile_025, quantile_975, ci_width)
end