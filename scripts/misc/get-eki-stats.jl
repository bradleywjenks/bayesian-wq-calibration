using JLD2
using DataFrames
using Statistics
using Printf
using LaTeXStrings

const RESULTS_PATH = "/Users/bradwjenks/Code/PhD/bayesian-wq-calibration/results/wq/eki_calibration/"
data_period = 18
grouping = "material"
δ_b = 0.025
δ_s = 0.1

# load eki results
filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s)).jld2"
filepath = joinpath(RESULTS_PATH, filename)
eki_results = JLD2.load(filepath, "eki_results")

# compute ensemble means
n_ensemble = length(eki_results)
θ_samples = hcat([eki_results[i]["θ"] for i in 1:n_ensemble]...)'
parameter_means = mean(θ_samples, dims=1)[1, :]


