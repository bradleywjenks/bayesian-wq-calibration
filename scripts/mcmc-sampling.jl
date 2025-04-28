"""
This script performs MCMC sampling using the GP emulator trained from the ensemble Kalman inversion calibration. The following steps are performed:
    1. load eki results, GP model, and operational data (DONE)
    2. define prior, likelihood, and posterior functions
    3. implement Metropolis-Hastings MCMC sampler
    4. run MCMC to sample from posterior distribution
    5. analyze and visualize results
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

### 1. load eki results, GP model, and operational data ###
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





### 2a. define prior function ###

# bulk parameter
θ_samples = hcat([eki_results[i]["θ"] for i in 1:n_ensemble]...)'
θ_b = mean(θ_samples[:, 1])

# wall parameters
θ_w_lb, θ_w_ub = if grouping == "single"
    ([-1.0], [0.0])  # G1: all pipes
elseif grouping == "material"
    ([-1.0, -0.15], [0.0, 0.0])  # G1: metallic, G2: cement + plastic
elseif grouping == "material-age"
    ([-1.0, -1.0, -0.15], [0.0, 0.0, 0.0])  # G1: metallic + > mean pipe age, G2: metallic + ≤ mean pipe age, G3: cement + plastic
elseif grouping == "material-age-velocity"
    ([-1.0, -1.0, -1.0, -1.0, -0.15], [0.0, 0.0, 0.0, 0.0, 0.0])  # G1: metallic + > mean pipe age + ≤ mean velocity, G2: metallic + > mean pipe age + > mean velocity, G3: metallic + ≤ mean pipe age + ≤ mean velocity, G4: metallic + ≤ mean pipe age + > mean velocity, G5: cement + plastic
else
    error("Unsupported grouping: $grouping")
end

    
function log_prior_(θ)

    log_prob = 0.0
    
    # θ_b
    if θ[1] > 0 || θ[1] < -Inf
        return -1e5
    end
    bulk_mean = θ_b
    bulk_std = abs(θ_b * δ_b)
    bulk_prior = Normal(bulk_mean, bulk_std)
    log_prob += logpdf(bulk_prior, θ[1])

    # θ_w
    for (i, (lb, ub)) in enumerate(zip(θ_w_lb, θ_w_ub))
        if θ[i+1] < lb || θ[i+1] > ub
            return -1e5
        end
        log_prob += -log(ub - lb)
    end
    
    return log_prob
end



θ = [-0.55, -0.08, -0.15]
lp = log_prior_(θ)




### 2b. define likelihood function ###