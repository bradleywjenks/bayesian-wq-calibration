using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using Distributions
using Statistics
using Random
using Plots

const EKP = EnsembleKalmanProcesses

# set random seed for reproducibility
rng_seed = 41
rng = Random.seed!(Random.GLOBAL_RNG, rng_seed)

# define simple forward model: f(θ) = θ₁ * x + θ₂ * x^2
function F(x, θ)
    y = θ[1] * x .+ θ[2] * x .^ 2
    return y
end

# true parameters
θ_true = [4, 1.2]
θ_n = length(θ_true)

# inputs
x = collect(range(1, 50, length=50))
y_n = length(F(x, θ_true))

# generate synthetic observations with noise
noise = 0.25
Γ = noise^2 * I(y_n)
δ = MvNormal(zeros(y_n), Γ)
ȳ = F(x, θ_true) .+ rand(δ)

# define prior distributions
θ_lb = [0.0, 0.0]
θ_ub = [10.0, 2.0]
prior_μ = (θ_ub + θ_lb) / 2
prior_σ = (θ_ub - θ_lb) / 8
# prior_1 = constrained_gaussian("θ_1", prior_μ[1], prior_σ[1], θ_lb[1], θ_ub[1])
# prior_2 = constrained_gaussian("θ_2", prior_μ[2], prior_σ[2], θ_lb[2], θ_ub[2])
# prior = combine_distributions([prior_1, prior_2])


prior_1 = ParameterDistribution(Parameterized(Uniform(θ_lb[1], θ_ub[1])), bounded(θ_lb[1], θ_ub[1]), "θ_1")
prior_2 = ParameterDistribution(Parameterized(Uniform(θ_lb[2], θ_ub[2])), bounded(θ_lb[2], θ_ub[2]), "θ_2")
prior = combine_distributions([prior_1, prior_2])


plot(prior)

# set up EKI
n_ensemble = 100
n_iter = 20

# create initial ensemble
ensemble_0 = EKP.construct_initial_ensemble(rng, prior, n_ensemble)

# create EKI process
process = nothing
process = EKP.EnsembleKalmanProcess(
    ensemble_0,
    ȳ,
    Γ,
    Inversion(); rng=rng,
)

# run EKI iterations
for i in 1:n_iter
    println("Iteration $i/$n_iter")
    θ_i = get_ϕ_final(prior, process)
    g_ens = hcat([F(x, θ_i[:, j]) for j in 1:n_ensemble]...)
    EKP.update_ensemble!(process, g_ens)
end

# get final ensemble
ensemble_n = get_ϕ_final(prior, process)

# plotting
begin
    p1 = histogram(
        ensemble_0[1, :], 
        label="Initial θ₁", 
        alpha=0.6, 
        bins=10,
        color=:red,
        normalize=true
    )
    histogram!(
        p1,
        ensemble_n[1, :], 
        label="Final θ₁", 
        alpha=0.6, 
        bins=10,
        color=:blue,
        normalize=true
    )
    vline!(p1, [θ_true[1]], label="True θ₁", linewidth=2, color=:black)
    xlabel!(p1, "θ₁")
    ylabel!(p1, "Density")
    title!(p1, "Parameter 1 Distribution")

    p2 = histogram(
        ensemble_0[2, :], 
        label="Initial θ₂", 
        alpha=0.6, 
        bins=10,
        color=:red,
        normalize=true
    )
    histogram!(
        p2,
        ensemble_n[2, :], 
        label="Final θ₂", 
        alpha=0.6, 
        bins=10,
        color=:blue,
        normalize=true
    )
    vline!(p2, [θ_true[2]], label="True θ₂", linewidth=2, color=:black)
    xlabel!(p2, "θ₂")
    ylabel!(p2, "Density")
    title!(p2, "Parameter 2 Distribution")

    # Combine plots side by side
    p = plot(p1, p2, layout=(1, 2), size=(900, 400))
end


    
    