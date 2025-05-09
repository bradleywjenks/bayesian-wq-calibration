using JLD2
using DataFrames
using Statistics
using Printf
using LaTeXStrings

const RESULTS_PATH = "/Users/bradwjenks/Code/PhD/bayesian-wq-calibration/results/wq/gp_models/"
data_period = 18
grouping = "material-age"
δ_b = 0.025
δ_s = 0.2

bwfl_ids = ["BW2_1", "BW3", "BW5_1", "BW6", "BW9", "BW12"]

function load_gp_results(data_period, grouping, sensor_id, δ_b, δ_s)
    filename = "$(data_period)_$(grouping)_δb_$(string(δ_b))_δs_$(string(δ_s))_$(sensor_id).jld2"
    try
        results = JLD2.load(RESULTS_PATH * filename, "gp_results")
        println(results)
        return results
    catch e
        @warn "Failed to load $filename: $e"
        return nothing
    end
end

function compute_metrics(y_test, y_pred_μ; tolerance=0.01, decimal_places=3)
    y_test_flat = vec(y_test)
    y_pred_flat = vec(y_pred_μ)
    
    rmse = sqrt(mean((y_test_flat .- y_pred_flat).^2))
    
    ss_res = sum((y_test_flat .- y_pred_flat).^2)
    ss_tot = sum((y_test_flat .- mean(y_test_flat)).^2)
    r2 = 1 - (ss_res / ss_tot)
    
    maxae = maximum(abs.(y_test_flat .- y_pred_flat))
    mae = mean(abs.(y_test_flat .- y_pred_flat))

    abs_diff = abs.(y_test_flat .- y_pred_flat)
    within_tolerance = count(abs_diff .<= tolerance) / length(abs_diff) * 100
    
    return Dict(
        "rmse" => round(rmse, digits=decimal_places),
        "r2" => round(r2, digits=decimal_places),
        "maxae" => round(maxae, digits=decimal_places),
        "mae" => round(mae, digits=decimal_places),
        "within_$(tolerance)_mg_L" => round(within_tolerance, digits=decimal_places)
    )
end

function analyze_gp_performance()
    results_table = Dict()
    
    println("\nLoading and analyzing GP model results...")
    println("Data Period: $data_period")
    println("Grouping: $grouping")
    
    # Containers for combined data
    all_y_test = Vector{Float64}()
    all_y_pred = Vector{Float64}()
    
    # Store individual sensor metrics
    for sensor_id in bwfl_ids
        results = load_gp_results(data_period, grouping, sensor_id, δ_b, δ_s)
        
        if results !== nothing
            y_test = results["y_test"]
            y_pred_μ = results["y_pred_μ"]
            
            # Compute individual metrics
            metrics = compute_metrics(y_test, y_pred_μ)
            results_table[sensor_id] = metrics
            
            # Append to combined data
            append!(all_y_test, vec(y_test))
            append!(all_y_pred, vec(y_pred_μ))
            
            println("Processed sensor: $sensor_id")
        else
            println("Skipping sensor $sensor_id (results not available)")
        end
    end
    
    # Compute combined metrics
    combined_metrics = compute_metrics(all_y_test, all_y_pred)
    
    # Display individual sensor results
    println("\nGP-EPANET Performance Statistics for Data Period $data_period, Grouping: $grouping")
    println("=" ^ 100)
    println("SENSOR ID |   RMSE   |    R²    |   MAX AE  |    MAE   | % WITHIN ±0.01 mg/L")
    println("-" ^ 100)
    
    for sensor_id in bwfl_ids
        if haskey(results_table, sensor_id)
            metrics = results_table[sensor_id]
            @printf("%9s | %8.3f | %8.3f | %8.3f | %8.3f | %18.1f\n",
                    sensor_id, 
                    metrics["rmse"], 
                    metrics["r2"], 
                    metrics["maxae"],
                    metrics["mae"],
                    metrics["within_0.01_mg_L"])
        else
            @printf("%9s | %8s | %8s | %8s | %8s | %18s\n", 
                    sensor_id, "N/A", "N/A", "N/A", "N/A", "N/A")
        end
    end
    println("-" ^ 100)
    
    # Display combined metrics
    @printf("COMBINED  | %8.3f | %8.3f | %8.3f | %8.3f | %18.1f\n",
            combined_metrics["rmse"], 
            combined_metrics["r2"], 
            combined_metrics["maxae"],
            combined_metrics["mae"],
            combined_metrics["within_0.01_mg_L"])
    println("=" ^ 100)
    
    # Return both individual and combined results
    results_with_combined = Dict(
        "sensors" => results_table,
        "combined" => combined_metrics
    )
    
    return results_with_combined
end

# Execute the analysis
results = analyze_gp_performance()