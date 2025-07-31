using DataFrames
using CSV
using Dates
using Statistics
using PyCall
using JLD2
using Colors

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




########## MAIN SCRIPT ##########

# load problem data
data_period = 18
padded_period = lpad(data_period, 2, "0")

flow_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-flow.csv", DataFrame); flow_df.datetime = DateTime.(flow_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
pressure_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-pressure.csv", DataFrame); pressure_df.datetime = DateTime.(pressure_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
wq_df = CSV.read(TIMESERIES_PATH * "/processed/" * padded_period * "-wq.csv", DataFrame); wq_df.datetime = DateTime.(wq_df.datetime, dateformat"yyyy-mm-dd HH:MM:SS")
cl_df = wq_df[wq_df.data_type .== "chlorine", :]

datetime = DateTime.(unique(cl_df.datetime))
bwfl_ids = unique(cl_df.bwfl_id)
datetime_test = datetime[(24 * 3 * 4)+1:end] # 5 test days (first is burn-in)

cl_df_test = subset(cl_df, :datetime => ByRow(in(datetime_test)))


# simulate cl concentrations
grouping = "material-age"
δ_s = 0.05
δ_b = 0.05


if δ_s == 0.2
    θ_mean = [-0.644, -0.077, -0.040, -0.121]
    θ_2_5 = [-0.550, -0.017, -0.016, -0.003]
    θ_97_5 = [-0.729, -0.121, -0.061, -0.340]
elseif δ_s == 0.05
    θ_mean = [-0.660, -0.091, -0.053, -0.052]
    θ_2_5 = [-0.639, -0.086, -0.047, -0.034]
    θ_97_5 = [-0.677, -0.097, -0.057, -0.071]
end

# run simulations for all three parameter sets
exclude_sensors = ["BW1", "BW4", "BW7", "BW5_2"]
burn_in = 24 * 4
demand_resolution = "wwmd"

y_mean = forward_model(
    epanet.build_model(df_2_pd(filter(row -> row.datetime ∈ datetime_test, flow_df)), 
                      df_2_pd(filter(row -> row.datetime ∈ datetime_test, pressure_df)), 
                      df_2_pd(filter(row -> row.datetime ∈ datetime_test, cl_df)),
                      sim_type="chlorine", demand_resolution=demand_resolution, bulk_coeff=θ_mean[1]),
    θ_mean, grouping, datetime_test, exclude_sensors; sim_type="chlorine", burn_in=burn_in)

y_2_5 = forward_model(
    epanet.build_model(df_2_pd(filter(row -> row.datetime ∈ datetime_test, flow_df)), 
                      df_2_pd(filter(row -> row.datetime ∈ datetime_test, pressure_df)), 
                      df_2_pd(filter(row -> row.datetime ∈ datetime_test, cl_df)),
                      sim_type="chlorine", demand_resolution=demand_resolution, bulk_coeff=θ_2_5[1]),
    θ_2_5, grouping, datetime_test, exclude_sensors; sim_type="chlorine", burn_in=burn_in)

y_97_5 = forward_model(
    epanet.build_model(df_2_pd(filter(row -> row.datetime ∈ datetime_test, flow_df)), 
                      df_2_pd(filter(row -> row.datetime ∈ datetime_test, pressure_df)), 
                      df_2_pd(filter(row -> row.datetime ∈ datetime_test, cl_df)),
                      sim_type="chlorine", demand_resolution=demand_resolution, bulk_coeff=θ_97_5[1]),
    θ_97_5, grouping, datetime_test, exclude_sensors; sim_type="chlorine", burn_in=burn_in)




# save meaurement and simulated data to csv
filename = "$(data_period)_$(grouping)_δb_$(δ_b)_δs_$(δ_s)_calibration_results.csv"
output_path = RESULTS_PATH * "wq/time_series/"

final_df = DataFrame(datetime = Dates.format.(datetime_test[burn_in + 1:end], "yyyy-mm-dd HH:MM:SS"))

for sensor_id in bwfl_ids
    if sensor_id ∉ exclude_sensors
        # measured data
        sensor_measured = filter(row -> row.bwfl_id == sensor_id && row.datetime in y_mean.datetime, cl_df_test)
        measured_values = [ismissing(sensor_measured[sensor_measured.datetime .== t, :mean]) || isempty(sensor_measured[sensor_measured.datetime .== t, :mean]) ? 
                          missing : sensor_measured[sensor_measured.datetime .== t, :mean][1] for t in y_mean.datetime]
        
        # simulated data
        final_df[!, "$(sensor_id)_measured"] = measured_values
        final_df[!, "$(sensor_id)_sim_mean"] = y_mean[!, Symbol(sensor_id)]
        final_df[!, "$(sensor_id)_sim_2.5"] = y_2_5[!, Symbol(sensor_id)]
        final_df[!, "$(sensor_id)_sim_97.5"] = y_97_5[!, Symbol(sensor_id)]
    end
end

CSV.write(output_path * filename, final_df)


# compute performance metrics for all sensor data
measured_cols = [col for col in names(final_df) if endswith(col, "_measured")]
simulated_cols = [col for col in names(final_df) if endswith(col, "_sim_mean")]
measured_stacked = vcat([final_df[!, col] for col in measured_cols]...)
simulated_stacked = vcat([final_df[!, col] for col in simulated_cols]...)

plot_df = DataFrame(measured = measured_stacked, simulated = simulated_stacked)
plot_df = dropmissing(plot_df)

# save 1:1 plot data
plot_filename = "$(data_period)_$(grouping)_δb_$(δ_b)_δs_$(δ_s)_1to1_plot.csv"
CSV.write(output_path * plot_filename, plot_df)

# compute performance metrics
residuals = plot_df.measured .- plot_df.simulated
rmse = sqrt(mean(residuals.^2))
abs_residuals = abs.(residuals)
pct_within_005 = (sum(abs_residuals .<= 0.05) / length(abs_residuals)) * 100
pct_within_01 = (sum(abs_residuals .<= 0.1) / length(abs_residuals)) * 100

# print results
println("Performance Metrics:")
println("===================")
println("Number of data points: $(nrow(plot_df))")
println("RMSE: $(round(rmse, digits=4))")
println("% residuals within 0.05: $(round(pct_within_005, digits=2))%")
println("% residuals within 0.1: $(round(pct_within_01, digits=2))%")






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
            insertcols!(y_df, 1, :datetime => datetime[burn_in + 1:end])
            for (old_name, new_name) in zip(Symbol.(sensor_model_id), Symbol.(sensor_bwfl_id))
                rename!(y_df, old_name => new_name)
            end
        else
            @error("Unsupported simulation type: $sim_type")
        end

        return y_df

    end
end
