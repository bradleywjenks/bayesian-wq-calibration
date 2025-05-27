using DataFrames
using CSV
using Dates
using Statistics
using PyCall
using JLD2

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
δ_s = 0.25
δ_b = 0.025


if δ_s == 0.25
    θ_mean = [-0.663, -0.104, -0.08, -0.011]
    θ_2_5 = [-0.64, -0.085, -0.064, -0.001]
    θ_97_5 = [-0.684, -0.117, -0.094, -0.049]
elseif δ_s == 0.1
    θ_mean = [-0.668, -0.096, -0.084, -0.003]
    θ_2_5 = [-0.662, -0.095, -0.069, -0.001]
    θ_97_5 = [-0.675, -0.097, -0.09, -0.004]
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





########## FUNCTIONS ##########

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
