# Key objective: Generate data for training the neural network
# Neural netowrk needs data extracted from the generated data
using PowerModelsADA
using Ipopt
using Plots
using BSON

path_to_code = "D:\\VSCode\\Julia\\"

include("$(path_to_code)\\vanilla_functions.jl")

# loading test case
# Dataset comes from pglib-opf
case_path = joinpath(path_to_code,"pglib_opf_case39_epri.m")
data_base = parse_file(case_path)

# define parameters
model_type = ACPPowerModel #DCPPowerModel
optimizer = Ipopt.Optimizer
# Actual tolerance value so that we don't terminte in designated iterations
tol = 1e-16 # 1e-16 in real senario
# Tolerance value we want for labeling the data
tol_value = 1e-3 # 1e-4 in real senario
max_iteration = 2000 # 5000 in real senario
alpha = 1000 # 1000 in real senario
areas_id = get_areas_id(data_base)

saved_data = ["shared_variable", "received_variable", "dual_variable", "solution", "mismatch", "counter"]
# solve using ADMM and save data
number_runs = 2 # 1000 in real senario

demand_change = 0.5 # 50% change in demand in differet senarios
result = Dict()
dataset = Dict()
c = 1
data_sample = deepcopy(data_base)

for run_id = 1: number_runs
    ## data variation
    data_sample = deepcopy(data_base)
    change_demand!(data_sample, demand_change)

    result = solve_dopf_admm(data_sample, model_type, optimizer; alpha = alpha, tol=tol, max_iteration=max_iteration, print_level=0, save_data=saved_data)

    itr = result[1]["counter"]["iteration"]-1
    for itr_id in 1:itr
        for area_id in areas_id
            dataset[c] = extract_data(result, saved_data, run_id, area_id, itr_id, tol_value)
            c += 1
        end
    end
end

file_name = "$(path_to_code)test.bson"
bson(file_name, dataset) # To save dataset into designated path
data = BSON.load(file_name) # To load dataset from designated path
# BSON.jl is the web page
# Julia.pace.gatech

# sample_number = 5
# dataset[sample_number]["input"]
# dataset[sample_number]["label"]
# dataset[sample_number]["area_id"]

# Extract and plot mismatch data
# [result[1]["previous_solution"]["mismatch"][k]["1"] for k in 1:100]
# x = Dict()
# area_num = length(areas_id)
# a = plot()
# for run in 1:number_runs
#     itr = result[run][1]["counter"]["iteration"] - 1
#     for area in 1:area_num
#         x[(run, area)] = [result[run][area]["previous_solution"]["mismatch"][k][string(area)] for k in 1:itr]
#     end
#     xa = sqrt.(sum(x[(run, area)].^2) for area in 1:area_num)
#     plot!(xa, label="Global", linewidth=3, yticks = [ 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0], legend=:topright, xlabel="Iteration", ylabel="l_2 norm", linecolor=:lightseagreen, title="IEEE case39 with 3 Areas")
#     for area in 1:area_num
#         plot!(a, x[(run, area)], yaxis=:log, label="Run"*string(run)*"-"*"Area"*string(area))
#     end
# end
# display(a)

# itr = result[1]["counter"]["iteration"] - 1
# for area in 1:area_num
#     x[area] = [result[area]["previous_solution"]["mismatch"][k][string(area)] for k in 1:itr]
# end
# xa = sqrt.(sum(x[area].^2 for area in 1:area_num))
# ax= plot(1:itr, xa, label="Global", linewidth=3, yaxis=:log, yticks = [ 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0], legend=:topright, xlabel="Iteration", ylabel="l_2 norm", linecolor=:lightseagreen, title="IEEE case39 with 3 Areas")
# for area in 1:area_num
#     plot(1:itr, x[area], yaxis=:log, label="Run"*string(1)*"-"*"Area"*string(area))
# end

# area = 1
# plot!(ax, 1:itr, x[1], yaxis=:log, label="Run"*string(1)*"-"*"Area"*string(area))

# # If the next iteration is greater than the convergence_index
#     # It is not the real convergence and all should be labeled as 0

#     # If the next iteration is less than the convergence_index
#         # If we can find the next peak value and is smaller than tol_value
#         # We found the real convergence and label datapoints based upon it
#         # Else we do not have enough information to confirm the convergence and should ask for longer iterationsd
#         function get_label(result, itr_id, area_id, tol_value)
#             mismatches = [result[area_id]["previous_solution"]["mismatch"][k][string(area_id)] for k in 1:itr_id]

#             # Find the last index where the mismatch drops below the tol_value
#             convergence_index = findlast(x -> x < tol_value, mismatches)

#             # Create a peak value list
#             peak_index = []
#             if 1 < itr_id < length(mismatches) && mismatches[itr_id] > mismatches[itr_id-1] && mismatches[itr_id] > mismatches[itr_id+1]
#                 push!(peak_index, itr_id)
#             end

#             if (isnothing(convergence_index) || convergence_index == max_iteration ||
#                 mismatches[convergence_index+1] > mismatches[convergence_index])
#                 return 0
#             else
#                 # If we can find a peak index greater than convergence_index and its value is smaller than tol_value
#                 for peak in peak_index
#                     if peak > convergence_index && mismatches[peak] < tol_value
#                         return 1
#                     end
#                 end
#                 # Else we do not have enough information to confirm the convergence and should ask for longer iterations
#                 return 0
#             end
#         end