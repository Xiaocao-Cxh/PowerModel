# 0;10;1c# Key objective: Generate data for training the neural network
# Neural netowrk needs data extracted from the generated data
using PowerModelsADA
using Ipopt
using BSON

path_to_code = "/storage/home/hcoda1/0/xcao306/LET_GO"

include("$(path_to_code)/vanilla_functions.jl")

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
tol_value = 1e-4 # 1e-4 in real senario
max_iteration = 5000 # 5000 in real senario
alpha = 1000 # 1000 in real senario
areas_id = get_areas_id(data_base)

saved_data = ["solution", "mismatch"]

demand_change = 0.5 # 50% change in demand in differet senarios
result = Dict()
dataset = Dict()
global c = 1
global k = 44
# solve using ADMM and save data
number_runs = 440 # 10*100 in real senario
for run_id = 431: number_runs
    ## data variation
    data_sample = deepcopy(data_base)
    change_demand!(data_sample, demand_change)

    global result = solve_dopf_admm(data_sample, model_type, optimizer; alpha = alpha, tol=tol, max_iteration=max_iteration, print_level=0, save_data=saved_data)

    itr = result[1]["counter"]["iteration"]-1
    for itr_id in 1:itr
        for area_id in areas_id
            dataset[c] = extract_data(result, saved_data, run_id, area_id, itr_id, tol_value)
            global c += 1
        end
    end
    println("Finish run number $run_id")
end

file_name = "$(path_to_code)/result_divided_$k.bson"
bson(file_name, dataset)