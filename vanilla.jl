# Key objective: Generate data for training the neural network
# Neural netowrk needs data extracted from the generated data
using PowerModelsADA
using Ipopt
using Plots
using BSON

path_to_code = "/storage/home/hcodal/0/xcao306/LET_GO"

include("$(path_to_code)/vanilla_functions.jl")

# loading test case
# Dataset comes from pglib-opf
case_path = joinpath(path_to_code,"/pglib_opf_case39_epri.m")
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

file_name = "$(path_to_code)/test.bson"
bson(file_name, dataset) # To save dataset into designated path
data = BSON.load(file_name) # To load dataset from designated path
# BSON.jl is the web page
# Julia.pace.gatech
