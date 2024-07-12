# Key objective: Generate data for training the neural network
# Neural netowrk needs data extracted from the generated data
using PowerModelsADA, Ipopt, Random, BSON

path_to_code = "D:\\VSCode\\Julia\\Special_Problem"

for k in 2:10
    # Load the data
    data = BSON.load("$(path_to_code)\\data\\result_divided_$k.bson")

    # Determine the number of entries to extract
    num_extract = ceil(Int, length(data) * 0.1)

    # Randomly select 10% of the entries
    keys_to_extract = randperm(length(data))[1:num_extract]
    extracted_data = Dict([key => data[key] for key in keys_to_extract])

    # Save the extracted entries in a new file
    file_name = "$(path_to_code)\\data\\test_$k.bson"
    BSON.bson(file_name, extracted_data)
end

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
tol_value = 1e-4 # 1e-4 in real senario
max_iteration = 5000 # 5000 in real senario
alpha = 1000 # 1000 in real senario
areas_id = get_areas_id(data_base)

# saved_data = ["shared_variable", "received_variable", "dual_variable", "solution", "mismatch", "counter"]
saved_data = ["solution", "mismatch"]
# solve using ADMM and save data
number_runs = 10 # 1000 in real senario

demand_change = 0.5 # 50% change in demand in differet senarios
dataset = Dict()
result = Dict()
c = 1
k = 1
for run_id = 1: number_runs
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

    println("run_id: ", run_id)
    if run_id % 2 == 0
        file_name = "$(path_to_code)/result_divided_$k.bson"
        bson(file_name, dataset) # To save dataset into designated path
        global dataset = Dict()
        GC.gc()
        global k += 1
    end
end

# file_name = "$(path_to_code)test.bson"
# bson(file_name, dataset) # To save dataset into designated path
# data = BSON.load(file_name) # To load dataset from designated path
# BSON.jl is the web page
# Julia.pace.gatech