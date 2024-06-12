# Get Data from result_divided_1-5.bson
using BSON
using PowerModels, PowerModelsADA
using Ipopt # C++ optimization solver

# Skeleton code
# How to get one sample of data from dataset
# To get the ouput data

path_to_data = "D:\\VSCode\\Julia\\Special_Problem"
num_area = 3
alpha = 1000
# This case is not very big bug it takes too long to converge
test_case = "pglib_opf_case39_epri.m"
case_path = joinpath(path_to_data, test_case)
data_base = parse_file(case_path)
# result_sample = solve_dopf_admm(data_base, ACPPowerModel, Ipopt.Optimizer; alpha = 1000, tol=1e-16, max_iteration=1, print_level=0, save_data=["solution", "mismatch","shared_variable", "received_variable", "dual_variable"])


# Shared variables with each area
shared_variable_ids = Dict()
# for area in keys(result_sample)
#     shared_variable_ids[area] = Dict()
#     for j in keys(result_sample[area]["shared_variable"])
#         shared_variable_ids[area][parse(Int,j)] = Dict()
#         for k in keys(result_sample[area]["shared_variable"][j])
#             shared_variable_ids[area][parse(Int,j)][k] = collect(keys(result_sample[area]["shared_variable"][j][k]))
#         end
#     end
# end


# k = 1 # Number of divided data (later we will loop through all data stored)
# Order of data in result_divided_$k.bson based on area_id

data = BSON.load("$(path_to_data)\\data\\result_divided_1.bson")
# From data_file to result_divided_10.bson
for k in 2:3
    data_file = BSON.load("$(path_to_data)\\data\\result_divided_$k.bson")
    # Indexing data_file from d1 + 1
    # Have a new data file because we want to re-index the data
    data_file_new = Dict()
    last_index = maximum(collect(keys(data)))
    c = 1
    for i in keys(data_file)
        data_file_new[c+last_index] = data_file[i]
        c += 1
    end
    # Merge d1 and data_file with merge(d1, data_file)
    merge!(data, data_file)
end

# push!(data, BSON.load("$(path_to_data)\\data\\result_divided_1.bson"))
# push!(data, BSON.load("$(path_to_data)\\data\\result_divided_2.bson"))
# push!(data, BSON.load("$(path_to_data)\\data\\result_divided_3.bson"))
# push!(data, BSON.load("$(path_to_data)\\data\\result_divided_4.bson"))
# push!(data, BSON.load("$(path_to_data)\\data\\result_divided_5.bson"))
# push!(data, BSON.load("$(path_to_data)\\data\\result_divided_6.bson"))
# push!(data, BSON.load("$(path_to_data)\\data\\result_divided_7.bson"))
# push!(data, BSON.load("$(path_to_data)\\data\\result_divided_8.bson"))
# push!(data, BSON.load("$(path_to_data)\\data\\result_divided_9.bson"))
# push!(data, BSON.load("$(path_to_data)\\data\\result_divided_10.bson"))
# for k in 1:10
#     file_data = BSON.load("$(path_to_data)\\data\\result_divided_$k.bson")
#     push!(data, file_data)
# end
# data = BSON.load("$(path_to_data)/result_divided_$k.bson")

data_ids = sort(collect(keys(data)))
max_itr_id = 5000 # maximum number of iterations in the data
max_run_id = Int64(length(data_ids)/(max_itr_id*num_area))

function get_sample_data(sample, features_vector)
    # Initialize a zero vector with the right size
    sample_data = zeros(length(features_vector))

    # Loop through the features
    for (i, feature) in enumerate(features_vector)
        # Extract the feature value from the sample
        # Assuming the sample is a dictionary and the feature is a key
        sample_data[i] = sample[feature]
    end

    return sample_data
end

# data_path: path to bson file that constains data
# featrues_vector: a vector constrains the features ids. Ex: [0, 2]
"function to read and arrange the data in a matrix format that is readiable to flux"
function read_arrange_data(data_path, features_vector)
    # 1. Read data
    data = BSON.load(data_path)
    data_ids = sort(collect(keys(data)))

    # 2. Get data parameters
    num_samples = length(data_ids)
    num_features = length(features_vector)

    # 3. Initialize a zero matrix with the right size
    data_matrix = zeros(num_samples, num_features)

    # 4. Loop through the number of samples
    for (i, id) in enumerate(data_ids)
        # 5. Call a helper function `get_sample_data` to get the sample data
        sample_data = get_sample_data(data[id], features_vector)

        # Store the sample data in the matrix
        data_matrix[i, :] = sample_data
    end

    return data_matrix
end


run_id = 1 # run index (later we will loop through all runs)
itr_idx = 1 # iteration index (later we will loop through all iterations) (should be equal to the number of iterations in the data multiplied by the number of scenarios)

# each row/sample: some of the 5 features of a one area, one iteration, and one run
# column/features: one of the 5 faetures
run_id_pointer = findfirst(i -> data[i]["run_id"] == run_id, data_ids) # pointer to the start of run_id data
run_id_pointer = 1 + max_itr_id*num_area*(run_id-1)

itr_start = data_ids[run_id_pointer+num_area*(itr_idx-1)] # start of a iteration
itr_input = Dict()
itr_label = Dict()
for a in 0:num_area-1
    itr_input[data[itr_start+a]["area_id"]] = data[itr_start+a]["input"]
    itr_label[data[itr_start+a]["area_id"]] = data[itr_start+a]["label"]
end

# label
label = Dict(i => itr_label[i] for i in keys(itr_input))

# 0 solution
solution = Dict(i => itr_input[i]["solution"] for i in keys(itr_input))


# construct a vector for solution sample
sample_solution = Dict(area => [value for variable_name in keys(solution[area]) for (variable_id,value) in solution[area][variable_name]] for area in [1,2,3])

# 1 shared_variable
shared_variable = Dict()
for area in keys(itr_input)
    shared_variable[area] = Dict() # Initialize shared_variable for areas_id
    for variable in ["qf", "va", "qt", "vm", "pf", "pt"]
        shared_variable[area][variable] = Dict()
        for idx in keys(itr_input[area]["solution"][variable])
            if idx in vcat([shared_variable_ids[area][area2][variable] for area2 in keys(shared_variable_ids[area])]...)
                shared_variable[area][variable][idx] = itr_input[area]["solution"][variable][idx]
            end
        end
    end
end
# 2 received_variable
# Received variables may have duplicate values from different areas

received_variable = Dict()
for area in keys(itr_input)
    received_variable[area] = Dict() # Initialize received_variable for areas_id
    for variable in ["qf", "va", "qt", "vm", "pf", "pt"]
        received_variable[area][variable] = Dict()
        for area2 in keys(shared_variable_ids[area])
            for idx in shared_variable_ids[area2][area][variable]
                received_variable[area][variable][idx] = itr_input[area2]["solution"][variable][idx]
            end
        end
    end
end


# construct a vector for solution sample
sample_receive_variable = Dict(area => [value for variable_name in keys(received_variable[area]) for (variable_id,value) in received_variable[area][variable_name]] for area in [1,2,3])

# 3 mismatch
mismatch = Dict(i => itr_input[i]["mismatch"] for i in keys(itr_input))

# 4 dual_variable
dual_variable = Dict()
itr_counter_area = -1
for area in keys(itr_input)
    dual_variable[area] = Dict() # Initialize dual_variable for areas_id
    for variable in ["qf", "va", "qt", "vm", "pf", "pt"]
        dual_variable[area][variable] = Dict()
        for area2 in keys(shared_variable_ids[area])
            for idx in shared_variable_ids[area2][area][variable]
                dual_variable[area][variable][idx] = 0
                for itr_counter in 1:itr_idx
                    itr_counter_start = data_ids[run_id_pointer+num_area*(itr_counter-1)]
                    for i in collect(1:num_area)
                        if data[itr_counter_start+(i-1)]["area_id"] == area
                            itr_counter_area = itr_counter_start+(i-1)
                        end
                    end
                    # May be improved
                    dual_variable[area][variable][idx] += alpha * data[itr_counter_area]["input"]["mismatch"]["$area2"][variable][idx]
                end
            end
        end
    end
end


sample_complete = Dict(area => [sample_solution[area_id]; sample_receive_variable[area_id]] for area in [1,2,3]) # add other featrues


##### arrange area sample
area_id = 1
number_featrues = length(sample_complete[area_id])
number_sample = 5 # number of lenth of loop
all_samples = zeros(number_sample, number_featrues)
all_samples[5,:] = sample_complete[area_id] # for a single area
## at the end of the loop