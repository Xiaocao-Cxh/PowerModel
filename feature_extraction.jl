# Get Data from result_divided_1-5.bson
using BSON
using PowerModels, PowerModelsADA
using Ipopt # C++ optimization solver

# Skeleton code
# How to get one sample of data from dataset
# To get the ouput data

path_to_data = "D:\\VSCode\\Julia\\Special_Problem"
data_list = [1, 2, 3, 4, 5] # from 1 to 10
test_case = "pglib_opf_case39_epri.m"

# k = 1 # Number of divided data (later we will loop through all data stored)
# Order of data in result_divided_$k.bson based on area_id

###########
# 1st part: load data from multiple files and merge them into one dictionary
###########
"""
Function to load data from multiple files and merge them into one dictionary
Arguments:
    path_to_data::String: path to the data files
    data_list::Vector{Int64}: list of data files to load e.g. [1, 2, 3, 4, 5]
Returns:
    data::Dict: merged data from all files
"""
function load_data(path_to_data::String, data_list::Vector{Int64})

    data = BSON.load("$(path_to_data)\\data\\result_divided_$(data_list[1]).bson")
    # From data_file to result_divided_10.bson
    for k in data_list
        if k != data_list[1]
            data_file = BSON.load("$(path_to_data)\\data\\result_divided_$k.bson")
            # Indexing data_file from d1 + 1
            # Have a new data file because we want to re-index the data
            data_file_new = Dict()
            last_index = maximum(collect(keys(data)))
            # Ensure that the index is ordered
            for i in eachindex(data_file)
                data_file_new[i+last_index] = data_file[i]
            end
            # Merge d1 and data_file with merge(d1, data_file)
            merge!(data, data_file_new)
        end
    end
    return data
end

###########
# 2nd part: arrange data into a dictionaries of features
###########

"function to get the shared variables names and ids for each area"
function get_shared_variable_ids(path_to_data, test_case)

    case_path = joinpath(path_to_data, test_case)
    data_base = parse_file(case_path)
    result_sample = solve_dopf_admm(data_base, ACPPowerModel, Ipopt.Optimizer; alpha = 1000, tol=1e-16, max_iteration=1, print_level=0, save_data=["solution", "mismatch","shared_variable", "received_variable", "dual_variable"])


    # Shared variables with each area
    shared_variable_ids = Dict()
    for area in keys(result_sample)
        shared_variable_ids[area] = Dict()
        for j in keys(result_sample[area]["shared_variable"])
            shared_variable_ids[area][parse(Int,j)] = Dict()
            for k in keys(result_sample[area]["shared_variable"][j])
                shared_variable_ids[area][parse(Int,j)][k] = collect(keys(result_sample[area]["shared_variable"][j][k]))
            end
        end
    end

    return shared_variable_ids
end


"""
Function to arrange the data into dictionaries of label and features. The label is converge and not converge from local data. The features are: solution, shared_variable, received_variable, mismatch, and dual_variable.
Arguments:
    data::Dict : dictionary contains data samples
    shared_variable_ids::Dict : dictionary contains shared variables ids
    num_area::Int64 : number of areas
    max_itr_id::Int64 : maximum number of iterations in the data
Returns:
    data_dict::Dict : dictionary constains the features (the dict has keys of feature_name => run_id => itr_id => area_id => variable_name => variable_id => feature_value)
"""

function get_data_dicts(data, shared_variable_ids,num_area=3, max_itr_id=5000)
    # get the ids of the runs in the data
    run_ids = sort(unique([data[i]["run_id"] for i in data_ids]))

    # Initialize dictionaries to store the features (each dict has keys of run_id => itr_id => area_id => variable_name => variable_id => feature_value) e.g. solution_dict[24][1500][1]["qf"]["27"] = -0.7868422377956684
    label_dict = Dict()
    solution_dict = Dict()
    shared_variable_dict = Dict()
    received_variable_dict = Dict()
    mismatch_dict = Dict()
    dual_variable_dict = Dict()

    for run_id in run_ids
        run_id_pointer = findfirst(i -> data[i]["run_id"] == run_id, data_ids) # pointer to the start of run_id data
        # run_id_pointer = 1 + max_itr_id*num_area*(run_id-1)

        label_dict[run_id] = Dict()
        solution_dict[run_id] = Dict()
        shared_variable_dict[run_id] = Dict()
        received_variable_dict[run_id] = Dict()
        mismatch_dict[run_id] = Dict()
        dual_variable_dict[run_id] = Dict()

        for itr_idx in 1:max_itr_id
            itr_start = data_ids[run_id_pointer+num_area*(itr_idx-1)] # start of a iteration
            itr_input = Dict()
            itr_label = Dict()
            # This is actually area - 1
            for area in 0:num_area-1
                itr_input[data[itr_start+area]["area_id"]] = data[itr_start+area]["input"]
                itr_label[data[itr_start+area]["area_id"]] = data[itr_start+area]["label"]
            end

            # label
            label_dict[run_id][itr_idx] = Dict(area => itr_label[area] for area in keys(itr_input))

            # 0 solution
            solution_dict[run_id][itr_idx] = Dict(area => itr_input[area]["solution"] for area in keys(itr_input))

            # 1 shared_variable
            shared_variable_dict[run_id][itr_idx] = Dict()
            for area in keys(itr_input)
                shared_variable_dict[run_id][itr_idx][area] = Dict() # Initialize shared_variable for areas_id
                for variable in ["qf", "va", "qt", "vm", "pf", "pt"]
                    shared_variable_dict[run_id][itr_idx][area][variable] = Dict()
                    for idx in keys(itr_input[area]["solution"][variable])
                        if idx in vcat([shared_variable_ids[area][area2][variable] for area2 in keys(shared_variable_ids[area])]...)
                            shared_variable_dict[run_id][itr_idx][area][variable][idx] = itr_input[area]["solution"][variable][idx]
                        end
                    end
                end
            end

            # 2 received_variable
            # Received variables may have duplicate values from different areas
            received_variable_dict[run_id][itr_idx] = Dict()
            for area in keys(itr_input)
                received_variable_dict[run_id][itr_idx][area] = Dict() # Initialize received_variable for areas_id
                for variable in ["qf", "va", "qt", "vm", "pf", "pt"]
                    received_variable_dict[run_id][itr_idx][area][variable] = Dict()
                    for area2 in keys(shared_variable_ids[area])
                        for idx in shared_variable_ids[area2][area][variable]
                            received_variable_dict[run_id][itr_idx][area][variable][idx] = itr_input[area2]["solution"][variable][idx]
                        end
                    end
                end
            end

            # 3 mismatch
            mismatch_dict[run_id][itr_idx] = Dict(area => itr_input[area]["mismatch"] for area in keys(itr_input))

            # 4 dual_variable
            dual_variable_dict[run_id][itr_idx] = Dict()
            itr_counter_area = -1
            for area in keys(itr_input)
                dual_variable_dict[run_id][itr_idx][area] = Dict() # Initialize dual_variable for areas_id
                for variable in ["qf", "va", "qt", "vm", "pf", "pt"]
                    dual_variable_dict[run_id][itr_idx][area][variable] = Dict()
                    for area2 in keys(shared_variable_ids[area])
                        for idx in shared_variable_ids[area2][area][variable]
                            dual_variable_dict[run_id][itr_idx][area][variable][idx] = 0
                            for itr_counter in 1:itr_idx
                                itr_counter_start = data_ids[run_id_pointer+num_area*(itr_counter-1)]
                                for i in collect(1:num_area)
                                    if data[itr_counter_start+(i-1)]["area_id"] == area
                                        itr_counter_area = itr_counter_start+(i-1)
                                    end
                                end
                                # May be improved
                                dual_variable_dict[run_id][itr_idx][area][variable][idx] += alpha * data[itr_counter_area]["input"]["mismatch"]["$area2"][variable][idx]
                            end
                        end
                    end
                end
            end
        end
    end
    data_dict = Dict("label" => label_dict, "solution" => solution_dict, "shared_variable" => shared_variable_dict, "received_variable" => received_variable_dict, "mismatch" => mismatch_dict, "dual_variable" => dual_variable_dict)

    return data_dict, run_ids
end

###########
# 3rd part: arrange data into a matrix format
###########

"""
function to arrange the data in a matrix format that is readiable to flux
    Arguments:
        data_dict::Dict : dictionary contains features and labels
        features_vector::Vector{Int64} : list of features to use. Ex: [0, 2]
            0: solution
            1: shared_variable
            2: received_variable
            3: mismatch
            4: dual_variable
    Returns:

    data_matrix::Dict{Int64, Matrix{Float64}} : dictionary contains labels (vector) and featrues (matrix) for each area. The dictionary arranges as area_id => "label" or "feature" => vector or matrix
"""
function get_data_matrix(data_dict, features_vector, run_ids)

    max_itr_id = 5000 # maximum number of iterations in the data
    label_vector = Dict()
    all_features_matrix = Dict()

    for area_id in 1:3

        # construct a vector of labels
        label_vector[area_id] = [data_dict["label"][run_id][itr_id][area_id] for run_id in run_ids for itr_id in 1:max_itr_id]

        # construct a matrix for solution
        if 0 in features_vector
            solution_matrix = [[data_dict["solution"][run_id][itr_id][area_id][variable_name][variable_id] for variable_name in keys(data_dict["solution"][run_id][itr_id][area_id]) for variable_id in keys(data_dict["solution"][run_id][itr_id][area_id][variable_name])] for run_id in run_ids for itr_id in 1:max_itr_id]

            # This may convert vector of vectors ( Vector{Vector{Float64}} ) into a matrix (Matrix{Float64})
            solution_matrix = vcat(solution_matrix...)
        else
            solution_matrix = []
        end
        # solution_matrix = [data_dict["solution"][run_id][itr_id][area_id][variable_name][variable_id] for run_id in run_ids for itr_id in 1:max_itr_id, for variable_name in keys(data_dict["solution"][run_id][itr_id][area_id]) for variable_id in keys(data_dict["solution"][run_id][itr_id][area_id][variable_name])]

        # construct a matrix for shared_variable
        if 1 in features_vector
            shared_variable_matrix = [[data_dict["shared_variable"][run_id][itr_id][area_id][variable_name][variable_id] for variable_name in keys(data_dict["shared_variable"][run_id][itr_id][area_id]) for variable_id in keys(data_dict["shared_variable"][run_id][itr_id][area_id][variable_name])] for run_id in run_ids for itr_id in 1:max_itr_id]
        else
            shared_variable_matrix = []
        end

        # construct a matrix for received_variable
        if 2 in features_vector
            received_variable_matrix = [[data_dict["received_variable"][run_id][itr_id][area_id][variable_name][variable_id] for variable_name in keys(data_dict["received_variable"][run_id][itr_id][area_id]) for variable_id in keys(data_dict["received_variable"][run_id][itr_id][area_id][variable_name])] for run_id in run_ids for itr_id in 1:max_itr_id]
        else
            received_variable_matrix = []
        end

        # construct a matrix for mismatch
        if 3 in features_vector
            mismatch_matrix = [[data_dict["mismatch"][run_id][itr_id][area_id][variable_name] for variable_name in keys(data_dict["mismatch"][run_id][itr_id][area_id])] for run_id in run_ids for itr_id in 1:max_itr_id]
        else
            mismatch_matrix = []
        end

        if 4 in features_vector
        # construct a matrix for dual_variable
            dual_variable_matrix = [[data_dict["dual_variable"][run_id][itr_id][area_id][variable_name][variable_id] for variable_name in keys(data_dict["dual_variable"][run_id][itr_id][area_id]) for variable_id in keys(data_dict["dual_variable"][run_id][itr_id][area_id][variable_name])] for run_id in run_ids for itr_id in 1:max_itr_id]
        else
            dual_variable_matrix = []
        end

        # construct a matrix for all features
        all_features_matrix[area_id] = hcat(solution_matrix, shared_variable_matrix, received_variable_matrix, mismatch_matrix, dual_variable_matrix)

    end

    return Dict("label" => label_vector, "feature" => all_features_matrix)
end

## Todo:
# 1. Test 3rd part is working: check vector of vectors construction, check vector of vectors conversion to matrix, check hcat of empty matrix, and the code in general (naming, commenting etc.)
# 2. Rewrite in 4th part functions get_dataset in NN_example.jl. Part 4 function should shuffle the data, split the data into training and testing and also based on label 0 and 1 balnced, and standardize the data.
###########
# 4th part: perpare data for training and testing
###########




























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
"""
function to arrange the data in a matrix format that is readiable to flux
    Arguments:
        data::Dict: dictionary constrains data samples
        features_vector::Vector{Int64}: list of features to use. Ex: [0, 2]
            0: solution
            1: shared_variable
            2: received_variable
            3: mismatch
            4: dual_variable
    Returns:
    data_matrix::Matrix{Float64}: matrix of data arranged in a matrix format
"""
function read_arrange_data(data, features_vector)
    # 1. Read the data ids and sort them
    data_ids = sort(collect(keys(data)))

    # 2. Get data parameters
    num_samples = length(data_ids)
    num_features = length(features_vector)

    areas
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