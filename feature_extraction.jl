# Get Data from result_test_1-5.bson
using Random, LinearAlgebra, Statistics, Printf # These are the standard libraries
using BSON, PowerModels, PowerModelsADA, StatsBase, Flux, Ipopt # C++ optimization solver
# Skeleton code
# How to get one sample of data from dataset
# To get the ouput data

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
    # From data_file to result_test_10.bson
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
    alpha::Int64 : alpha value for dual variable calculation
Returns:
    data_dict::Dict : dictionary constains the features (the dict has keys of feature_name => run_id => itr_id => area_id => variable_name => variable_id => feature_value)
"""
function get_data_dicts(data, shared_variable_ids, num_area=3, max_itr_id=5000, alpha=1000)
    # get the ids of the runs in the data

    data_ids = sort(collect(keys(data)))
    run_ids = sort(unique([data[i]["run_id"] for i in data_ids]))

    # Precompute the starting index for each run_id
    run_id_to_index = Dict()
    for run_id in run_ids
        run_id_to_index[run_id] = findfirst(i -> data[i]["run_id"] == run_id, data_ids)
    end

    # Initialize dictionaries to store the features (each dict has keys of run_id => itr_id => area_id => variable_name => variable_id => feature_value) e.g. solution_dict[24][1500][1]["qf"]["27"] = -0.7868422377956684
    label_dict = Dict()
    solution_dict = Dict()
    shared_variable_dict = Dict()
    received_variable_dict = Dict()
    mismatch_dict = Dict()
    dual_variable_dict = Dict()

    for run_id in run_ids
        println("Processing run_id: ", run_id)
        run_id_pointer = run_id_to_index[run_id] # pointer to the start of run_id data
        # run_id_pointer = 1 + max_itr_id*num_area*(run_id-1)
        println("Starting index for run_id ", run_id, ": ", run_id_to_index[run_id])

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

            # possible optimization: calls keys(itr_input) multiple times
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
            # y[k] = y[k-1] + alpha * mismatch[k]
            # not y[k] = alpha * (mismatch[1] + mismatch[2] + ... + mismatch[k])
            ## Todo: Suggestion for implementation

            dual_variable_dict[run_id][itr_idx] = Dict()
            itr_counter_area = -1
            for area in keys(itr_input)
                dual_variable_dict[run_id][itr_idx][area] = Dict() # Initialize dual_variable for areas_id
                for variable in ["qf", "va", "qt", "vm", "pf", "pt"]
                    dual_variable_dict[run_id][itr_idx][area][variable] = Dict()
                    for area2 in keys(shared_variable_ids[area])
                        for idx in shared_variable_ids[area2][area][variable]
                            # dual_variable_dict[run_id][itr_idx][area][variable][idx] = 0
                            itr_counter_start = data_ids[run_id_pointer+num_area*(itr_idx-1)]
                            for i in collect(1:num_area)
                                if data[itr_counter_start+(i-1)]["area_id"] == area
                                    itr_counter_area = itr_counter_start+(i-1)
                                end
                            end
                            if itr_idx == 1
                                dual_variable_dict[run_id][itr_idx][area][variable][idx] = alpha * data[itr_counter_area]["input"]["mismatch"]["$area2"][variable][idx]
                            else
                                dual_variable_dict[run_id][itr_idx][area][variable][idx] = dual_variable_dict[run_id][itr_idx-1][area][variable][idx] + alpha * data[itr_counter_area]["input"]["mismatch"]["$area2"][variable][idx]
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
            solution_matrix_ = [[data_dict["solution"][run_id][itr_id][area_id][variable_name][variable_id] for variable_name in keys(data_dict["solution"][run_id][itr_id][area_id]) for variable_id in keys(data_dict["solution"][run_id][itr_id][area_id][variable_name])] for run_id in run_ids for itr_id in 1:max_itr_id]
            solution_matrix = hcat(solution_matrix_...)
            println("solution_matrix: ", size(solution_matrix))
        end

        # construct a matrix for shared_variable
        if 1 in features_vector
            shared_variable_matrix_ = [[data_dict["shared_variable"][run_id][itr_id][area_id][variable_name][variable_id] for variable_name in keys(data_dict["shared_variable"][run_id][itr_id][area_id]) for variable_id in keys(data_dict["shared_variable"][run_id][itr_id][area_id][variable_name])] for run_id in run_ids for itr_id in 1:max_itr_id]
            shared_variable_matrix = hcat(shared_variable_matrix_...)
            println("shared_variable_matrix: ", size(shared_variable_matrix))
        end

        # construct a matrix for received_variable
        if 2 in features_vector
            received_variable_matrix_ = [[data_dict["received_variable"][run_id][itr_id][area_id][variable_name][variable_id] for variable_name in keys(data_dict["received_variable"][run_id][itr_id][area_id]) for variable_id in keys(data_dict["received_variable"][run_id][itr_id][area_id][variable_name])] for run_id in run_ids for itr_id in 1:max_itr_id]
            received_variable_matrix = hcat(received_variable_matrix_...)
            println("received_variable_matrix: ", size(received_variable_matrix))
        end

        # construct a matrix for mismatch
        if 3 in features_vector
            mismatch_matrix_ = [[data_dict["shared_variable"][run_id][itr_id][area_id][variable_name][variable_id] - data_dict["received_variable"][run_id][itr_id][area_id][variable_name][variable_id] for variable_name in keys(data_dict["shared_variable"][run_id][itr_id][area_id]) for variable_id in keys(data_dict["shared_variable"][run_id][itr_id][area_id][variable_name])] for run_id in run_ids for itr_id in 1:max_itr_id]
            mismatch_matrix = hcat(mismatch_matrix_...)
            println("mismatch_matrix: ", size(mismatch_matrix))
        end

        # construct a matrix for dual_variable
        if 4 in features_vector
            dual_variable_matrix_ = [[data_dict["dual_variable"][run_id][itr_id][area_id][variable_name][variable_id] for variable_name in keys(data_dict["dual_variable"][run_id][itr_id][area_id]) for variable_id in keys(data_dict["dual_variable"][run_id][itr_id][area_id][variable_name])] for run_id in run_ids for itr_id in 1:max_itr_id]
            dual_variable_matrix = hcat(dual_variable_matrix_...)
            println("dual_variable_matrix: ", size(dual_variable_matrix))
        end

        # construct a matrix for all features
        all_features_matrix[area_id] = vcat(solution_matrix, shared_variable_matrix, received_variable_matrix, mismatch_matrix, dual_variable_matrix)
        println("all_features_matrix: ", size(all_features_matrix[area_id]))
    end

    return Dict("label" => label_vector, "feature" => all_features_matrix)
end

###########
# 4th part: perpare data for training and testing
###########

"""
Function to split the data into training and testing, shuffle the data, and standardize the data
Arguments:
    data_matrix::Dict : dictionary contains data samples
    train_percent::Float64 : percentage of data to use for training
    standardize::String : type of standardization to use
    stats_dics::Dict : dictionary contains statistics for standardization
Returns:
    train_data::Dict : dictionary contains training data
    test_data::Dict : dictionary contains testing data
"""

function get_dataset(data_matrix::Dict, train_percent::Float64=0.8)

    areas_id = keys(data_matrix["feature"]) # Get unique area ids
    data_area = Dict()
    for area in areas_id # Get data for each area, it's the reverse of data_matrix
        data_area[area] = Dict("label" => data_matrix["label"][area], "feature" =>  data_matrix["feature"][area])
    end

    # Should be arranged data for each area
    data_arranged = Dict(area => Dict() for area in areas_id)

    not_converge_ids = Dict(area => [] for area in areas_id)
    converge_ids = Dict(area => [] for area in areas_id)
    for area in areas_id
        # X is a vector with each entery is another vector of the input data of the NN+
        # Here, we use X1 for inputs with output label 1 (for example, not converged) and X2 for inputs with output label 2 (for example, converged).

        not_converge_ids[area] = findall(x -> x == 0, data_area[area]["label"])
        converge_ids[area] = findall(x -> x == 1, data_area[area]["label"])

        # Perpare output data (labeling)
        # Shaffling, partitioning data into training, and testing
        train_length_not_converge = Int(ceil(train_percent*length(not_converge_ids[area])))
        # train_ids = Int.(sample(not_converge_ids[area], train_length_not_converge, replace=false))
        train_ids = rand(1:train_length_not_converge, Int(ceil(train_length_not_converge*train_percent)))
        test_ids = setdiff(1:train_length_not_converge, train_ids)
        Xtrain_1 = data_area[area]["feature"][:,not_converge_ids[area][train_ids]]
        Xtest_1 = data_area[area]["feature"][:,not_converge_ids[area][test_ids]]
        ytrain_1 = data_area[area]["label"][not_converge_ids[area][train_ids]]
        ytest_1 = data_area[area]["label"][not_converge_ids[area][test_ids]]

        train_length_converge = Int(ceil(train_percent*length(converge_ids[area])))
        # Shaffling
        train_ids = sample(converge_ids[area], train_length_converge, replace=false)
        test_ids = setdiff(converge_ids[area], train_ids)
        Xtrain_2 = converge_ids[area][:,train_ids]
        Xtest_2 = converge_ids[area][:,test_ids]
        ytrain_2 = 1 .* ones(1,size(Xtrain_2)[2])
        ytest_2 = 1 .* ones(1,size(Xtest_2)[2])

        # Make sure label with 0 and 1 are balanced, maybe 50% of each
        Xtrain = hcat(Xtrain_1, Xtrain_2)
        ytrain = hcat(ytrain_1, ytrain_2)
        Xtest = hcat(Xtest_1, Xtest_2)
        ytest = hcat(ytest_1, ytest_2)

        # Shuffling
        shuf_1 = shuffle(1:size(Xtrain)[2])
        shuf_2 = shuffle(1:size(Xtest)[2])

        Xtrain = Xtrain[:, shuf_1]
        ytrain = ytrain[:, shuf_1]
        Xtest = Xtest[:, shuf_2]
        ytest = ytest[:, shuf_2]

        data_arranged[area] = Dict("Xtrain" => Xtrain, "ytrain" => ytrain, "Xtest" => Xtest, "ytest" => ytest)
    end
    return data_arranged
end

function normalize_arranged_data(data_arranged, normalizatoin_method; stats_dict=Dict())
    for area in keys(data_arranged)
        if normalizatoin_method == "standardize"
            stats_dict[area] = Dict()
            stats_dict[area]["mean"] = mean(data_arranged[area]["Xtrain"], dims=2)
            stats_dict[area]["std"] = std(data_arranged[area]["Xtrain"], dims=2)
            data_arranged[area]["Xtrain"] = (data_arranged[area]["Xtrain"] .- stats_dict[area]["mean"]) ./ stats_dict[area]["std"]
            data_arranged[area]["Xtest"] = (data_arranged[area]["Xtest"] .- stats_dict[area]["mean"]) ./ stats_dict[area]["std"]
        elseif normalizatoin_method == "minmax"
            stats_dict[area] = Dict()
            stats_dict[area]["min"] = minimum(data_arranged[area]["Xtrain"], dims=2)
            stats_dict[area]["max"] = maximum(data_arranged[area]["Xtrain"], dims=2)
            data_arranged[area]["Xtrain"] = (data_arranged[area]["Xtrain"] .- stats_dict[area]["min"]) ./ (stats_dict[area]["max"] - stats_dict[area]["min"])
            data_arranged[area]["Xtest"] = (data_arranged[area]["Xtest"] .- stats_dict[area]["min"]) ./ (stats_dict[area]["max"] - stats_dict[area]["min"])
        elseif normalizatoin_method == "none"
            continue
        else
            error("Invalid normalization method")
        end
    end
    return data_arranged, stats_dict

end

# function to normalize the test data
function normalize_test_data(data_arranged, normalizatoin_method, stats_dict)
    for area in keys(data_arranged)
        if normalizatoin_method == "standardize"
            data_arranged[area]["Xtest"] = (data_arranged[area]["Xtest"] .- stats_dict[area]["mean"]) ./ stats_dict[area]["std"]
        elseif normalizatoin_method == "minmax"
            data_arranged[area]["Xtest"] = (data_arranged[area]["Xtest"] .- stats_dict[area]["min"]) ./ (stats_dict[area]["max"] - stats_dict[area]["min"])
        elseif normalizatoin_method == "none"
            continue
        else
            error("Invalid normalization method")
        end
    end
    return data_arranged

end

function get_area_dataset(data_arranged, area)
    Xtrain = data_arranged[area]["Xtrain"]
    ytrain = data_arranged[area]["ytrain"]
    Xtest = data_arranged[area]["Xtrain"]
    ytest = data_arranged[area]["ytrain"]
    return Xtrain, ytrain, Xtest, ytest
end

### running the code
path_to_data = "D:\\VSCode\\Julia\\Special_Problem"
data_list = [1] # from 1 to 10
test_case = "pglib_opf_case39_epri.m"
feature_selection = [0, 1, 2, 3, 4]
train_percent = 0.8
normalizatoin_method = "standardize"

@time data = load_data(path_to_data, data_list)
shared_variable_ids = get_shared_variable_ids(path_to_data, test_case)
data_dict, run_ids = get_data_dicts(data, shared_variable_ids)
data_matrix = get_data_matrix(data_dict, feature_selection, run_ids)
data_arranged = get_dataset(data_matrix, train_percent)
data_arranged, stats_dict = normalize_arranged_data(data_arranged, normalizatoin_method)
