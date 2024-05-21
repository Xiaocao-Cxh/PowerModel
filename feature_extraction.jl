# Get Data from result_divided_1-5.bson
using BSON
using PowerModels, PowerModelsADA
using Ipopt # C++ optimization solver

path_to_data = "D:\\VSCode\\Julia\\Special_Problem"
num_area = 3
alpha = 1000
test_case = "pglib_opf_case39_epri.m"
case_path = joinpath(path_to_data, test_case)
data_base = parse_file(case_path)
result_sample = solve_dopf_admm(data_base, ACPPowerModel, Ipopt.Optimizer; alpha = 1000, tol=1e-16, max_iteration=1, print_level=0, save_data=["solution", "mismatch","shared_variable", "received_variable", "dual_variable"])



shared_variable_ids = Dict()
for area in keys(result_sample)
    shared_variable_ids[area] = Dict()
    for j in keys(result_sample[i]["shared_variable"])
        shared_variable_ids[i][parse(Int,j)] = Dict()
        for k in keys(result_sample[i]["shared_variable"][j])
            shared_variable_ids[i][parse(Int,j)][k] = collect(keys(result_sample[i]["shared_variable"][j][k]))
        end
    end
end



k = 1 # Number of divided data (later we will loop through all data stored)
# Order of data in result_divided_$k.bson based on area_id
data = BSON.load("$(path_to_data)/result_divided_$k.bson")
data_ids = sort(collect(keys(data)))
max_itr_id = 5000 # maximum number of iterations in the data
max_run_id = Int64(length(data_ids)/(max_itr_id*num_area))

run_id = 1 # run index (later we will loop through all runs)
itr_idx = 1 # iteration index (later we will loop through all iterations) (should be equal to the number of iterations in the data multiplied by the number of scenarios)



run_id_pointer = findfirst(i -> data[i]["run_id"] == run_id, data_ids) # pointer to the start of run_id data
run_id_pointer = 1 + max_itr_id*num_area*(run_id-1)

itr_start = data_ids[run_id_pointer+num_area*(itr_idx-1)] # start of a iteration
itr_input = Dict()
for a in 0:num_area-1
    itr_input[data[itr_start+a]["area_id"]] = data[itr_start+a]["input"]
end

# 0 solution
solution = Dict(i => itr_input[i]["solution"] for i in keys(itr_input))

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
