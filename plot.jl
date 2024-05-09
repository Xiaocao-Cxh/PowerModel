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
