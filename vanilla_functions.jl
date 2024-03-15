# Change the real power and imaginary power of the load in each senarios
function change_demand!(data, demand_change)
    for (i, load) in data["load"]
        frac_change = (rand()-0.5) * 2 * demand_change # Random number between -1 and 1 times the demand change
        load["pd"] = load["pd"] * (1 + frac_change)
        load["qd"] = load["qd"] * (1 + frac_change)
    end
end

function get_label(result, itr_id, area_id, tol_value)
    mismatches = [result[area_id]["previous_solution"]["mismatch"][k][string(area_id)] for k in 1:itr_id]

    # Find the last index where the mismatch drops below the tol_value
    convergence_index = findlast(x -> x > tol_value, mismatches)

    if isnothing(convergence_index)
        # If never below tol_value or the last below is before current iteration
        return 0
    elseif itr_id > convergence_index
        # If the last below is before current iteration
        return 1
    else
        # If the last below is before current iteration
        return 0
    end
end

function extract_data(result, saved_data, run_id, area_id, itr_id, tol_value)
    dataset = Dict("input" => Dict(), "label" => 0,"run_id" => run_id, "area_id" => area_id, "itr_id" => itr_id)
    # Output of label is 0 or 1
    for str in saved_data
        dataset["input"][str] = deepcopy(result[area_id]["previous_solution"][str][itr_id])
    end
    dataset["label"] = get_label(result, itr_id, area_id, tol_value)
    return dataset
end