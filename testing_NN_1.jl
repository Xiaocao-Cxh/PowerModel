# Deploy the model to test the model for Area 1.

# parameters
path_to_data = "D:\\VSCode\\Julia\\Special_Problem"
cd(path_to_data)
data_list = [2]
test_case = "pglib_opf_case39_epri.m"
feature_selection = [0, 3, 4]
train_percent = 0.8
normalizatoin_method = "standardize"
include("$(path_to_data)\\feature_extraction.jl")

# Load model.
BSON.@load "model.bson" model
# Load new dataset.
data = load_data(path_to_data, data_list)
shared_variable_ids = get_shared_variable_ids(path_to_data, test_case)
data_dict, run_ids = get_data_dicts(data, shared_variable_ids)
data_matrix = get_data_matrix(data_dict, feature_selection, run_ids)
data_arranged = get_dataset(data_matrix, train_percent)
data_arranged, stats_dict = normalize_arranged_data(data_arranged, normalizatoin_method)
area = 1
Xtrain, ytrain, Xtest, ytest = get_area_dataset(data_arranged, area)

y_pred = model(Xtest)
mse = Flux.Losses.mse(y_pred, ytest)
# Get data for first run.
# Get correct termination.
# Loop throught iterations, and check if the termination is correct for each iteration.
# Error measures: MSE, Monotonicity, Difference between first convergence flag and correct iteration (Most important, positive are better than negative effect)
