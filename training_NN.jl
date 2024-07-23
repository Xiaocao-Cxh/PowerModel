# Get Data from result_test_k.bson
using Random, LinearAlgebra, Statistics, Printf # These are the standard libraries
using BSON, PowerModels, PowerModelsADA, StatsBase, Flux, Ipopt # C++ optimization solver

### running the code
path_to_data = "D:\\VSCode\\Julia\\Special_Problem"
data_list = [1] # from 1 to 10
test_case = "pglib_opf_case39_epri.m"
feature_selection = [0, 1, 2, 3, 4]
train_percent = 0.8
normalizatoin_method = "standardize"

include("$(path_to_data)\\feature_extraction.jl")
data = load_data(path_to_data, data_list)
shared_variable_ids = get_shared_variable_ids(path_to_data, test_case)
data_dict, run_ids = get_data_dicts(data, shared_variable_ids)
data_matrix = get_data_matrix(data_dict, feature_selection, run_ids)
data_arranged = get_dataset(data_matrix, train_percent)
data_arranged, stats_dict = normalize_arranged_data(data_arranged, normalizatoin_method)


area = 1
Xtrain, ytrain, Xtest, ytest = get_area_dataset(data_arranged, area)

n_in = size(Xtrain)[1] # number of NN inputs (you might need to include this in the previous function)
n_hidden = 50
n_out = 1


## specify NN arch
# dropout layer may be considered
model = Chain(Dense(n_in,n_in,relu,init=Flux.kaiming_normal),Dense(n_in,n_hidden,relu,init=Flux.kaiming_normal),Dense(n_hidden,n_hidden,relu,init=Flux.kaiming_normal),Dense(n_hidden,n_hidden,relu,init=Flux.kaiming_normal),Dense(n_hidden,n_hidden,relu,init=Flux.kaiming_normal),Dense(n_hidden,n_hidden,relu,init=Flux.kaiming_normal),Dense(n_hidden,n_out,sigmoid,init=Flux.kaiming_normal));
# https://fluxml.ai/Flux.jl/stable/models/activation/, https://fluxml.ai/Flux.jl/stable/models/layers/

## specify NN parameters
opt = ADAM(5e-4)
number_epochs = 50000
batchsize = 200
loss_tr = Vector{Float32}()
val_tr = Vector{Float32}()
params = Flux.params(model)

for n = 1:number_epochs
    println("Epoch ", n)
    my_custom_train!(params, Xtrain, ytrain, opt, loss_tr, batchsize, model)
end

y_pred = model(Xtest)
fp = 0 # false positive
fn = 0 # false negative
tp = 0 # true positive
tn = 0 # true negative
for n = 1:length(axes(y_pred,2))
    if y_pred[n] < 0.5 && ytest[n] == 0
        global tn += 1
    elseif y_pred[n] < 0.5 && ytest[n] == 1
        global fn += 1
    elseif y_pred[n] >= 0.5 && ytest[n] == 0
        global fp += 1
    elseif y_pred[n] >= 0.5 && ytest[n] == 1
        global tp += 1
    end
end
println("Precision: ", tp/(tp+fp))
println("Recall: ", tp/(tp+fn))
println("Accuracy: ", (tp+tn)/(tp+fp+fn+tn))
test_loss = Flux.Losses.mae(y_pred,ytest)
println("Test loss ", test_loss)
plot(loss_tr, yscale=:log10)
