using Random, LinearAlgebra, Statistics, Printf # These are the standard libraries
using BSON, PowerModels, PowerModelsADA, StatsBase, Flux, Ipopt, Plots # Ipopt is the C++ optimization solver
using JuMP # Julia optimization solver

# running the code
path_to_data = "D:\\VSCode\\Julia\\Special_Problem"
cd(path_to_data)
data_list = [1]
test_case = "pglib_opf_case39_epri.m"
feature_selection = [0, 3, 4]
train_percent = 0.8
normalizatoin_method = "standardize"
include("$(path_to_data)\\feature_extraction.jl")

# Data Wrangling
data = load_data(path_to_data, data_list)
shared_variable_ids = get_shared_variable_ids(path_to_data, test_case)
data_dict, run_ids = get_data_dicts(data, shared_variable_ids)
data_matrix = get_data_matrix(data_dict, feature_selection, run_ids)
data_arranged = get_dataset(data_matrix, train_percent)
data_arranged, stats_dict = normalize_arranged_data(data_arranged, normalizatoin_method)

# Data Splitting
area = 1
Xtrain, ytrain, Xtest, ytest = get_area_dataset(data_arranged, area)

## NN Architecture
# dropout layer may be considered
n_in = size(Xtrain)[1] # number of NN inputs (you might need to include this in the previous function)
n_hidden = 50 #!!!
n_out = 1
model = Chain(Dense(n_in,n_in,relu,init=Flux.kaiming_normal),Dense(n_in,n_in,relu,init=Flux.kaiming_normal),Dense(n_in,n_hidden,relu,init=Flux.kaiming_normal),Dense(n_hidden,n_hidden,relu,init=Flux.kaiming_normal),Dense(n_hidden,n_hidden,relu,init=Flux.kaiming_normal),Dense(n_hidden,n_hidden,relu,init=Flux.kaiming_normal),Dense(n_hidden,n_out,sigmoid,init=Flux.kaiming_normal));
# https://fluxml.ai/Flux.jl/stable/models/activation/, https://fluxml.ai/Flux.jl/stable/models/layers/

## specify NN parameters
opt = ADAM(5e-4)
number_epochs = 500 #!!!
batchsize = 1000 #!!!
loss_tr = Vector{Float64}()
val_tr = Vector{Float64}()
loss_ts = Vector{Float64}()
params = Flux.params(model)
# 1st: weights, 2nd: bias, 3rd: weights of 2nd layer, 4th: bias of 2nd layer, ...
# Check if FLux have models to store and load the model parameters

function my_custom_train!(params, X, Y, opt, loss_tr, batchsize, model)
    local training_loss
    # training_loss is declared local so it will be available for logging outside the gradient calculation.
    idcs = sample(1:length(axes(X,2)), batchsize, replace=false)
    Xm = X[:, idcs]
    Ym = Y[idcs]
    Ym = reshape(Y[idcs], 1, length(idcs))
    gs = gradient(params) do
        training_loss = Flux.Losses.mse(model(Xm),Ym)
        return training_loss
    end
    Flux.Optimise.update!(opt, params, gs)
    push!(loss_tr, training_loss);
    println("Loss ", training_loss);
end

for n = 1:number_epochs
    println("Epoch ", n)
    my_custom_train!(params, Xtrain, ytrain, opt, loss_tr, batchsize, model)

    y_pred = model(Xtest)
    y_pred = reshape(y_pred, length(y_pred))
    push!(loss_ts, Flux.Losses.mse(y_pred, ytest))
end

y_pred = model(Xtest)
y_pred = reshape(y_pred, length(y_pred))
fp = 0 # false positive
fn = 0 # false negative
tp = 0 # true positive
tn = 0 # true negative
for n in eachindex(y_pred)
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

# Save the model
BSON.@save "model.bson" model

# # Plot the loss
# ax = plot(loss_tr, label="Tr")
# plot!(ax, loss_ts, label="Ts")
# xlabel!("Epoch")
# ylabel!("Loss (MSE)")
# title!("Training and Test Loss")
# savefig("$(path_to_data)\\plots\\area_$(area)_loss.png")
