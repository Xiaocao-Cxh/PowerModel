

## Arrange input/outout data for training and testing
using Flux
using Flux.Optimise: update!
using Flux.Losses: mae
using Printf
using BSON
using Random
using StatsBase

# function get_dataset( train_percent, dataset_path::String; standardize="none", stats_dics=Dict())
#     dataset = BSON.load(dataset_path)
#     get_dataset( train_percent, dataset; standardize=standardize, stats_dics=stats_dics)
# end

function get_dataset(data::Dict, train_percent::Float64=0.8; standardize="none", stats_dics=Dict())


    areas_id = unique(data[i]["area_id"] for i in keys(data)) # Get unique area ids
    data_area = Dict()
    for area in areas_id # Get data for each area
        data_area[area] = Dict(i => data[i] for i in keys(data) if data[i]["area_id"] == area)
    end


    # [data_area1[2]["input"]["solution"][k][j] for k in keys(data_area1[2]["input"]["solution"]) for j in keys(data_area1[2]["input"]["solution"][k])]
    data_arranged = Dict()
    for area in areas_id
        # X is a vector with each entery is another vector of the input data of the NN+
        # Here, we use X1 for inputs with output label 1 (for example, not converged) and X2 for inputs with output label 2 (for example, converged)
        X1 = Vector{Vector{Float32}}() # X1 for inputs with output label 0 (for example, not converged)
        for n in eachindex(data_area[area])
            if  data_area[area][n]["label"] == 0
                # input_sample1 =[data_area[area][n]["input"]["solution"][k][j] for k in keys(data_area[area][n]["input"]["solution"]) for j in keys(data_area[area][n]["input"]["solution"][k]) ]
                input_sample2 =[data_area[area][n]["input"]["mismatch"]["$area2"][k][j] for area2 in areas_id if area2 != area for k in keys(data_area[area][n]["input"]["mismatch"]["$area2"]) for j in keys(data_area[area][n]["input"]["mismatch"]["$area2"][k]) ]
                input_sample3 = data_area[area][n]["input"]["mismatch"]["$area"]

                push!(X1,[input_sample2; input_sample3])
            end
        end
        X1 = hcat(X1...)

        X2 = Vector{Vector{Float32}}() # X2 for inputs with output label 1 (for example, converged)
        for n in eachindex(data_area[area])
            if  data_area[area][n]["label"] == 1
                # input_sample1 =[data_area[area][n]["input"]["solution"][k][j] for k in keys(data_area[area][n]["input"]["solution"]) for j in keys(data_area[area][n]["input"]["solution"][k]) ]
                input_sample2 =[data_area[area][n]["input"]["mismatch"]["$area2"][k][j] for area2 in areas_id if area2 != area for k in keys(data_area[area][n]["input"]["mismatch"]["$area2"]) for j in keys(data_area[area][n]["input"]["mismatch"]["$area2"][k]) ]
                input_sample3 = data_area[area][n]["input"]["mismatch"]["$area"]

                push!(X2,[input_sample2; input_sample3])
            end
        end
        X2 = hcat(X2...)

        # Perpare output data (labeling)
        train_length_conv = Int(ceil(train_percent*size(X1)[2]))
        train_idcs = sample(1:size(X1)[2], train_length_conv, replace=false)
        test_idcs = setdiff(1:size(X1)[2], train_idcs)
        Xtrain_1 = X1[:,train_idcs]
        Xtest_1 = X1[:,test_idcs]
        ytrain_1 = 0 .* ones(1,size(Xtrain_1)[2])
        ytest_1 = 0 .* ones(1,size(Xtest_1)[2])

        train_length_notconv = Int(ceil(train_percent*size(X2)[2]))
        train_idcs = sample(1:size(X2)[2], train_length_notconv, replace=false)
        test_idcs = setdiff(1:size(X2)[2], train_idcs)
        Xtrain_2 = X2[:,train_idcs]
        Xtest_2 = X2[:,test_idcs]
        ytrain_2 = 1 .* ones(1,size(Xtrain_2)[2])
        ytest_2 = 1 .* ones(1,size(Xtest_2)[2])

        Xtrain = hcat(Xtrain_1, Xtrain_2)
        ytrain = hcat(ytrain_1, ytrain_2)
        Xtest = hcat(Xtest_1, Xtest_2)
        ytest = hcat(ytest_1, ytest_2)

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

#     if standardize == "scale"
#         max_x = maximum(Xtrain, dims =2)
#         min_x = minimum(Xtrain, dims =2)

#         stats = Dict{Symbol,Any}(:min => min_x, :max => max_x)
#         Xtrain = (Xtrain .- min_x) ./ (max_x - min_x)
#         Xtest = (Xtest .- min_x) ./ (max_x - min_x)

#     elseif standardize == "normalize"
#         mean_x = mean(Xtrain, dims =2)
#         std_x = std(Xtrain, dims =2)
#         stats = Dict{Symbol,Any}(:mean => mean_x, :std => std_x)

#         if std_x != 0
#             Xtrain = (Xtrain .- mean_x) ./ std_x
#             Xtest = (Xtest .- mean_x) ./ std_x
#         else
#             Xtrain = (Xtrain .- mean_x)
#             Xtest = (Xtest .- mean_x)
#         end

#     else
#         stats = stats_dics
#     end
# return Xtrain, ytrain, Xtest, ytest, stats # Input, label, test, label, stats for standardization
# end

function get_area_dataset(data_arranged, area)
    Xtrain = data_arranged[area]["Xtrain"]
    ytrain = data_arranged[area]["ytrain"]
    Xtest = data_arranged[area]["Xtrain"]
    ytest = data_arranged[area]["ytrain"]
    return Xtrain, ytrain, Xtest, ytest
end


function loss_function(x,y,params)
    return Flux.Losses.mse(model(x),y)   #penalize weights, but not bias
end

function my_custom_train!(params,X, Y, opt, loss_tr, batchsize, model)

    # training_loss is declared local so it will be available for logging outside the gradient calculation.
    idcs = sample(1:length(axes(X,2)), batchsize, replace=false)
    Xm = X[:, idcs]
    Ym = Y[:, idcs]
    local training_loss
    gs = gradient(params) do
        training_loss = loss_function(Xm, Ym, params)
        return training_loss
    end
    update!(opt, params, gs)
    push!(loss_tr, training_loss);
    println("Loss ", training_loss);
end


# ## Running the code
# train_percent = 0.85
# Xtrain, ytrain, Xtest, ytest = get_dataset(train_percent, "datapath")

# # importants parameters
# n_in = size(Xtrain)[1] # number of NN inputs (you might need to include this in the previous function)
# n_hidden = 10
# n_out = 1

# ## specify NN arch
# # dropout layer may be considered
# model = Chain(Dense(n_in,n_in,relu,init=Flux.kaiming_normal),Dense(n_in,n_in,relu,init=Flux.kaiming_normal),Dense(n_in,n_in,relu,init=Flux.kaiming_normal),Dense(n_in,n_in,relu,init=Flux.kaiming_normal),Dense(n_in,n_out,sigmoid,init=Flux.kaiming_normal));
# # https://fluxml.ai/Flux.jl/stable/models/activation/, https://fluxml.ai/Flux.jl/stable/models/layers/

# ## specify NN parameters
# opt = ADAM(5e-4)
# number_epochs = 1000
# batchsize = 5000
# loss_tr = Vector{Float32}()
# val_tr = Vector{Float32}()
# params = Flux.params(model)

# for n = 1:number_epochs
#     println("Epoch ", n)
#     my_custom_train!(params, Xtrain, ytrain, opt, loss_tr, batchsize, model)
# end

# y_pred = model(Xtest)
# fp = 0 # false positive
# fn = 0 # false negative
# tp = 0 # true positive
# tn = 0 # true negative
# for n = 1:length(axes(y_pred,2))
#     if y_pred[n] < 0.5 && ytest[n] == 0
#         global tn += 1
#     elseif y_pred[n] < 0.5 && ytest[n] == 1
#         global fn += 1
#     elseif y_pred[n] >= 0.5 && ytest[n] == 0
#         global fp += 1
#     elseif y_pred[n] >= 0.5 && ytest[n] == 1
#         global tp += 1
#     end
# end
# println("Precision: ", tp/(tp+fp))
# println("Recall: ", tp/(tp+fn))
# println("Accuracy: ", (tp+tn)/(tp+fp+fn+tn))
# test_loss = Flux.Losses.mae(y_pred,ytest)
# println("Test loss ", test_loss)
# plot(loss_tr, yscale=:log10)
