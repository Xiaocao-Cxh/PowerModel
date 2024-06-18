# Possible features
# 0.Solution
# 1.Shared variable: s
# 2.Received variable: r
# 3.Mismatch: delta_x = s - r
# 4.Dual variable: lambda**k = lambda**(k-1) + alpha * delta_x**k, delta_x**0 = 0, lambda**0 = 0

## Arrange input/outout data for training and testing
using Flux
using Flux.Optimise: update!
using Flux.Losses: mae
using Printf
using BSON
using Random
using StatsBase


area = 1
Xtrain, ytrain, Xtest, ytest = get_area_dataset(data_arranged, area)


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
