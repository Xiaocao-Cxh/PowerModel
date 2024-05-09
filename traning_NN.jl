using Flux, BSON, StatsBase, Random
data = BSON.load("D:\\VSCode\\Julia\\PowerModeltest.bson")
# One NN for each area

train_percent = 0.85
data_arranged = get_dataset(data, train_percent)

area = 3
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

6y_pred = model(Xtest)
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
