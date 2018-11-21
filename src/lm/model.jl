mutable struct Model
    config
    worddict
    nn
end

function Model(config)
    traindata, testdata, worddict = readdata(config)
    nn = NN_LSTM(Float32, length(worddict))
    m = Model(config, worddict, nn)
    train!(m, traindata, testdata)
    m
end

function train!(model, traindata, testdata)
    config = model.config
    device = config["device"]
    nn = todevice(model.nn, device)
    traindata = DataLoader(traindata, device=device, batchsize=config["seqlength"])
    testdata = DataLoader(testdata, device=device, batchsize=1)

    for epoch = 1:config["nepochs"]
        println("Epoch:\t$epoch")
        opt.rate = config["learning_rate"]
        println("Learning rate: $(opt.rate)")

        loss = fit!(traindata, nn)
        loss /= length(traindata.data.data)
        println("Loss:\t$loss")
        #perplexity = 2^loss
        #println("Perplexity:\t$perplexity")

        println()
    end
end
