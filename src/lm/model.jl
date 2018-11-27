mutable struct Model
    config
    worddict
    nn
end

function Model(config)
    worddict = Dict("<eos>"=>1)
    traindata = readdata!(config["train_file"], config["batchsize"], worddict, true)
    testdata = readdata!(config["test_file"], config["batchsize"], worddict, false)
    @info "#Training examples:\t$(length(traindata))"
    @info "#Testing examples:\t$(length(testdata))"
    @info "#Words:\t$(length(worddict))"

    nn = NN_LSTM(Float32, length(worddict))
    m = Model(config, worddict, nn)
    train!(m, traindata, testdata)
    m
end

function train!(model, traindata, testdata)
    config = model.config
    device = config["device"]
    opt = SGD()
    nn = todevice(model.nn, device)
    batchsize = config["batchsize"]
    #traindata = DataLoader(traindata, device=device, batchsize=config["seqlength"])
    #testdata = DataLoader(testdata, device=device, batchsize=1)

    for epoch = 1:config["nepochs"]
        println("Epoch:\t$epoch")
        opt.rate = config["learning_rate"]
        println("Learning rate: $(opt.rate)")

        loss = minimize!(nn, traindata, opt, batchsize=batchsize, shuffle=false, device=device)
        loss /= length(traindata)
        println("Loss:\t$loss")

        #perplexity = 2^loss
        #println("Perplexity:\t$perplexity")

        println()
    end
end
