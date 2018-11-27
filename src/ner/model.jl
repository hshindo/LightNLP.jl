mutable struct Model
    config
    dicts
    nn
end

function Model(config::Dict)
    words = h5read(config["wordvec_file"], "words")
    wordembeds = h5read(config["wordvec_file"], "vectors")
    worddict = Dict(words[i] => i for i=1:length(words))

    chardict, tagdict = initvocab(config["train_file"])
    charembeds = Normal(0,0.01)(eltype(wordembeds), 20, length(chardict))
    dicts = (w=worddict, c=chardict, t=tagdict)

    traindata = readconll(config["train_file"], dicts, true)
    testdata = readconll(config["test_file"], dicts, false)

    if config["nn"] == "cnn"
        #nn = nn_cnn(wordembeds, charembeds, length(tagdict))
    elseif config["nn"] == "lstm"
        nn = NN_LSTM(wordembeds, charembeds, length(tagdict))
    else
        throw("Unknown nn")
    end

    @info "#Training examples:\t$(length(traindata))"
    @info "#Testing examples:\t$(length(testdata))"
    @info "#Words:\t$(length(worddict))"
    @info "#Chars:\t$(length(chardict))"
    @info "#Tags:\t$(length(tagdict))"
    m = Model(config, dicts, nn)
    train!(m, traindata, testdata)
    m
end

function train!(model::Model, traindata, testdata)
    config = model.config
    device = config["device"]
    opt = SGD()
    nn = todevice(model.nn, device)
    batchsize = config["batchsize"]

    for epoch = 1:config["nepochs"]
        println("Epoch:\t$epoch")
        #opt.rate = config["learning_rate"]
        opt.rate = config["learning_rate"] * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))
        println("Learning rate: $(opt.rate)")

        loss = minimize!(nn, traindata, opt, batchsize=batchsize, shuffle=true, device=device)
        loss /= length(traindata)
        println("Loss:\t$loss")

        yz = evaluate(nn, testdata, batchsize=100, device=device)
        golds, preds = Int[], Int[]
        for (y,z) in yz
            append!(golds, Array{Int}(y))
            append!(preds, Array{Int}(z))
        end

        length(preds) == length(golds) || throw("Length mismatch: $(length(preds)), $(length(golds))")
        preds = bioes_decode(preds, model.dicts.t)
        golds = bioes_decode(golds, model.dicts.t)
        fscore(golds, preds)
        println()
    end
    #model.nn = todevice(model.nn, -1)
end

function fscore(golds::Vector{T}, preds::Vector{T}) where T
    set = intersect(Set(golds), Set(preds))
    count = length(set)
    prec = round(count/length(preds), digits=5)
    recall = round(count/length(golds), digits=5)
    fval = round(2*recall*prec/(recall+prec), digits=5)
    println("Prec:\t$prec")
    println("Recall:\t$recall")
    println("Fscore:\t$fval")
end

function bioes_decode(ids::Vector{Int}, tagdict::Dict{String,Int})
    id2tag = Array{String}(undef, length(tagdict))
    for (k,v) in tagdict
        id2tag[v] = k
    end

    spans = Tuple{Int,Int,String}[]
    bpos = 0
    for i = 1:length(ids)
        tag = id2tag[ids[i]]
        tag == "O" && continue
        startswith(tag,"B") && (bpos = i)
        startswith(tag,"S") && (bpos = i)
        nexttag = i == length(ids) ? "O" : id2tag[ids[i+1]]
        if (startswith(tag,"S") || startswith(tag,"E")) && bpos > 0
            tag = id2tag[ids[bpos]]
            basetag = length(tag) > 2 ? tag[3:end] : ""
            push!(spans, (bpos,i,basetag))
            bpos = 0
        end
    end
    spans
end
