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

    traindata = readconll(config["train_file"], dicts)
    testdata = readconll(config["test_file"], dicts)

    if config["nn"] == "cnn"
        nn = nn_cnn(wordembeds, charembeds, length(tagdict))
    elseif config["nn"] == "lstm"
        nn = nn_lstm(wordembeds, charembeds, length(tagdict))
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
    nn = todevice(model.nn, device)
    opt = SGD()
    params = parameters(nn)
    batchsize = config["batchsize"]
    traindata = DataLoader(traindata, batchsize=batchsize, shuffle=true, device=device)
    testdata = DataLoader(testdata, batchsize=100, shuffle=false, device=device)

    for epoch = 1:config["nepochs"]
        println("Epoch:\t$epoch")
        #opt.rate = config["learning_rate"]
        opt.rate = config["learning_rate"] * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))
        println("Learning rate: $(opt.rate)")

        loss = 0.0
        Merlin.settrain(true)
        foreach(traindata) do data
            z = nn(data)[1]
            y = data.t
            l = softmax_crossentropy(y, z)
            gradient!(l)
            opt.(params)
            loss += sum(Array(l.data))
        end
        loss /= length(traindata)
        println("Loss:\t$loss")

        # test
        Merlin.settrain(false)
        golds = Int[]
        preds = Int[]
        foreach(testdata) do data
            y = data.t.data
            z = nn(data)[1]
            if device < 0
                ind = map(x -> x[1], argmax(z.data,dims=1))
                z = vec(ind)
            else
                maxind = vec(argmax(z.data,dims=1))
                y = Array{Int}(Array(y))
                z = Array{Int}(Array(maxind)) .+ 1
            end
            append!(golds, y)
            append!(preds, z)
        end
        length(preds) == length(golds) || throw("Length mismatch: $(length(preds)), $(length(golds))")
        preds = bioes_decode(preds, model.dicts.t)
        golds = bioes_decode(golds, model.dicts.t)
        fscore(golds, preds)
        println()
    end
    model.nn = todevice(model.nn, -1)
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
