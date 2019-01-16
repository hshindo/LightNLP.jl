mutable struct Model
    config
    dicts
    nn
end

function Model(config::Dict)
    words = h5read(config["wordvec_file"], "words")
    wordembeds = h5read(config["wordvec_file"], "vectors")
    worddict = Dict(words[i] => i for i=1:length(words))
    # wordembeds[:,worddict["unk"]] = zeros(Float32,size(wordembeds,1))

    flair_train = Var(h5read(".data/flair.eng.train.h5", "vectors"))
    flair_test = Var(h5read(".data/flair.eng.testb.h5", "vectors"))

    # chardict, tagdict = initvocab(config["train_file"])
    dicts = (word=worddict, char=Dict{String,Int}(), tag=Dict{String,Int}())
    traindata = readconll(config["train_file"], dicts, true)
    testdata = readconll(config["test_file"], dicts, false)
    T = eltype(wordembeds)
    n = length(worddict) - size(wordembeds,2)
    #if n > 0
    #    e = Normal(0,0.01)(T, size(wordembeds,1), n)
    #    wordembeds = cat(wordembeds, e, dims=2)
    #end
    charembeds = Uniform(-0.01,0.01)(T, 20, length(dicts.char))

    if config["nn"] == "cnn"
        #nn = nn_cnn(wordembeds, charembeds, length(tagdict))
    elseif config["nn"] == "lstm"
        nn = NN_Graph(wordembeds, flair_train, flair_test, charembeds, length(dicts.tag))
        # nn = NN_LSTM(wordembeds, flair_train, flair_test, length(dicts.tag))
    else
        throw("Unknown nn")
    end

    @info "#Training examples:\t$(length(traindata))"
    @info "#Testing examples:\t$(length(testdata))"
    @info "#Words:\t$(length(dicts.word))"
    @info "#Chars:\t$(length(dicts.char))"
    @info "#Tags:\t$(length(dicts.tag))"
    m = Model(config, dicts, nn)
    train!(m, traindata, testdata)
    m
end

function train!(model::Model, traindata, testdata)
    config = model.config
    Merlin.setdevice(config["device"])
    opt = SGD()
    nn = todevice(model.nn)
    params = parameters(nn)
    batchsize = config["batchsize"]
    #batchsize = 32
    #losses = Float32[]
    for epoch = 1:config["nepochs"]
        println("Epoch:\t$epoch")
        # epoch == 100 && (opt.on = true)
        #opt.alpha = opt.alpha / (1 + 0.1*epoch)
        opt.rate = 0.1 * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))
        # opt.rate = 0.1 * batchsize / sqrt(batchsize) / (1 + 0.05*epoch)
        # opt.rate = config["learning_rate"] * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))
        #opt.opt.rate = config["learning_rate"] * batchsize / sqrt(batchsize)
        #opt.opt.rate = 0.0
        println("Learning rate: $(opt.rate)")

        loss = minimize!(nn, traindata, opt, batchsize=batchsize, shuffle=true)
        # push!(losses, loss)
        loss /= length(traindata)
        println("Loss:\t$loss")
        #if epoch % 5 == 0
        #    minimum(losses) < losses[1] || (opt.rate /= 2.0)
        #    empty!(losses)
        #end

        #if opt.on
        #    yz = replace!(opt,params) do
        #        evaluate(nn, testdata, batchsize=100)
        #    end
        #else
        #    yz = evaluate(nn, testdata, batchsize=100)
        #end
        # yz = evaluate(nn, testdata, batchsize=100, device=device)
        yz = evaluate(nn, testdata, batchsize=100)
        golds, preds = Int[], Int[]
        for (y,z) in yz
            append!(golds, y)
            append!(preds, z)
        end
        # accuracy(golds, preds, model.dicts.tag)
        # oracles = bioes_decode_oracle(preds, model.dicts.tag)
        preds = bioes_decode(preds, model.dicts.tag)
        golds = bioes_decode(golds, model.dicts.tag)
        fscore(golds, preds)
        println()
    end
    #model.nn = todevice(model.nn, -1)
end

function fscore_tag(golds::Vector, preds::Vector)
    dict = Dict()
    for (i,j,tag) in golds
        data = get!(dict, tag) do
            (golds=[], preds=[])
        end
        push!(data.golds, (i,j,tag))
    end
    for (i,j,tag) in preds
        data = dict[tag]
        push!(data.preds, (i,j,tag))
    end
    for (tag,data) in dict
        println("----------")
        println("Tag: $tag")
        fscore(data.golds, data.preds)
    end
    println("----------")
    println("All")
    fscore(golds, preds)
end

function fscore(golds::Vector, preds::Vector)
    set = intersect(Set(golds), Set(preds))
    count = length(set)
    prec = round(count/length(preds), digits=5)
    recall = round(count/length(golds), digits=5)
    fval = round(2*recall*prec/(recall+prec), digits=5)
    println("----------")
    println("Prec:\t$prec")
    println("Recall:\t$recall")
    println("Fscore:\t$fval")
end

function accuracy(golds::Vector, preds::Vector, dict::Dict)
    @assert length(golds) == length(preds)
    counts = [0 for _=1:length(dict)]
    totals = [0 for _=1:length(dict)]
    for i = 1:length(golds)
        g, p = golds[i], preds[i]
        totals[g] += 1
        g == p && (counts[g] += 1)
    end
    tags = Array{String}(undef, length(dict))
    foreach(x -> tags[x[2]] = x[1], dict)
    println("----------")
    println("Accuracy:")
    acc = round(sum(counts)/sum(totals), digits=5)
    println("Total:\t$acc")
    for i = 1:length(counts)
        c, t = counts[i], totals[i]
        acc = round(c/t, digits=5)
        println("$(tags[i]):\t$acc ($c/$t)")
    end
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

function bioes_decode_oracle(ids::Vector{Int}, tagdict::Dict{String,Int})
    id2tag = Array{String}(undef, length(tagdict))
    for (k,v) in tagdict
        id2tag[v] = k
    end

    spans = Tuple{Int,Int,String}[]
    bpos = 0
    bprev = 0
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
            if bprev > 0
                tag = id2tag[ids[bprev]]
                basetag = length(tag) > 2 ? tag[3:end] : ""
                push!(spans, (bprev,i,basetag))
            end
            bprev = bpos
            bpos = 0
        end
    end
    spans
end
