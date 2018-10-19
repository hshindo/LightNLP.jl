export NER

mutable struct NER
    config::NamedTuple
    worddict::Dict
    chardict::Dict
    tagdict::Dict
    nn
end

function NER(config::NamedTuple)
    words = h5read(config.wordvec_file, "words")
    wordembeds = h5read(config.wordvec_file, "vectors")
    worddict = Dict(words[i] => i for i=1:length(words))

    chardict, tagdict = initvocab(config.train_file)
    charembeds = Normal(0,0.01)(eltype(wordembeds), 20, length(chardict))

    traindata = CoNLL(config.train_file, worddict, chardict, tagdict)
    #traindata = CoNLL(traindata.data[1:1000])
    testdata = CoNLL(config.test_file, worddict, chardict, tagdict)
    nn = model_lstm(wordembeds, charembeds, length(tagdict))

    @info "#Training examples:\t$(length(traindata))"
    @info "#Testing examples:\t$(length(testdata))"
    @info "#Words:\t$(length(worddict))"
    @info "#Chars:\t$(length(chardict))"
    @info "#Tags:\t$(length(tagdict))"
    ner = NER(config, worddict, chardict, tagdict, nn)
    train!(ner, traindata, testdata)
    ner
end

function train!(ner::NER, traindata, testdata)
    config = ner.config
    Merlin.setdevice(config.device)
    opt = SGD()
    batchsize = config.batchsize
    traindata = todevice(traindata)
    testdata = todevice(testdata)
    nn = todevice(ner.nn)
    for epoch = 1:config.nepochs
        println("Epoch:\t$epoch")
        #opt.rate = LEARN_RATE / BATCHSIZE
        opt.rate = config.learning_rate * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))
        println("Learning rate: $(opt.rate)")

        loss = fit!(nn, traindata, opt, batchsize=batchsize, shuffle=true) do (nn,data)
            z = nn(data)[1]
            y = data.t
            softmax_crossentropy(y, z)
        end
        println("Loss:\t$loss")

        # test
        res = evaluate(nn, testdata, batchsize=100) do (nn,data)
            y = data.t.data
            z = nn(data)[1]
            maxind = vec(argmax(z.data,dims=1))
            if Merlin.oncpu()
                z = maxind(x -> x[1], maxind)
            else
                y = Array{Int}(Array(y))
                z = Array{Int}(Array(maxind)) .+ 1
            end
            (y, z)
        end
        golds = collect(Iterators.flatten(map(r -> r[1], res)))
        preds = collect(Iterators.flatten(map(r -> r[2], res)))
        length(preds) == length(golds) || throw("Length mismatch: $(length(preds)), $(length(golds))")
        preds = bioes_decode(preds, ner.tagdict)
        golds = bioes_decode(golds, ner.tagdict)
        fscore(golds, preds)
        println()
    end
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
