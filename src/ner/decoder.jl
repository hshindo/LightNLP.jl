export decode

mutable struct Decoder
    worddict::Dict
    chardict::Dict
    tagdict::Dict
    nn
    config
end

function Decoder(config::Dict)
    words = h5read(config["wordvec_file"], "words")
    worddict = Dict(words[i] => i for i=1:length(words))
    wordembeds = h5read(config["wordvec_file"], "vectors")

    chardict, tagdict = initvocab(config["train_file"])
    charembeds = Normal(0,0.01)(eltype(wordembeds), 20, length(chardict))

    traindata = readconll(config["train_file"], worddict, chardict, tagdict)
    testdata = readconll(config["test_file"], worddict, chardict, tagdict)
    nn = NN(wordembeds, charembeds, length(tagdict))

    info("#Training examples:\t$(length(traindata))")
    info("#Testing examples:\t$(length(testdata))")
    info("#Words:\t$(length(worddict))")
    info("#Chars:\t$(length(chardict))")
    info("#Tags:\t$(length(tagdict))")
    testdata = create_batch(catsample, testdata, 100)
    dec = Decoder(worddict, chardict, tagdict, nn, config)
    train!(dec, traindata, testdata)
    dec
end

function train!(dec::Decoder, traindata, testdata)
    config = dec.config
    opt = SGD()
    batchsize = config["batchsize"]
    for epoch = 1:config["nepochs"]
        println("Epoch:\t$epoch")
        #opt.rate = LEARN_RATE / BATCHSIZE
        opt.rate = config["learning_rate"] * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))
        println("Learning rate: $(opt.rate)")

        Merlin.settrain(true)
        shuffle!(traindata)
        samples = create_batch(catsample, batchsize, traindata)
        prog = Progress(length(samples))
        loss = 0.0
        for (x,t) in samples
            z = nn.g(x...)
            z = softmax_crossentropy(t, z)
            loss += sum(z.data)
            params = gradient!(z)
            foreach(opt, params)
            ProgressMeter.next!(prog)
        end
        loss /= length(batches)
        println("Loss:\t$loss")

        # test
        println("Testing...")
        Merlin.settrain(false)
        preds = Int[]
        golds = Int[]
        for (x,t) in testdata
            z = nn.g(x...)
            z = argmax(z.data, 1)
            append!(preds, z)
            append!(golds, t)
        end
        length(preds) == length(golds) || throw("Length mismatch: $(length(preds)), $(length(golds))")

        preds = bioes_decode(preds, tagdict)
        golds = bioes_decode(golds, tagdict)
        fscore(golds, preds)
        println()
    end
end

function fscore(golds::Vector{T}, preds::Vector{T}) where T
    set = intersect(Set(golds), Set(preds))
    count = length(set)
    prec = round(count/length(preds), 5)
    recall = round(count/length(golds), 5)
    fval = round(2*recall*prec/(recall+prec), 5)
    println("Prec:\t$prec")
    println("Recall:\t$recall")
    println("Fscore:\t$fval")
end

function bioes_decode(ids::Vector{Int}, tagdict::Dict{String,Int})
    id2tag = Array{String}(length(tagdict))
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
