mutable struct Decoder
    chardict::Dict
    tagset
    nn
end

function Decoder(vocabfile::String, trainfile::String, testfile::String, nepochs::Int, learnrate::Float64, batchsize::Int)
    chars = open(readlines, vocabfile)
    chardict = Dict("UNKNOWN"=>1, "[EOL]"=>2)

    tagset = BIOES()
    traindata = readdata(trainfile, chardict, tagset)
    testdata = readdata(testfile, chardict, tagset)
    testdata = batch(testdata, 100)
    charembeds = embeddings(Float32, length(chardict), 20, init_w=Normal(0,0.05))
    nn = NN(length(chardict), length(tagset.tag2id))

    info("#Training examples:\t$(length(traindata))")
    info("#Testing examples:\t$(length(testdata))")
    info("#Chars:\t$(length(chardict))")
    info("#Wordtags:\t$(length(tagset.tag2id))")

    opt = SGD()
    for epoch = 1:nepochs
        println("Epoch:\t$epoch")
        opt.rate = learnrate * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))
        println("Learning rate: $(opt.rate)")

        shuffle!(traindata)
        batches = batch(traindata, batchsize)
        prog = Progress(length(batches))
        loss = 0.0
        for i in 1:length(batches)
            c, t1, t2 = batches[i]
            y = nn(c, t1, t2)
            loss += sum(y.data)
            params = gradient!(y)
            foreach(opt, params)
            ProgressMeter.next!(prog)
        end
        loss /= length(batches)
        println("Loss:\t$loss")
        continue

        # test
        println("Testing...")
        preds = Int[]
        golds = Int[]
        for (w,c,t) in testdata
            y = nn(w, c)
            append!(preds, y)
            append!(golds, t.data)
        end
        length(preds) == length(golds) || throw("Length mismatch: $(length(preds)), $(length(golds))")

        preds = decode(tagset, preds)
        golds = decode(tagset, golds)
        fscore(golds, preds)
        println()
    end
    Decoder(chardict, tagset, nn)
end

function encode(chardict::Dict, chars::Vector{String})
    unk = chardict["UNKNOWN"]
    ids = map(c -> get(chardict,c,unk), chars)
    Var(ids)
end

function readdata(path::String, chardict::Dict, tagset)
    data = NTuple{2,Var}[]
    chars, wordtags, senttags = String[], String[], Int[]
    lines = open(readlines, path)
    push!(lines, "")
    for line in lines
        if isempty(line)
            isempty(chars) && continue
            c = encode(chardict, chars)
            t1 = Var(encode(tagset,tags))
            push!(data, (w,c,t))
            empty!(chars)
            empty!(wordtags)
            empty!(senttags)
        else
            items = split(line, "\t")
            push!(chars, String(items[1]))
            push!(wordtags, String(items[2]))
            push!(senttags, parse(Int,items[3]))
        end
    end
    data
end

function train(dec::Decoder, trainfile::String, testfile::String)

end
