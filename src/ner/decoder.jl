mutable struct Decoder
    worddict::Dict
    chardict::Dict
    tagset
    nn
end

function Decoder(embedsfile::String, trainfile::String, testfile::String, nepochs::Int, learnrate::Float64, batchsize::Int)
    words = h5read(embedsfile, "words")
    worddict = Dict(words[i] => i for i=1:length(words))
    chardict = Dict("UNKNOWN" => 1)
    for word in words
        for c in Vector{Char}(word)
            get!(chardict, string(c), length(chardict)+1)
            get!(chardict, string(uppercase(c)), length(chardict)+1)
        end
    end

    tagset = BIOES()
    traindata = readdata(trainfile, worddict, chardict, tagset)
    testdata = readdata(testfile, worddict, chardict, tagset)
    wordembeds = embeddings(h5read(embedsfile,"vectors"))
    charembeds = embeddings(Float32, length(chardict), 20, init_w=Normal(0,0.05))
    nn = NN(wordembeds, charembeds, length(tagset.tag2id))

    info("#Training examples:\t$(length(traindata))")
    info("#Testing examples:\t$(length(testdata))")
    info("#Words:\t$(length(worddict))")
    info("#Chars:\t$(length(chardict))")
    info("#Tags:\t$(length(tagset.tag2id))")
    testdata = batch(testdata, 100)

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
            w, c, t = batches[i]
            y = nn(w, c, t)
            loss += sum(y.data)
            params = gradient!(y)
            foreach(opt, params)
            ProgressMeter.next!(prog)
        end
        loss /= length(batches)
        println("Loss:\t$loss")

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
    Decoder(worddict, chardict, tagset, nn)
end

function decode(dec::Decoder, path::String)
    lines = open(readlines, path)
    map(lines) do line
        words = Vector{String}(split(line," "))
        w = encode_word(dec.worddict, words)
        c = encode_char(dec.chardict, words)
        y = dec.nn(w, c)
        decode(dec.tagset, y)
    end
end

function encode_word(worddict::Dict, words::Vector{String})
    unkword = worddict["UNKNOWN"]
    ids = map(words) do w
        w = lowercase(w)
        @assert !isempty(w)
        # w = replace(word, r"[0-9]", '0')
        get(worddict, w, unkword)
    end
    Var(ids)
end

function encode_char(chardict::Dict, words::Vector{String})
    unkchar = chardict["UNKNOWN"]
    batchdims = Int[]
    ids = Int[]
    for w in words
        # w = replace(word, r"[0-9]", '0')
        chars = Vector{Char}(w)
        @assert !isempty(chars)
        push!(batchdims, length(chars))
        for c in chars
            push!(ids, get(chardict,string(c),unkchar))
        end
    end
    Var(ids, batchdims)
end

function readdata(path::String, worddict::Dict, chardict::Dict, tagset)
    data = NTuple{3,Var}[]
    words, tags = String[], String[]
    lines = open(readlines, path)
    push!(lines, "")
    for i = 1:length(lines)
        line = lines[i]
        if isempty(line)
            isempty(words) && continue
            w = encode_word(worddict, words)
            c = encode_char(chardict, words)
            t = Var(encode(tagset,tags))
            push!(data, (w,c,t))
            empty!(words)
            empty!(tags)
        else
            items = Vector{String}(split(line,"\t"))
            items[2] == "O\r" && (items[2] = "O")
            word = items[1]
            isempty(word) && (word = "UNKNOWN")
            push!(words, word)
            push!(tags, items[2])
        end
    end
    data
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
