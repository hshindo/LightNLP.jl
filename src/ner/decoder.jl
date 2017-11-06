export decode

mutable struct Decoder
    worddict1::Dict
    worddict2::Dict
    chardict::Dict
    tagdict::Dict
    nn
end

function Decoder(embedsfile::String, trainfile::String, testfile::String, nepochs::Int, learnrate::Float64, batchsize::Int)
    words1 = h5read(embedsfile, "words")
    worddict1 = Dict(words1[i] => i for i=1:length(words1))
    wordembeds1 = Var(h5read(embedsfile,"vectors"))
    #wordembeds1 = [zerograd(wordembeds1[:,i]) for i=1:size(wordembeds1,2)]
    worddict2, chardict, tagdict = initvocab(trainfile)
    wordembeds2 = embeddings(Float32, length(worddict2), 100, init_w=Zeros())
    charembeds = embeddings(Float32, length(chardict), 20, init_w=Normal(0,0.01))
    traindata = readdata(trainfile, worddict1, worddict2, chardict, tagdict)
    testdata = readdata(testfile, worddict1, worddict2, chardict, tagdict)
    nn = NN(wordembeds1, wordembeds2, charembeds, length(tagdict))

    info("#Training examples:\t$(length(traindata))")
    info("#Testing examples:\t$(length(testdata))")
    info("#Words1:\t$(length(worddict1))")
    info("#Words2:\t$(length(worddict2))")
    info("#Chars:\t$(length(chardict))")
    info("#Tags:\t$(length(tagdict))")
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
            y = nn(batches[i]...)
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
        for (w1,w2,c,t) in testdata
            y = nn(w1, w2, c)
            append!(preds, y)
            append!(golds, t.data)
        end
        length(preds) == length(golds) || throw("Length mismatch: $(length(preds)), $(length(golds))")

        preds = BIOES.decode(preds, tagdict)
        golds = BIOES.decode(golds, tagdict)
        fscore(golds, preds)
        println()
    end
    Decoder(worddict1, worddict2, chardict, tagdict, nn)
end

function initvocab(path::String)
    worddict = Dict{String,Int}()
    chardict = Dict{String,Int}()
    tagdict = Dict{String,Int}()
    lines = open(readlines, path)
    for line in lines
        isempty(line) && continue
        items = Vector{String}(split(line,"\t"))
        word = strip(items[1])
        word0 = replace(lowercase(word), r"[0-9]", '0')
        if haskey(worddict, word0)
            worddict[word0] += 1
        else
            worddict[word0] = 1
        end
        chars = Vector{Char}(word)
        for c in chars
            c = string(c)
            if haskey(chardict, c)
                chardict[c] += 1
            else
                chardict[c] = 1
            end
        end
        tag = strip(items[2])
        haskey(tagdict,tag) || (tagdict[tag] = length(tagdict)+1)
    end

    words = String[]
    for (k,v) in worddict
        v >= 3 && push!(words,k)
    end
    worddict = Dict(words[i] => i for i=1:length(words))
    worddict["UNKNOWN"] = length(worddict) + 1

    chars = String[]
    for (k,v) in chardict
        v >= 3 && push!(chars,k)
    end
    chardict = Dict(chars[i] => i for i=1:length(chars))
    chardict["UNKNOWN"] = length(chardict) + 1

    worddict, chardict, tagdict
end

function decode(dec::Decoder, path::String)
    data = readdata(path, dec.worddict1, dec.worddict2, dec.chardict, dec.tagdict)
    data = batch(data, 100)
    id2tag = Array{String}(length(dec.tagdict))
    for (k,v) in dec.tagdict
        id2tag[v] = k
    end

    preds = Int[]
    for x in data
        y = dec.nn(x[1], x[2], x[3])
        append!(preds, y)
    end

    lines = open(readlines, path)
    i = 1
    for line in lines
        isempty(line) && continue
        tag = id2tag[preds[i]]
        println("$line\t$tag")
        i += 1
    end
end

function readdata(path::String, worddict1::Dict, worddict2::Dict, chardict::Dict, tagdict::Dict)
    data = Tuple[]
    words, tags = String[], String[]
    unkword1 = worddict1["UNKNOWN"]
    unkword2 = worddict2["UNKNOWN"]
    unkchar = chardict["UNKNOWN"]

    lines = open(readlines, path)
    push!(lines, "")
    for i = 1:length(lines)
        line = lines[i]
        if isempty(line)
            isempty(words) && continue
            wordids1 = Int[]
            wordids2 = Int[]
            charbatchdims = Int[]
            charids = Int[]
            for w in words
                w0 = replace(lowercase(w), r"[0-9]", '0')
                id = get(worddict1, w0, unkword1)
                push!(wordids1, id)
                id = get(worddict2, w0, unkword2)
                push!(wordids2, id)

                chars = Vector{Char}(w)
                push!(charbatchdims, length(chars))
                for c in chars
                    push!(charids, get(chardict,string(c),unkchar))
                end
            end
            w1 = Var(wordids1)
            w2 = Var(wordids2)
            c = Var(charids, charbatchdims)

            if isempty(tags)
                push!(data, (w1,w2,c))
            else
                tagids = map(t -> tagdict[t], tags)
                t = Var(tagids)
                push!(data, (w1,w2,c,t))
            end
            empty!(words)
            empty!(tags)
        else
            items = Vector{String}(split(line,"\t"))
            word = strip(items[1])
            @assert !isempty(word)
            push!(words, word)
            if length(items) >= 2
                tag = strip(items[2])
                push!(tags, tag)
            end
        end
    end
    Array{typeof(data[1])}(data)
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
