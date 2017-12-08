export decode

mutable struct Decoder
    worddict1::Dict
    worddict2::Dict
    chardict::Dict
    tagdict::Dict
    nn
end

struct Sample
    w1::Var
    w2::Var
    c::Var
    batchdims_w
    batchdims_c
    t::Var
end

function create_batch(samples::Vector{Sample}, batchsize::Int)
    batches = Sample[]
    for i = 1:batchsize:length(samples)
        range = i:min(i+batchsize-1,length(samples))
        s = samples[range]
        w1 = Var(cat(1, map(x -> x.w1.data, s)...))
        w2 = Var(cat(1, map(x -> x.w2.data, s)...))
        c = Var(cat(1, map(x -> x.c.data, s)...))
        batchdims_w = cat(1, map(x -> x.batchdims_w, s)...)
        batchdims_c = cat(1, map(x -> x.batchdims_c, s)...)
        t = s[1].t == nothing ? nothing : Var(cat(1, map(x -> x.t.data, s)...))
        push!(batches, Sample(w1,w2,c,batchdims_w,batchdims_c,t))
    end
    batches
end

function Decoder(embedsfile::String, trainfile::String, testfile::String, nepochs::Int, learnrate::Float64, batchsize::Int)
    words1 = h5read(embedsfile, "key")
    worddict1 = Dict(words1[i] => i for i=1:length(words1))
    wordembeds1 = Var(h5read(embedsfile,"value"))
    #wordembeds1 = [zerograd(wordembeds1[:,i]) for i=1:size(wordembeds1,2)]
    worddict2, chardict, tagdict = initvocab(trainfile)
    wordembeds2 = embeddings(Float32, length(worddict2), 100, init_w=Uniform(-0.01,0.01))
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
    testdata = create_batch(testdata, 100)

    opt = SGD()
    for epoch = 1:nepochs
        println("Epoch:\t$epoch")
        opt.rate = learnrate * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))
        println("Learning rate: $(opt.rate)")

        shuffle!(traindata)
        batches = create_batch(traindata, batchsize)
        prog = Progress(length(batches))
        loss = 0.0
        for i in 1:length(batches)
            s = batches[i]
            y = nn.g("w1"=>s.w1, "w2"=>s.w2, "c"=>s.c, "batchdims_c"=>s.batchdims_c, "batchdims_w"=>s.batchdims_w, "train"=>true)
            y = softmax_crossentropy(s.t, y)
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
        for s in testdata
            y = nn.g("w1"=>s.w1, "w2"=>s.w2, "c"=>s.c, "batchdims_c"=>s.batchdims_c, "batchdims_w"=>s.batchdims_w, "train"=>false)
            y = argmax(y.data, 1)
            append!(preds, y)
            append!(golds, s.t.data)
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

function decode2(dec::Decoder, path::String)
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
    samples = Sample[]
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
            charids = Int[]
            batchdims_c = Int[]
            for w in words
                w0 = replace(lowercase(w), r"[0-9]", '0')
                id = get(worddict1, w0, unkword1)
                push!(wordids1, id)
                id = get(worddict2, w0, unkword2)
                push!(wordids2, id)

                chars = Vector{Char}(w)
                ids = map(chars) do c
                    get(chardict, string(c), unkchar)
                end
                append!(charids, ids)
                push!(batchdims_c, length(ids))
            end
            batchdims_w = [length(words)]
            w1, w2, c = Var(wordids1), Var(wordids2), Var(charids)
            t = isempty(tags) ? nothing : Var(map(t -> tagdict[t], tags))
            push!(samples, Sample(w1,w2,c,batchdims_w,batchdims_c,t))
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
    samples
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
