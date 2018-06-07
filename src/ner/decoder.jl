export decode

mutable struct Decoder
    worddict::Dict
    chardict::Dict
    tagdict::Dict
    nn
    config
end

struct Sample
    w
    c
    t
end

function Base.cat(xs::Vector{Sample})
    w = map(x -> x.w, xs)
    c = Vector{Int}[]
    for x in xs
        append!(c, x.c)
    end
    t = map(x -> x.t, xs)
    Sample(w, c, t)
end

function Decoder(config::Dict)
    words = h5read(config["wordvec_file"], "words")
    worddict = Dict(words[i] => i for i=1:length(words))
    wordembeds = h5read(config["wordvec_file"], "vectors")

    chardict, tagdict = initvocab(config["train_file"])
    charembeds = Normal(0,0.01)(eltype(wordembeds), 20, length(chardict))

    traindata = readdata(config["train_file"], worddict, chardict, tagdict)
    testdata = readdata(config["test_file"], worddict, chardict, tagdict)
    nn = NN(wordembeds, charembeds, length(tagdict))

    info("#Training examples:\t$(length(traindata))")
    info("#Testing examples:\t$(length(testdata))")
    info("#Words:\t$(length(worddict))")
    info("#Chars:\t$(length(chardict))")
    info("#Tags:\t$(length(tagdict))")
    # testdata = create_batch(testdata, 100)
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

        shuffle!(traindata)
        samples = create_batch(cat, batchsize, traindata)
        prog = Progress(length(samples))
        loss = 0.0
        for s in samples
            z = nn.g(s.batchsize_c, s.batchsize_w, s.c, s.w)
            softmax_crossentropy(Var(x.t), z)
            loss += sum(z.data)
            params = gradient!(z)
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
            z = nn(s, false)
            argmax(z.data, 1)
            append!(preds, z)
            append!(golds, s.t)
        end
        length(preds) == length(golds) || throw("Length mismatch: $(length(preds)), $(length(golds))")

        preds = bioes_decode(preds, tagdict)
        golds = bioes_decode(golds, tagdict)
        fscore(golds, preds)
        println()
    end
end

function initvocab(path::String)
    chardict = Dict{String,Int}()
    tagdict = Dict{String,Int}()
    lines = open(readlines, path)
    for line in lines
        isempty(line) && continue
        items = Vector{String}(split(line,"\t"))
        word = strip(items[1])
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

    chars = String[]
    for (k,v) in chardict
        v >= 3 && push!(chars,k)
    end
    chardict = Dict(chars[i] => i for i=1:length(chars))
    chardict["UNKNOWN"] = length(chardict) + 1

    chardict, tagdict
end

function readdata(path::String, worddict::Dict, chardict::Dict, tagdict::Dict)
    samples = Sample[]
    words = String[]
    tagids = Int[]
    unkword = worddict["UNKNOWN"]
    unkchar = chardict["UNKNOWN"]

    lines = open(readlines, path)
    push!(lines, "")
    for i = 1:length(lines)
        line = lines[i]
        if isempty(line)
            isempty(words) && continue
            wordids = Int[]
            charids = Vector{Int}[]
            for w in words
                # w0 = replace(lowercase(w), r"[0-9]", '0')
                id = get(worddict, lowercase(w), unkword)
                push!(wordids, id)

                chars = Vector{Char}(w)
                ids = map(chars) do c
                    get(chardict, string(c), unkchar)
                end
                push!(charids, ids)
            end
            w = reshape(wordids, 1, length(wordids))
            c = reshape(charids, 1, length(charids))
            t = reshape(copy(tagids), 1, length(tagids))
            push!(samples, Sample(w,c,t))
            empty!(words)
            empty!(tagids)
        else
            items = Vector{String}(split(line,"\t"))
            word = strip(items[1])
            @assert !isempty(word)
            push!(words, word)
            if length(items) >= 2
                tag = strip(items[2])
                push!(tagids, tagdict[tag])
            else
                push!(tagids, 0)
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
