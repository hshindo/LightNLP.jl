const BACKEND = CUDABackend()

mutable struct Decoder
    worddict::Dict
    posdict::Dict
    nn
end

struct Sample
    w::Var
    c::Var
    batchdims_w
    batchdims_c
    p::Var
    heads
end

function create_batch(samples::Vector{Sample}, batchsize::Int)
    batches = Sample[]
    for i = 1:batchsize:length(samples)
        range = i:min(i+batchsize-1,length(samples))
        s = samples[range]
        sort!(s, by = x -> x.batchdims_w[1], rev=true)
        w = cat(2, map(x -> x.w.data, s)...)
        w = Var(BACKEND(w))
        c = cat(2, map(x -> x.c.data, s)...)
        c = Var(BACKEND(c))
        batchdims_w = cat(1, map(x -> x.batchdims_w, s)...)
        batchdims_c = cat(1, map(x -> x.batchdims_c, s)...)
        p = cat(2, map(x -> x.p.data, s)...)
        p = Var(BACKEND(p))
        heads = map(x -> Var(BACKEND(x.heads.data)), s)
        push!(batches, Sample(w,c,batchdims_w,batchdims_c,p,heads))
    end
    batches
end

function Decoder(config::Dict)
    chardict = Dict("UNKNOWN" => 1)
    words = h5read(config["wordvec_file"], "words")
    wordembeds = h5read(config["wordvec_file"], "vectors")
    worddict = Dict(words[i] => i for i=1:length(words))
    posdict = Dict{String,Int}()
    traindata = readconll(config["train_file"], worddict, chardict, posdict)[1:10000]
    testdata = readconll(config["test_file"], worddict, chardict, posdict)
    posembeds = Uniform(-0.01,0.01)(Float32, 50, length(posdict))
    nn = NN(wordembeds, posembeds)

    info("#Training examples:\t$(length(traindata))")
    info("#Testing examples:\t$(length(testdata))")
    info("#Words:\t$(length(worddict))")
    info("#Chars:\t$(length(chardict))")
    info("#POS-tags:\t$(length(posdict))")
    testdata = create_batch(testdata, 100)

    opt = SGD()
    batchsize = config["batchsize"]
    for epoch = 1:config["nepochs"]
        println("Epoch:\t$epoch")
        #opt.rate = learnrate / batchsize
        opt.rate = config["learning_rate"] / batchsize * sqrt(batchsize) / (1 + 0.05*(epoch-1))
        println("Learning rate: $(opt.rate)")

        shuffle!(traindata)
        batches = create_batch(traindata, batchsize)
        prog = Progress(length(batches))
        loss = 0.0
        for i in 1:length(batches)
            s = batches[i]
            y = nn(s, true)
            loss += sum(Array(y.data))
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
            y = nn(s, false)
            append!(preds, y)
            for h in s.heads
                append!(golds, Array(h.data))
            end
        end
        length(preds) == length(golds) || throw("Length mismatch: $(length(preds)), $(length(golds))")
        count = 0
        for k = 1:length(preds)
            preds[k] == golds[k] && (count += 1)
        end
        acc = round(count/length(preds), 5)
        println("Acc:\t$acc")
        println()
    end
    Decoder(worddict, posdict, nn)
end

function initvocab(path::String)
    chardict = Dict{String,Int}()
    #tagdict = Dict{String,Int}()
    lines = open(readlines, path)
    for line in lines
        isempty(line) && continue
        items = Vector{String}(split(line,"\t"))
        word = strip(items[2])
        chars = Vector{Char}(word)
        for c in chars
            c = string(c)
            if haskey(chardict, c)
                chardict[c] += 1
            else
                chardict[c] = 1
            end
        end
        #tag = strip(items[2])
        #haskey(tagdict,tag) || (tagdict[tag] = length(tagdict)+1)
    end

    chars = String[]
    for (k,v) in chardict
        v >= 3 && push!(chars,k)
    end
    chardict = Dict(chars[i] => i for i=1:length(chars))
    chardict["UNKNOWN"] = length(chardict) + 1
    chardict
end

function readconll(path::String, worddict::Dict, chardict::Dict, posdict::Dict)
    samples = Sample[]
    words = String[]
    postags = String[]
    heads = Int[]
    unkword = worddict["UNKNOWN"]
    unkchar = chardict["UNKNOWN"]

    lines = open(readlines, path)
    push!(lines, "")
    for i = 1:length(lines)
        line = lines[i]
        if isempty(line)
            isempty(words) && continue
            wordids = Int[]
            charids = Int[]
            batchdims_w = [length(words)]
            batchdims_c = Int[]
            for w in words
                w0 = replace(lowercase(w), r"[0-9]", '0')
                id = get(worddict, w0, unkword)
                push!(wordids, id)

                chars = Vector{Char}(w)
                ids = map(chars) do c
                    get(chardict, string(c), unkchar)
                end
                append!(charids, ids)
                push!(batchdims_c, length(ids))
            end
            posids = Int[]
            for p in postags
                id = get!(posdict, p, length(posdict)+1)
                push!(posids, id)
            end

            w = Var(reshape(wordids,1,length(wordids)))
            c = Var(reshape(charids,1,length(charids)))
            p = Var(reshape(posids,1,length(posids)))
            h = Var(heads)
            push!(samples, Sample(w,c,batchdims_w,batchdims_c,p,h))
            empty!(words)
            empty!(postags)
            heads = Int[]
        else
            items = Vector{String}(split(line,"\t"))
            word = strip(items[2])
            @assert !isempty(word)
            push!(words, word)
            push!(postags, strip(items[6]))
            push!(heads, parse(Int,items[9]))
        end
    end
    samples
end
