mutable struct Decoder
    worddict::Dict
    posdict::Dict
    nn
end

struct Sample
    word::Var
    char::Var
    batchdims_w
    batchdims_c
    pos::Var
    head
end

function create_batch(samples::Vector{Sample}, batchsize::Int)
    batches = Sample[]
    for i = 1:batchsize:length(samples)
        range = i:min(i+batchsize-1,length(samples))
        s = samples[range]
        word = Var(cat(1, map(x -> x.word.data, s)...))
        char = Var(cat(1, map(x -> x.char.data, s)...))
        batchdims_w = cat(1, map(x -> x.batchdims_w, s)...)
        batchdims_c = cat(1, map(x -> x.batchdims_c, s)...)
        pos = Var(cat(1, map(x -> x.pos.data, s)...))
        head = map(x -> x.head, s)
        #t = s[1].t == nothing ? nothing : Var(cat(1, map(x -> x.t.data, s)...))
        push!(batches, Sample(word,char,batchdims_w,batchdims_c,pos,head))
    end
    batches
end

function Decoder(embedsfile::String, trainfile::String, testfile::String, nepochs::Int, learnrate::Float64, batchsize::Int)
    words = h5read(embedsfile, "words")
    worddict = Dict(words[i] => i for i=1:length(words))
    w = h5read(embedsfile, "vectors")
    wordembeds = [zerograd(w[:,i]) for i=1:size(w,2)]
    chardict = initvocab(trainfile)
    #charembeds = embeddings(Float32, length(chardict), 20, init_w=Normal(0,0.01))
    posdict = Dict{String,Int}()
    traindata = readconll(trainfile, worddict, chardict, posdict)[1:10000]
    testdata = readconll(testfile, worddict, chardict, posdict)
    posembeds = embeddings(Float32, length(posdict), 50)
    nn = NN(wordembeds, posembeds)

    info("#Training examples:\t$(length(traindata))")
    info("#Testing examples:\t$(length(testdata))")
    info("#Words:\t$(length(worddict))")
    info("#Chars:\t$(length(chardict))")
    info("#POS-tags:\t$(length(posdict))")
    testdata = create_batch(testdata, 100)

    opt = SGD()
    for epoch = 1:nepochs
        println("Epoch:\t$epoch")
        #opt.rate = learnrate / batchsize
        opt.rate = learnrate * batchsize / sqrt(batchsize) / (1 + 0.05*(epoch-1))
        println("Learning rate: $(opt.rate)")

        shuffle!(traindata)
        batches = create_batch(traindata, batchsize)
        prog = Progress(length(batches))
        loss = 0.0
        losscount = 0
        for i in 1:length(batches)
            s = batches[i]
            os = nn(s.word, s.pos, s.batchdims_w, true)
            ys = Var[]
            for (h,o) in zip(s.head,os)
                y = softmax_crossentropy(h, o)
                push!(ys, y)
                loss += sum(y.data)
                losscount += length(y.data)
            end
            params = gradient!(ys...)
            foreach(opt, params)
            ProgressMeter.next!(prog)
        end
        loss /= losscount
        println("Loss:\t$loss")

        # test
        println("Testing...")
        preds = Int[]
        golds = Int[]
        for s in testdata
            os = nn(s.word, s.pos, s.batchdims_w, false)
            for (h,o) in zip(s.head,os)
                y = argmax(o, 1)
                append!(preds, y)
                append!(golds, h.data)
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

            w = Var(wordids)
            c = Var(charids)
            p = Var(posids)
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
