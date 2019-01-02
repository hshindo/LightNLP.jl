using Unicode

struct Dataset
    data::Vector
    training::Bool
end

Base.length(dataset::Dataset) = length(dataset.data)

function Merlin.todevice(dataset::Dataset)
    data = map(dataset.data) do (w,c,dims_c,t,u)
        w = todevice(w)
        c = todevice(c)
        t = todevice(t)
        u = todevice(u)
        (w, c, dims_c, t, u)
    end
    Dataset(data, dataset.training)
end

function Base.getindex(dataset::Dataset, indexes::Vector{Int})
    data = dataset.data[indexes]
    data = sort(data, by=x->length(x[1]), rev=true)
    ws = map(x -> x[1], data)
    dims_w = length.(ws)
    w = cat(ws..., dims=1)
    w = reshape(w, 1, length(w))
    c = cat(map(x -> x[2], data)..., dims=2)
    # c = reshape(c, 1, length(c))
    dims_c = cat(map(x -> x[3], data)..., dims=1)
    t = cat(map(x -> x[4], data)..., dims=1)
    u = cat(map(x -> x[5], data)..., dims=1)
    u = reshape(u, 1, length(u))
    (w=Var(w), c=Var(c), dims_w=dims_w, dims_c=dims_c, t=Var(t), u=Var(u), training=dataset.training)
end

function readconll(path::String, dicts, training::Bool)
    if training
        @assert isempty(dicts.c)
        dicts.c["unk"] = 1
        dicts.c["lowercase"] = 2
        dicts.c["uppercase"] = 3
        @assert isempty(dicts.t)
        dicts.t["O"] = 1
    end
    data = []
    words = String[]
    tagids = Int[]
    unkword = dicts.w["unk"]
    unkchar, lowerchar, upperchar = dicts.c["unk"], dicts.c["lowercase"], dicts.c["uppercase"]

    lines = open(readlines, path)
    push!(lines, "")
    for i = 1:length(lines)
        line = lines[i]
        if isempty(line)
            isempty(words) && continue
            wordids = Int[]
            charids = []
            chardims = Int[]
            for w in words
                # w0 = replace(lowercase(w), r"[0-9]", '0')
                id = get(dicts.w, lowercase(w), unkword)
                push!(wordids, id)

                chars = Vector{Char}(w)
                push!(chardims, length(chars))
                cmat = Array{Int}(undef, 2, length(chars))
                for k = 1:length(chars)
                    c = chars[k]
                    cc = lowercase(c)
                    cmat[1,k] = training ? get!(dicts.c,string(cc),length(dicts.c)+1) : get(dicts.c,string(cc),unkchar)
                    cmat[2,k] = islowercase(c) ? lowerchar : upperchar
                    #cmat[2,k] = lowerchar
                end
                push!(charids, cmat)
            end
            charids = cat(charids..., dims=2)
            predtags = Int[tagids[k] for k=1:length(tagids)-1]
            pushfirst!(predtags, 1)
            push!(data, (wordids,charids,chardims,tagids,predtags))
            words = String[]
            tagids = Int[]
        else
            items = Vector{String}(split(line,"\t"))
            word = strip(items[1])
            isempty(word) && throw("line $i")
            push!(words, word)
            if length(items) >= 2
                tag = strip(items[2])
                id = training ? get!(dicts.t,tag,length(dicts.t)+1) : dicts.t[tag]
                push!(tagids, id)
                if training
                    get!(dicts.l, tag[3:end], length(dicts.l)+1)
                end
            else
                throw("")
            end
        end
    end
    data = Vector{typeof(data[1])}(data)
    Dataset(data, training)
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
    chardict["LOWERCASE"] = length(chardict) + 1
    chardict["UPPERCASE"] = length(chardict) + 1
    chardict[" "] = length(chardict) + 1
    chardict, tagdict
end
