using Unicode

struct Dataset
    data::Vector
end

Base.length(dataset::Dataset) = length(dataset.data)

function Merlin.todevice(dataset::Dataset)
    data = map(dataset.data) do (w,c,dims_c,t)
        w = todevice(w)
        c = todevice(c)
        t = todevice(t)
        (w=w, c=c, dims_c=dims_c, t=t)
    end
    Dataset(data)
end

function Base.getindex(dataset::Dataset, indexes::Vector{Int})
    data = dataset.data[indexes]
    data = sort(data, by=x->length(x[1]), rev=true)
    ws = map(x -> x[1], data)
    dims_w = length.(ws)
    w = cat(ws..., dims=1)
    w = reshape(w, 1, length(w))
    c = cat(map(x -> x[2], data)..., dims=2)
    dims_c = cat(map(x -> x[3], data)..., dims=1)
    t = cat(map(x -> x[4], data)..., dims=1)
    (w=Var(w), c=Var(c), dims_w=dims_w, dims_c=dims_c, t=Var(t))
end

function readconll(path::String, dicts, training::Bool)
    if training
        @assert isempty(dicts.char)
        dicts.char["unk"] = 1
        dicts.char["lowercase"] = 2
        dicts.char["uppercase"] = 3
        @assert isempty(dicts.tag)
        dicts.tag["O"] = 1
    end
    data = []
    words = String[]
    tagids = Int[]
    unkword = dicts.word["unk"]
    unkchar, lowerchar, upperchar = dicts.char["unk"], dicts.char["lowercase"], dicts.char["uppercase"]

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
                id = get(dicts.word, lowercase(w), unkword)
                push!(wordids, id)

                chars = Vector{Char}(w)
                push!(chardims, length(chars))
                cmat = Array{Int}(undef, 2, length(chars))
                for k = 1:length(chars)
                    c = chars[k]
                    lc = lowercase(c)
                    cmat[1,k] = training ? get!(dicts.char,string(lc),length(dicts.char)+1) : get(dicts.char,string(lc),unkchar)
                    cmat[2,k] = islowercase(c) ? lowerchar : upperchar
                end
                push!(charids, cmat)
            end
            charids = cat(charids..., dims=2)
            push!(data, (wordids,charids,chardims,tagids))
            words = String[]
            tagids = Int[]
        else
            items = Vector{String}(split(line,"\t"))
            word = strip(items[1])
            isempty(word) && throw("line $i: Empty word is detected.")
            push!(words, word)
            if length(items) >= 2
                tag = strip(items[2])
                tag == "O" || length(tag) >= 3 || tag[2] == '-' || throw("line $i: Invalid tag is detected.")
                # tag[1] == 'I' && (tag = "O")
                # cat = length(tag) == 1 ? "" : tag[3:end]
                id = training ? get!(dicts.tag,tag,length(dicts.tag)+1) : dicts.tag[tag]
                push!(tagids, id)
            else
                throw("line $i: Invalid line.")
            end
        end
    end
    data = Vector{typeof(data[1])}(data)
    Dataset(data)
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
