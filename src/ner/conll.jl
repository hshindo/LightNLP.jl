struct CoNLL
    data::Vector
end

Base.length(conll::CoNLL) = length(conll.data)

function Merlin.todevice(conll::CoNLL)
    dev = Merlin.getdevice()
    dev < 0 && return conll
    data = map(conll.data) do (w,c,dims_c,t)
        w = todevice(w)
        c = todevice(c)
        t = todevice(t)
        (w, c, dims_c, t)
    end
    CoNLL(data)
end

function Base.getindex(conll::CoNLL, indexes::Vector{Int})
    data = conll.data[indexes]
    data = sort(data, by=x->length(x[1]), rev=true)
    ws = map(x -> x[1], data)
    dims_w = length.(ws)
    w = cat(ws..., dims=1)
    w = reshape(w, 1, length(w))
    c = cat(map(x -> x[2], data)..., dims=1)
    c = reshape(c, 1, length(c))
    dims_c = cat(map(x -> x[3], data)..., dims=1)
    t = cat(map(x -> x[4], data)..., dims=1)
    (w=Var(w), c=Var(c), dims_w=dims_w, dims_c=dims_c, t=Var(t))
end

function CoNLL(path::String, worddict::Dict, chardict::Dict, tagdict::Dict)
    data = []
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
            charids = Int[]
            chardims = Int[]
            for w in words
                # w0 = replace(lowercase(w), r"[0-9]", '0')
                id = get(worddict, lowercase(w), unkword)
                push!(wordids, id)

                chars = Vector{Char}(w)
                push!(chardims, length(chars))
                cids = map(chars) do c
                    get(chardict, string(c), unkchar)
                end
                append!(charids, cids)
            end
            push!(data, (wordids,charids,chardims,tagids))
            words = String[]
            tagids = Int[]
        else
            items = Vector{String}(split(line,"\t"))
            word = strip(items[1])
            @assert !isempty(word)
            push!(words, word)
            if length(items) >= 2
                tag = strip(items[2])
                push!(tagids, tagdict[tag])
            else
                throw("")
            end
        end
    end
    data = Vector{typeof(data[1])}(data)
    CoNLL(data)
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
