using Unicode

struct Dataset
    data::Vector
end

Base.length(dataset::Dataset) = length(dataset.data)
Base.getindex(dataset::Dataset, index::Int) = dataset.data[index]

function Merlin.todevice(dataset::Dataset)
    data = map(dataset.data) do (w,c,dims_c,t,s,dims_s,categ,count)
        w = todevice(w)
        c = todevice(c)
        t = todevice(t)
        s = todevice(s)
        categ = todevice(categ)
        #count = todevice(count)
        (w=w, c=c, dims_c=dims_c, t=t, s=s, dims_s=dims_s, categ=categ)
    end
    Dataset(data)
end

function Base.getindex(dataset::Dataset, indexes::Vector{Int})
    data = dataset.data[indexes]
    # data = sort(data, by=x->length(x[1]), rev=true)
    ws = map(x -> x.w, data)
    dims_w = length.(ws)
    w = cat(ws..., dims=1)
    w = reshape(w, 1, length(w))
    c = cat(map(x -> x.c, data)..., dims=2)
    dims_c = cat(map(x -> x.dims_c, data)..., dims=1)
    t = cat(map(x -> x.t, data)..., dims=1)

    offs = 0
    s = map(data) do x
        s = copy(x.s)
        addto!(s, Cint(offs))
        offs += length(x.w)
        s
    end
    s = cat(s..., dims=1)
    s = reshape(s, 1, length(s))
    dims_s = cat(map(x -> x.dims_s, data)..., dims=1)
    categ = cat(map(x -> x.categ, data)..., dims=1)
    # count = cat(map(x -> x[5], data)..., dims=1)
    # count = reshape(count, 1, length(count))
    (w=Var(w), c=Var(c), dims_w=dims_w, dims_c=dims_c, t=Var(t), s=Var(s), dims_s=dims_s, categ=Var(categ))
end

function readconll(path::String, worddict, chardict, tagdict, training::Bool)
    data = []
    words = String[]
    tags = String[]
    unkword = get!(worddict, "unk")
    unkchar = get!(chardict, "unk")

    lines = open(readlines, path)
    push!(lines, "")
    wordcount = 1
    for i = 1:length(lines)
        line = lines[i]
        if isempty(line)
            isempty(words) && continue
            wordids = Int[]
            charids = Matrix{Int}[]
            chardims = Int[]
            countids = Int[]
            for w in words
                # w0 = replace(lowercase(w), r"[0-9]"=>'0')
                id = get(worddict, lowercase(w), unkword)
                push!(wordids, id)
                push!(countids, wordcount)
                wordcount += 1

                # w0 = replace(w, r"[0-9]"=>'0')
                chars = Vector{Char}(w)
                push!(chardims, length(chars))
                #=
                for c in chars
                    c = string(c)
                    charid = training ? get!(chardict,c) : get(chardict,c,unkchar)
                    push!(charids, charid)
                end
                =#

                cmat = Array{Int}(undef, 3, length(chars))
                for k = 1:length(chars)
                    c = string(chars[k])
                    if occursin(r"[A-Z]", c)
                        case = "UPPER"
                    elseif occursin(r"[a-z]", c)
                        case = "LOWER"
                    elseif occursin(r"[0-9]", c)
                        case = "NUMBER"
                    else
                        case = "MISC"
                    end
                    lc = lowercase(c)
                    cmat[1,k] = training ? get!(chardict,c) : get(chardict,c,unkchar)
                    cmat[2,k] = training ? get!(chardict,lc) : get(chardict,lc,unkchar)
                    cmat[3,k] = training ? get!(chardict,case) : chardict[case]
                end
                push!(charids, cmat)
                #=
                chars = Vector{Char}(lowercase(w))
                pushfirst!(chars, ' ')
                push!(chars, ' ')
                for k = 2:length(chars)-1
                    c = string(chars[k-1:k+1])
                    # charid = training ? get!(chardict,c) : get(chardict,c,unkchar)
                    cmat[4,k-1] = training ? get!(chardict,c) : get(chardict,c,unkchar)
                end
                =#
            end
            charids = cat(charids..., dims=2)
            # charids = reshape(charids, 1, length(charids))
            tagids, spanids, spandims, categids = bioes_encode!(tagdict, tags)
            push!(data, (wordids,charids,chardims,tagids,spanids,spandims,categids,countids))
            empty!(words)
            empty!(tags)
        else
            items = Vector{String}(split(line,"\t"))
            word = strip(items[1])
            isempty(word) && throw("Line $i: empty word is detected.")
            push!(words, word)
            if length(items) >= 2
                tag = strip(items[2])
                isempty(tag) && throw("Invalid tag.")
                push!(tags, tag)
            else
                throw("Line $i: invalid line.")
            end
        end
    end
    data = Vector{typeof(data[1])}(data)

    if training
        for x in data
            charids = x[2]
            for k = 1:length(charids)
                if count(chardict,charids[k]) < 5
                    charids[k] = unkchar
                end
            end
        end
    end
    Dataset(data)
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
