function catsample(samples::Vector)
    samples = sort(sample, by=s->length(s[1]), rev=true)
    batchdims_w = Int[]
    batchdims_c = Int[]
    w = Int[]
    c = Int[]
    t = Int[]
    for (_w,_cs,_t) in samples
        push!(batchdims_w, length(_w))
        append!(w, _w)
        for _c in _cs
            push!(batchdims_c, length(_c))
            append!(c, _c)
        end
        append!(t, _t)
    end
    w = Var(reshape(w,1,length(w)))
    c = Var(reshape(c,1,length(c)))
    t = Var(t)
    (batchdims_c,batchdims_w,c,w), t
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

function readconll(path::String, worddict::Dict, chardict::Dict, tagdict::Dict)
    samples = []
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
            push!(samples, (wordids,charids,tagids))
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
    samples
end
