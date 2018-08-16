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
