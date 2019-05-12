using Unicode

mutable struct Sample
    word
    dims_word
    char
    dims_char
    tag
    span
    dims_span
    categ
end

function Sample(samples::Vector, indexes::Vector{Int})
    data = samples[indexes]
    # data = sort(data, by=x->length(x[1]), rev=true)
    ws = map(x -> x.word, data)
    dims_word = length.(ws)
    w = cat(ws..., dims=1)
    w = reshape(w, 1, length(w)) |> todevice
    c = cat(map(x -> x.char, data)..., dims=2) |> todevice
    dims_char = cat(map(x -> x.dims_char, data)..., dims=1)
    tag = cat(map(x -> x.tag, data)..., dims=1) |> todevice

    offs = 0
    s = map(data) do x
        s = x.span .+ offs
        offs += length(x.word)
        s
    end
    s = cat(s..., dims=1)
    span = reshape(s, 1, length(s)) |> todevice
    dims_span = cat(map(x -> x.dims_span, data)..., dims=1)
    categ = cat(map(x -> x.categ, data)..., dims=1) |> todevice

    # count = cat(map(x -> x.count, data)..., dims=1)
    # flair = cat(map(c -> dataset.flair[:,c], count)..., dims=2) |> todevice
    # count = reshape(count, 1, length(count)) |> todevice
    Sample(Var(w), dims_word, Var(c), dims_char, Var(tag), Var(span), dims_span, Var(categ))
end

function readconll(path::String, worddict, chardict, tagdict, training::Bool)
    data = Sample[]
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

                w0 = replace(w, r"[0-9]"=>'0')
                chars = Vector{Char}(w0)
                push!(chardims, length(chars))

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
                        case = c
                    end
                    lc = lowercase(c)
                    cmat[1,k] = training ? get!(chardict,c) : get(chardict,c,unkchar)
                    cmat[2,k] = training ? get!(chardict,lc) : get(chardict,lc,unkchar)
                    cmat[3,k] = training ? get!(chardict,case) : get(chardict,case,unkchar)
                end
                push!(charids, cmat)
            end
            charids = cat(charids..., dims=2)
            # charids = reshape(charids, 1, length(charids))
            tagids, spanids, spandims, categids = bioes_encode!(tagdict, tags)
            sample = Sample(wordids, nothing, charids, chardims, tagids, spanids, spandims, categids)
            length(wordids) > 1 && push!(data, sample)
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
    data
end
