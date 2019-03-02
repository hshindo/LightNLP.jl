function bioes_encode!(tagdict::IntDict{String}, tags::Vector{String})
    tag2id = Dict("O"=>1, "B"=>2, "I"=>3, "E"=>4, "S"=>5)
    tagids = map(tags) do tag
        t = tag == "O" ? tag : tag[1:1]
        tag2id[t]
    end

    spanids = Int[]
    spandims = Int[]
    categids = Int[]
    i = 1
    while i <= length(tags)
        tag = tags[i]
        tag == "O" && (i += 1; continue)
        categid = get!(tagdict, tag[3:end])
        if startswith(tag, "B-")
            I = "I-" * tag[3:end]
            E = "E-" * tag[3:end]
            j = findnext(t -> t != I, tags, i+1)
            tags[j] == E || throw("Invalid BIOES.")
            append!(spanids, i:j)
            push!(spandims, j-i+1)
            push!(categids, categid)
            i = j
        elseif startswith(tag, "S-")
            push!(spanids, i)
            push!(spandims, 1)
            push!(categids, categid)
        end
        i += 1
    end
    tagids, spanids, spandims, categids
end

function bioes2span(tagids::Vector{Int})
    id2tag = ["O", "B", "I", "E", "S"]
    spans = Tuple{Int,Int}[]
    bpos = 0
    for i = 1:length(tagids)
        tag = id2tag[tagids[i]]
        tag == "O" && continue
        tag == "B" && (bpos = i)
        tag == "S" && (bpos = i)
        if tag == "S" || (tag == "E" && bpos > 0)
            push!(spans, (bpos,i))
            bpos = 0
        end
    end
    spans
end

#=
function bioes_decode(catdict::IntDict{String}, tagids::Vector{Int})
    id2tag = ["O", "B", "I", "E", "S"]
    spans = Tuple{Int,Int,String}[]
    bpos = 0
    for i = 1:length(tagids)
        tag = id2tag[tagids[i]]
        tag == "O" && continue
        startswith(tag,"B") && (bpos = i)
        startswith(tag,"S") && (bpos = i)
        if (startswith(tag,"S") || startswith(tag,"E")) && bpos > 0
            push!(spans, (bpos,i,""))
            bpos = 0
        end
    end
    spans
end

function bioes_decode(tagids::Vector{Int})
    id2tag = ["O", "B", "I", "E", "S"]
    s = Int[]
    dims_s = Int[]
    bpos = 0
    for i = 1:length(tagids)
        tag = id2tag[tagids[i]]
        tag == "O" && continue
        tag == "B" && (bpos = i)
        tag == "S" && (bpos = i)
        if tag == "S" || tag == "E" && bpos > 0
            append!(s, bpos:i)
            push!(dims_s, i-bpos+1)
            bpos = 0
        end
    end
    s = reshape(s, 1, length(s))
    s, dims_s
end
=#

function bioes_check(ids::Vector{Int}, tagdict::Dict{String,Int})
    id2tag = Array{String}(undef, length(tagdict))
    for (k,v) in tagdict
        id2tag[v] = k
    end
    count = 0
    for i = 1:length(ids)-1
        tag1 = id2tag[ids[i]][1]
        tag2 = id2tag[ids[i+1]][1]
        tag1 == 'S' && continue
        if tag1 == 'O'
            tag2 == 'I' && (count += 1)
            tag2 == 'E' && (count += 1)
        elseif tag1 == 'B'
            tag2 == 'B' && (count += 1)
            tag2 == 'O' && (count += 1)
            tag2 == 'S' && (count += 1)
        elseif tag1 == 'I'
            tag2 == 'B' && (count += 1)
            tag2 == 'O' && (count += 1)
        elseif tag1 == 'S'
            tag2 == 'I' && (count += 1)
            tag2 == 'E' && (count += 1)
        end
    end
    println("BIOES check: $count")
end

function bioes_decode_oracle(ids::Vector{Int}, tagdict::Dict{String,Int})
    id2tag = Array{String}(undef, length(tagdict))
    for (k,v) in tagdict
        id2tag[v] = k
    end

    spans = Tuple{Int,Int,String}[]
    bpos = 0
    bprev = 0
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
            if bprev > 0
                tag = id2tag[ids[bprev]]
                basetag = length(tag) > 2 ? tag[3:end] : ""
                push!(spans, (bprev,i,basetag))
            end
            bprev = bpos
            bpos = 0
        end
    end
    spans
end

function bioes_decode2(tagdict::IntDict{String}, ids::Vector{Int})
    id2tag = Array{String}(undef, length(tagdict))
    for (k,v) in tagdict.key2id
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
