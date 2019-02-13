function bioes_decode(ids::Vector{Int}, tagdict::Dict{String,Int})
    id2tag = Array{String}(undef, length(tagdict))
    for (k,v) in tagdict
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

function bioes_3gram(tagdict::Dict{String,Int})
    id2tag = Array{String}(undef, length(tagdict))
    for (k,v) in tagdict
        id2tag[v] = k
    end
    "O", "O"
    "O", "B"
    "O", "S"
end
