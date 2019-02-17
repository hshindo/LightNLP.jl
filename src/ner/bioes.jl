function bioes_decode(tagdict::IntDict{String}, tagids::Vector{Int})
    nospan = tagdict["NO_SPAN"]
    spans = Tuple{Int,Int,String}[]
    offs = 0
    for k = 1:10
        n = length(tags)- k + 1
        ids = tagids[offs+1:offs+n]
        for i = 1:n
            ids[i] == nospan && continue
            tag = tagdict[ids[i]]
            push!(spans, (i,i+k-1,tag))
        end
        offs += n
    end
    spans
end

function bioes_encode!(tagdict::IntDict{String}, tags::Vector{String})
    spantags = [[tagdict["NO_SPAN"] for _=1:length(tags)-k+1] for k=1:10]
    bpos = 0
    for i = 1:length(tags)
        tag = tags[i]
        tag == "O" && continue
        catid = get!(tagdict, tag[3:end])
        if startswith(tag,"B")
            bpos = i
        elseif startswith(tag,"S")
            spantags[1][i] = catid
        elseif startswith(tag,"E")
            spantags[i-bpos+1][bpos] = catid
            bpos = 0
        end
    end
    cat(Iterators.flatten(spantags)..., dims=1)
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

function bioes_decode2(tagdict::IntDict{String}, ids::Vector{Int})
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
