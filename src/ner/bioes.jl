struct BIOES
    tag2id::Dict{String,Int}
    id2tag::Vector{String}
end

BIOES() = BIOES(Dict{String,Int}(), String[])

function encode(tagset::BIOES, tags::Vector{String})
    map(tags) do tag
        get!(tagset.tag2id, tag) do
            push!(tagset.id2tag, tag)
            length(tagset.tag2id) + 1
        end
    end
end

function decode(tagset::BIOES, ids::Vector{Int})
    id2tag = tagset.id2tag
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
