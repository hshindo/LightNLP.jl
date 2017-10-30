struct Decoder
    worddict::Dict
    chardict::Dict
    tagset
    nn
end

function Decoder(words::Vector{String}, wordembeds)
    worddict = Dict(words[i] => i for i=1:length(words))
    chardict = Dict("UNKNOWN" => 1)
    for word in words
        for c in Vector{Char}(word)
            get!(chardict, string(c), length(chardict)+1)
            get!(chardict, string(uppercase(c)), length(chardict)+1)
        end
    end
    Decoder(worddict, chardict, BIOES(), nothing)
end

function decode(dec::Decoder, data::Vector{String})
    for (w,c) in data
        y = ner.model(w, c)
        y = argmax(y.data, 1)
    end
end

function encode_word(dec::Decoder, words::Vector{String})
    worddict = ner.worddict
    unkword = worddict["UNKNOWN"]
    ids = map(words) do w
        w = lowercase(w)
        # w = replace(word, r"[0-9]", '0')
        get(worddict, w, unkword)
    end
    Var(ids)
end

function encode_char(dec::Decoder, words::Vector{String})
    chardict = ner.chardict
    unkchar = chardict["UNKNOWN"]
    batchdims = Int[]
    ids = Int[]
    for w in words
        # w = replace(word, r"[0-9]", '0')
        chars = Vector{Char}(w)
        push!(batchdims, length(chars))
        for c in chars
            push!(ids, get(chardict,string(c),unkchar))
        end
    end
    Var(ids, batchdims)
end

function readdata!(dec::Decoder, path::String)
    data = NTuple{3,Var}[]
    words, tags = String[], String[]
    lines = open(readlines, path)
    for i = 1:length(lines)
        line = lines[i]
        if isempty(line) || i == length(lines)
            isempty(words) && continue
            w = encode_word(ner, words)
            c = encode_char(ner, words)
            t = Var(encode(ner.tagset,tags))
            push!(data, (w,c,t))
            empty!(words)
            empty!(tags)
        else
            items = split(line, "\t")
            push!(words, String(items[1]))
            push!(tags, String(items[2]))
        end
    end
    data
end
