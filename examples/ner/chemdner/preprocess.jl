function readdata(path::String)
    lines = open(readlines, "$path.annotations.txt")
    dict = Dict()
    for line in lines
        items = Vector{String}(split(line,"\t"))
        id = items[1]
        anno = (TA=items[2], s=parse(Int,items[3])+1, e=parse(Int,items[4]), label=items[6])
        if anno.s > anno.e
            #println(anno.s)
            #println(anno.e)
            @warn "Annotation index is wrong: $(id)"
            continue
        end
        get!(dict, id, [])
        push!(dict[id], anno)
    end

    conll = String[]
    lines = open(readlines, "$path.abstracts.txt")
    for line in lines
        items = Vector{String}(split(line,"\t"))
        id = items[1]
        title = items[2]
        abst = items[3]
        haskey(dict,id) || continue
        annos = dict[id]
        for TA in ("T","A")
            str = TA == "T" ? title : abst
            char2word, words = indexing(str)
            tags = map(_ -> "_", words)
            for anno in annos
                anno.TA == TA || continue
                s = char2word[anno.s]
                e = char2word[anno.e]
                if s == e
                    tags[s] = "S-$(anno.label)"
                else
                    tags[s] = "B-$(anno.label)"
                    for k = s+1:e-1
                        tags[k] = "I-$(anno.label)"
                    end
                    tags[e] = "E-$(anno.label)"
                end
            end
            for (w,t) in zip(words,tags)
                if w == "."
                    push!(conll, "")
                    continue
                elseif endswith(w, ".")
                    cs = Vector{Char}(w)
                    w = join(cs[1:end-1])
                    push!(conll, "$w\t$t")
                    push!(conll, "")
                else
                    push!(conll, "$w\t$t")
                end
            end
        end
    end
    open("out.BIOES","w") do io
        for line in conll
            println(io, line)
        end
    end
end

function indexing(str::String)
    chars = Vector{Char}(str)
    inds = findall(c -> c == ' ', chars)
    push!(inds, length(chars)+1)
    char2word = Dict()
    wordid = 1
    words = String[]
    for i = 1:length(inds)
        s = i == 1 ? 1 : inds[i-1]+1
        e = inds[i] - 1
        for k = s:e
            char2word[k] = wordid
        end
        push!(words, join(chars[s:e]))
        wordid += 1
    end
    char2word, words
end

readdata(".data/evaluation")
