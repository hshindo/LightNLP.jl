const threshold = 5

function normalize_text(path::String)
    println("Tokenizing...")
    text = readstring(`java -cp stanford-ner.jar edu.stanford.nlp.process.PTBTokenizer $path -preserveLines`)

    println("Normalizing...")
    words = String[]
    countdict = Dict{String,Int}()
    for w in split(text," ")
        w = normalize_string(w, :NFKC)
        w = lowercase(w)
        w = replace(w, r"[0-9]", '0') # ex. 123.45 -> 000.00
        push!(words, w)
        if haskey(countdict, w)
            countdict[w] += 1
        else
            countdict[w] = 1
        end
    end
    println("Original vocabulary size: $(length(countdict))")

    unkdict = Dict{String,String}()
    for (k,v) in countdict
        v < threshold && (unkdict[k] = k)
    end
    c = length(countdict) - length(unkdict)
    println("Normalized (>= $threshold) vocabulary size: $c")

    words = map(words) do w
        haskey(unkdict,w) ? "UNKNOWN" : w
    end
    str = join(words, " ")
    open("$path.clean","w") do io
        println(io, str)
    end
end

normalize_text(ARGS[1])
