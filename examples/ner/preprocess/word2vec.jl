using HDF5

function word2vec(path::String)
    run(`python word2vec.py $path`)
    lines = open(readlines, "word2vec.model")
    rm("word2vec.model")
    w = String[]
    v = Float32[]
    for i = 2:length(lines)
        line = lines[i]
        items = split(line, " ")
        push!(w, items[1])
        for k = 2:length(items)
            push!(v, parse(Float32,items[k]))
        end
    end
    v = reshape(v, length(v)÷length(w), length(w))
    h5write("$(path).100d.h5", "words", w)
    h5write("$(path).100d.h5", "vectors", v)
end

word2vec(ARGS[1])
