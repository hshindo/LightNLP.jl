using HDF5

"""
    Convert glove txt file into HDF5.
"""
function glove2h5(filename::String)
    lines = open(readlines, filename)
    words = String[]
    vectors = Float32[]
    for i = 1:length(lines)
        line = lines[i]
        items = split(line, ' ')
        word = items[1]
        vector = map(x -> parse(Float32,x), items[2:end])
        push!(words, word)
        append!(vectors, vector)
        i % 1000 == 0 && println(i)
    end
    n = length(lines)
    vectors = reshape(vectors, length(vectors)÷n, n)
    h5write("glove.42B.300d.h5", "words", words)
    h5write("glove.42B.300d.h5", "vectors", vectors)
end
glove2h5("glove.42B.300d.txt")
