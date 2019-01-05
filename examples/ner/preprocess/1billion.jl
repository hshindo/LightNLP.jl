using HDF5

function main(dir::String)
    data = Vector{String}[]
    for file in readdir(dir)
        path = joinpath(dir, file)
        lines = open(readlines, path)
        for line in lines
            words = split(line, " ")
            push!(data, words)
        end
        break
    end
    h5write("1billion.h5", "words", data)
end

main("D:/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled")
