function convert_BIOES(path::String)
    lines = open(readlines, path)
    data = []
    for i = 1:length(lines)
        line = lines[i]
        if isempty(line)
            push!(data, ())
        else
            items = split(line, "\t")
            word = items[1]
            tag = items[3]
            push!(data, (word,tag))
        end
    end
    lines = String[]
    prev_data = nothing
    for i = 1:length(data)
        if isempty(data[i])
            push!(lines, "")
        else
            word, tag = data[i]
            if tag == "O"
                push!(lines, "$(word)\tO")
            else
                @assert tag == "POLY"
                if !isempty(prev_data) && prev_data[2] == "POLY"
                    if i == length(data) || isempty(data[i+1]) || data[i+1][2] == "O"
                        tag = "E-POLY"
                    else
                        tag = "I-POLY"
                    end
                else
                    if i == length(data) || isempty(data[i+1]) || data[i+1][2] == "O"
                        tag = "S-POLY"
                    else
                        tag = "B-POLY"
                    end
                end
                push!(lines, "$(word)\t$(tag)")
            end
        end
        prev_data = data[i]
    end
    open("a.out", "w") do io
        for line in lines
            println(io, line)
        end
    end
end

convert_BIOES("poly_test.conll")
