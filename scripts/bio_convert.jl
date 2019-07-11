function bio2bioes(filename::String)
    lines = open(readlines, filename)
    isempty(lines[end]) || push!(lines,"")
    lines = map(lines) do line
        isempty(line) && return line
        items = split(line, "\t")
        (items[1], items[4])
    end
    data = String[]
    for i = 1:length(lines)-1
        line = lines[i]
        if isempty(line)
            push!(data, line)
        else
            token1, tag1 = lines[i]
            token2, tag2 = isempty(lines[i+1]) ? ("","O") : lines[i+1]
            if tag1 == "B" && (tag2 == "O" || tag2 == "B")
                tag1 = "S"
            elseif tag1 == "I" && (tag2 == "O" || tag2 == "B")
                tag1 = "E"
            end
            push!(data, "$token1\t$tag1")
        end
    end
    open("$filename.BIOES","w") do io
        foreach(s -> println(io,s), data)
    end
end
