function main()
    filename = "news.en-00001-of-00100"
    dict = Dict{String,Int}()
    lines = open(readlines, filename)
    for line in lines
        words = split(line, "\t")
        for w in words
            
        end
    end
end
