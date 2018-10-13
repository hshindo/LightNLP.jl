using LightNLP
using LightNLP.NERs
using JLD2, FileIO
using Profile

include("config.jl")

if config.training
    ner = NER(config)
    #Profile.print()
    # save("ner.jld2", "ner", ner)
else
    ner = load("ner.jld2", "ner")
    decode(ner, testfile)
end
