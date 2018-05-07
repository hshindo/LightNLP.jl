using LightNLP
using LightNLP.NER
using JLD2, FileIO

include("config.jl")

if CONFIG["training"]
    ner = NER.Decoder()
    #save("ner.jld2", "ner", ner)
else
    ner = load("ner.jld2", "ner")
    decode(ner, testfile)
end
