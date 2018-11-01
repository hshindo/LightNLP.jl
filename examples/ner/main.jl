using LightNLP
using LightNLP.NERs
using JSON
using JLD2, FileIO

config = JSON.parsefile(ARGS[1])
if config["training"]
    ner = NER(config)
    #save("ner.jld2", "ner", ner)
else
    ner = load("ner.jld2", "ner")
    decode(ner, testfile)
end
