using LightNLP
using LightNLP.NER
using JSON

config = JSON.parsefile(ARGS[1])

if config["training"]
    ner = NER.Model(config)
    #LightNLP.save("ner.jld2", ner)
else
    ner = LightNLP.load("ner.jld2")
    testfile = ".data/eng.testb.BIOES"
    ner(testfile)
end
println("Finish.")
