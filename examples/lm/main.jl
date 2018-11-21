using LightNLP
using LightNLP.LM
using JSON

config = JSON.parsefile(ARGS[1])

if config["training"]
    m = LM.Model(config)
    #save(model)
    #ner = LM.Decoder(ner)
    #LightNLP.save("ner.jld2", ner)
else
    #ner = LightNLP.load("ner.jld2")
    #testfile = ".data/eng.testb.BIOES"
    #ner(testfile)
end
println("Finish.")
