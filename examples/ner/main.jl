using LightNLP
using LightNLP.NER
using JLD2, FileIO

include(ARGS[1])

if training
    ner = NER.Decoder(embedsfile, trainfile, testfile, nepochs, learnrate, batchsize)
    save("ner.jld2", "ner", ner)
else
    ner = load("ner.jld2", "ner")
    decode(ner, testfile)
end
