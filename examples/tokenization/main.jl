using LightNLP
using LightNLP.Tokenization
using JLD2, FileIO

training = true

if training
    embedsfile = ".data/glove.6B.100d.h5"
    trainfile = ".data/eng.train.BIOES"
    testfile = ".data/eng.testb.BIOES"
    nepochs = 20
    learnrate = 0.0005
    batchsize = 10
    ner = NER.Decoder(embedsfile, trainfile, testfile, nepochs, learnrate, batchsize)
    save("ner.jld2", "ner", ner)
else
    ner = load("ner.jld2", "ner")
    testfile = ".data/eng.train.BIOES"
    decode(ner, testfile)
end
