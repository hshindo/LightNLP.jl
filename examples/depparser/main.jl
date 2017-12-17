using LightNLP
using LightNLP.DepParser
using JLD2, FileIO

training = true

if training
    embedsfile = ".data/word2vec.NYT.100d.h5"
    trainfile = ".data/CoNLL2009-ST-English-train.txt"
    testfile = ".data/CoNLL2009-ST-evaluation-English.txt"
    nepochs = 50
    learnrate = 0.001
    batchsize = 25
    ner = DepParser.Decoder(embedsfile, trainfile, testfile, nepochs, learnrate, batchsize)
    save("ner.jld2", "ner", ner)
else
    ner = load("ner.jld2", "ner")
    #testfile = ".data/eng.testb.BIOES"
    decode(ner, testfile)
end
