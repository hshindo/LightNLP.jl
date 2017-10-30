using LightNLP
using LightNLP.NER
using HDF5, JLD2, FileIO

embedfile = ".data/glove.6B.100d.h5"
words = h5read(embedfile, "key")
wordembeds = h5read(embedfile, "value")
ner = NER.Decoder(words, wordembeds)
train(ner, ".data/eng.train.BIOES", ".data/eng.testb.BIOES")
save("ner.jld2", "ner", ner)
