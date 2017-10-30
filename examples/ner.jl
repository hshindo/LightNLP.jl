using LightNLP
using LightNLP.NER
using HDF5

const NEPOCHS = 10
const WORD_EMBEDS = "data/glove.6B.100d.h5"

words = h5read(WORD_EMBEDS, "key")
wordembeds = embeddings(h5read(WORD_EMBEDS,"value"))
charembeds = embeddings(Float32, length(ner.chardict), 20, init_w=Normal(0,0.05))
model = NER.setup_model(wordembeds, charembeds, length(ner.tagset.tag2id))

if training
    ner = NER.Decoder()
    train(ner, "data/eng.train.BIOES", "data/eng.testb.BIOES")
else
    ner = NER.Decoder(model)
    decode(ner, "data/...")
end
