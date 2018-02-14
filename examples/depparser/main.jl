using LightNLP
using LightNLP.DepParser
using JLD2, FileIO

include(ARGS[1])

if TRAINING
    parser = DepParser.Decoder(embedsfile, trainfile, testfile, nepochs, learnrate, batchsize)
    save("parser.jld2", "parser", parser)
else
    parser = load("parser.jld2", "parser")
    decode(parser, testfile)
end
