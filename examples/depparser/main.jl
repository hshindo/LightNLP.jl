using LightNLP
using LightNLP.DepParser
# using JLD2, FileIO

include("config.jl")

if CONFIG["training"]
    parser = DepParser.Decoder(CONFIG)
    # save("parser.jld2", "parser", parser)
else
    parser = load("parser.jld2", "parser")
    decode(parser, testfile)
end
