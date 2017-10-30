using Merlin

include("bioes.jl")
include("ner.jl")
include("model.jl")

const wordembeds_file = ".data/glove.6B.100d.h5"

function main()
    argdict = Dict()
    for arg in ARGS
        kv = Vector{String}(split(arg,"="))
        length(kv) == 2 || throw("Invalid arguments: $arg")
        argdict[kv[1]] = kv[2]
    end

    if parse(Bool, argdict["train"])
        nepochs = parse(Int, argdict["nepochs"])
        #savefile = argdict["savefile"]
        ner = NER()
        traindata = readdata!(ner, ".data/eng.train.BIOES")
        testdata = readdata!(ner, ".data/eng.testb.BIOES")
        train(ner, traindata, testdata)
    else
        #seg = load("NER.jld2")
        #println(seg["a"].char2id)
        #seg = Merlin.load("NER.merlin")
    end
end
main()
