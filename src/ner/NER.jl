module NER

using Merlin
using HDF5
using ..LightNLP

include("dataset2.jl")
include("model.jl")
include("bioes.jl")
include("nn_rcnn.jl")

end
