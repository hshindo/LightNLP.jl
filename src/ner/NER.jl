module NER

using Merlin
using HDF5
using ..LightNLP

include("dataset.jl")
include("model.jl")
include("nn_rcnn.jl")

end
