module NER

using Merlin
using HDF5
using ..LightNLP

include("dataset3.jl")
include("model.jl")
include("bioes.jl")
include("nn_rcnn.jl")
# include("nn_flair.jl")

end
