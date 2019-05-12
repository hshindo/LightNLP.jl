module NER

using Merlin
using HDF5
using ..LightNLP

include("data.jl")
include("model.jl")
include("bioes.jl")
include("nn_rcnn.jl")
include("nn_rann.jl")
# include("nn_flair.jl")

end
