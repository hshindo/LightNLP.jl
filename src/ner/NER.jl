module NER

using Merlin
using HDF5
using ..LightNLP

include("dataset.jl")
include("model.jl")
include("nn_lstm.jl")
include("slstm.jl")
include("nn_slstm.jl")
include("gatedunit.jl")

end
