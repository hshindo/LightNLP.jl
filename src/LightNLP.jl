module LightNLP

include("intdict.jl")
#include("depparser/DepParser.jl")
include("ner/NERs.jl")
#include("tokenization/Tokenization.jl")

export namedtuple

function namedtuple(config::Dict)
    k = map(Symbol, collect(keys(config)))
    NamedTuple{tuple(k...)}(tuple(values(config)...))
end

end
