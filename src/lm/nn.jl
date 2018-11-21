function nn_lstm(::Type{T}, vocab::Int) where T
    wordembeds = Normal(0,0.01)(T, 100, vocab)
    x = lookup(parameter(wordembeds), Node(name=:x))
    hsize = size(wordembeds, 1)
    h = LSTM(T,hsize,hsize,1,0.0,false)(x, Node(name=:dims_x))
    h = Linear(T,hsize,size(wordembeds,2))(h)
    Graph(h)
end

mutable struct LM_LSTM <: Parametric
    params
end

Merlin.parameters(f::LM_LSTM) = parameters(f.params...)
Merlin.todevice(f::LM_LSTM, device) = LM_LSTM(todevice(f.params,device))

function LM_LSTM(::Type{T}, vocab::Int) where T
    wordembeds = Normal(0,0.01)(T, 100, vocab)
    hsize = size(wordembeds, 1)
    lstm = LSTM(T, hsize, hsize, 1, 0. 0, false)
    linear = Linear(T, hsize, size(wordembeds,2))
    params = (wordembeds=wordembeds, lstm=lstm, linear=linear)
    LM_LSTM(params)
end

function (nn::LM_LSTM)(nt::NamedTuple)
    p = nn.params
    h = lookup(p.wordembeds, nt.x)
    h = p.lstm(h, nt.dims_x)
    z = p.linear(h)
    if t.train
        l = softmax_crossentropy(nt.y, z)
        gradient!(l)
        sum(Array(l.data))
    else
        throw("Not implemented yet.")
    end
end
