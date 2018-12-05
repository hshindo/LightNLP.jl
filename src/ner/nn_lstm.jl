mutable struct NN_LSTM <: Functor
    wordembeds
    charembeds
    conv
    lstm
    linear
end

function NN_LSTM(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    wordembeds = parameter(wordembeds)
    charembeds = parameter(charembeds)
    csize = size(charembeds, 1)
    conv = Conv1d(T, 5, csize, 5csize, padding=2)
    hsize = size(wordembeds,1) + 5csize
    lstm = LSTM(T, hsize, hsize, 1, 0.0, true)
    linear = Linear(T, 2hsize, ntags)
    NN_LSTM(wordembeds, charembeds, conv, lstm, linear)
end

function (nn::NN_LSTM)(x::NamedTuple)
    w = lookup(nn.wordembeds, x.w)
    c = lookup(nn.charembeds, x.c)
    c = nn.conv(c, x.dims_c)
    c = max(c, x.dims_c)

    h = concat(1, w, c)
    h = dropout(h, 0.5, x.training)
    h, _, _ = nn.lstm(h, x.dims_w, x.training)
    h = dropout(h, 0.5, x.training)
    h = nn.linear(h)
    if x.training
        softmax_crossentropy(x.t, h)
    else
        y = Array{Int}(Array(x.t.data))
        z = Array{Int}(Array(argmax(h.data,1)))
        vec(y), vec(z)
    end
end

function nn_lstm(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    dims_w = Node(name=:dims_w)
    dims_c = Node(name=:dims_c)

    w = lookup(parameter(wordembeds), Node(name=:w))
    c = lookup(parameter(charembeds), Node(name=:c))
    csize = size(charembeds, 1)
    c = Conv1d(T,5,csize,5csize,padding=2)(c, dims_c)
    c = max(c, dims_c)

    h = concat(1, w, c)
    h = dropout(h, 0.5)
    hsize = size(wordembeds,1) + 5csize
    h = LSTM(T,hsize,hsize,1,0.0,true)(h, dims_w)
    h = dropout(h, 0.5)
    h = Linear(T,2hsize,ntags)(h)
    Graph(h)
end
