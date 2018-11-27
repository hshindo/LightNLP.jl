function nn_lstm(::Type{T}, vocab::Int) where T
    wordembeds = Normal(0,0.01)(T, 100, vocab)
    x = lookup(parameter(wordembeds), Node(name=:x))
    hsize = size(wordembeds, 1)
    h = LSTM(T,hsize,hsize,1,0.0,false)(x, Node(name=:dims_x))
    h = Linear(T,hsize,size(wordembeds,2))(h)
    Graph(h)
end

mutable struct NN_LSTM <: Functor
    wordembeds
    lstm
    linear
end

function NN_LSTM(::Type{T}, vocab::Int) where T
    wordembeds = parameter(Normal(0,0.01)(T,100,vocab))
    hsize = size(wordembeds, 1)
    lstm = LSTM(T, hsize, hsize, 1, 0.0, false)
    linear = Linear(T, hsize, size(wordembeds,2))
    NN_LSTM(wordembeds, lstm, linear)
end

function (nn::NN_LSTM)(x::NamedTuple)
    h = lookup(nn.wordembeds, x.x)
    h,hy,cy = nn.lstm(h, x.dims_x, x.training)
    h = nn.linear(h)
    if x.training
        softmax_crossentropy(x.y, h)
    else
        throw("Not implemented yet.")
    end
end
