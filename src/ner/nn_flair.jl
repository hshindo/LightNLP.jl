mutable struct NN_LSTM <: Functor
    wordembeds
    flair_train
    flair_test
    lstm
    linear
end

function NN_LSTM(wordembeds::Matrix{T}, flair_train, flair_test, ntags::Int) where T
    wordembeds = Var(wordembeds)
    wsize = size(wordembeds, 1) + size(flair_train, 1)
    #l1 = Linear(T, wsize, wsize)
    hsize = 256
    lstm = LSTM(T, wsize, hsize, 1, 0.0, true)
    linear = Linear(T, 2hsize, ntags)
    NN_LSTM(wordembeds, flair_train, flair_test, lstm, linear)
end

function (nn::NN_LSTM)(x::NamedTuple)
    w = lookup(nn.wordembeds, x.w)
    if Merlin.istraining()
        fw = lookup(nn.flair_train, x.count)
    else
        fw = lookup(nn.flair_test, x.count)
    end
    w = concat(1, w, fw)
    h = w
    h = dropout_dim(h, 1, 0.5)
    h = dropout_dim(h, 2, 0.05)
    # h = nn.l1(h)
    h, _, _ = nn.lstm(h, x.dims_w)
    h = dropout_dim(h, 1, 0.5)
    h = nn.linear(h)
    if Merlin.istraining()
        softmax_crossentropy(x.t, h, x.dims_w)
    else
        y = Array{Int}(Array(x.t.data))
        z = Array{Int}(Array(argmax(h.data,1)))
        vec(y), vec(z)
    end
end
