struct NN
    g
end

Merlin.graphs(nn::NN) = (nn.g,)

function model_lstm(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
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

function model_cnn(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    batchdims_w = Node(name="batchdims_w")
    batchdims_c = Node(name="batchdims_c")
    w = Node(name="w")
    c = Node(name="c")

    w = lookup(param(wordembeds), w)

    csize = size(charembeds, 1)
    c = lookup(param(charembeds), c)
    c = dropout(c, 0.5)
    c = Conv1d(T,5,csize,5csize,padding=2)(c, batchdims_c)
    c = max(c, batchdims_c)

    h = concat(1, w, c)
    h = dropout(h, 0.5)
    hsize = size(wordembeds,1) + 5csize
    h = Conv1d(T,5,hsize,2hsize,padding=2)(h, batchdims_w)
    h = dropout(h, 0.5)
    h = Linear(T,2hsize,ntags)(h)
    Graph(h)
end

struct SLSTM
    W::Var
    b::Var
    h::Var
    c::Var
end

function SLSTM()
    W = parameter()
    b = parameter()
    h = parameter()
end

function (f::SLSTM)(x::Var, dims)
    g = f.h
    hxg = concat(1, f.h, x, g)
    h = linear(hxg, f.W, f.b)
    n = length(h) ÷ 7
    ilrfso = sigmoid(h[1:6n])
    u = tanh(h[6n+1:7n])
    #ilrfso[1:n] .*
end
(f::SLSTM)(x::Node, dims) = Node(f, (x,dims))

function model_slstm(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
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
    h = Conv1d(T,3,hsize,hsize,padding=1)(h, dims_w)

    W = parameter()
    b = parameter()
    h = parameter()

end
