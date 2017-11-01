struct NN
    g
end

function NN(wordembeds::Vector{Var}, charembeds::Vector{Var}, ntags::Int)
    T = Float32
    w = Node()
    hw = lookup(wordembeds, w)

    c = Node()
    hc = lookup(charembeds, c)
    d = size(charembeds[1], 1)
    hc = Conv1D(T,5,d,5d,2,1)(hc)
    hc = max(hc, 2)
    hc = resize(hc, batchsize(w))

    h = concat(1, hw, hc)
    d = size(wordembeds[1],1) + 5size(charembeds[1],1)
    dh = 300
    h = Conv1D(T,5,d,dh,2,1)(h)
    h = relu(h)

    istrain = Node()
    for i = 1:2
        h = dropout(h, 0.3, istrain)
        h = Conv1D(T,5,dh,dh,2,1)(h)
        h = relu(h)
    end
    h = Linear(T,dh,ntags)(h)
    g = Graph(input=(w,c,istrain), output=h)
    NN(g)
end

function (nn::NN)(w::Var, c::Var, t=nothing)
    if t == nothing
        y = nn.g(w, c, false)
        argmax(y.data, 1)
    else
        y = nn.g(w, c, true)
        softmax_crossentropy(t, y)
    end
end
