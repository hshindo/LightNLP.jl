struct NN
    g
end

function NN()
    w = Node()
    c = Node()

    h = lookup(weights, 3, 7)
    b = Linear(T,10,10)(h[1])
    e = Linear(T,10,10(h[end]))
    i = Linear(T,10,10)(h[2:end-1])
    
end

function NN(wordembeds1::Var, wordembeds2::Vector{Var}, charembeds::Vector{Var}, ntags::Int)
    T = Float32
    w1 = Node()
    hw1 = lookup(wordembeds1, w1)
    w2 = Node()
    hw2 = lookup(wordembeds2, w2)

    c = Node()
    hc = lookup(charembeds, c)
    d = size(charembeds[1], 1)
    hc = Conv1D(T,5,d,5d,2,1)(hc)
    hc = max(hc, 2)
    hc = resize(hc, batchsize(w1))

    h = concat(1, hw1, hw2, hc)
    d = 200 + 5size(charembeds[1],1)
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
    g = Graph(input=(w1,w2,c,istrain), output=h)
    NN(g)
end

function (nn::NN)(w1::Var, w2::Var, c::Var, t=nothing)
    if t == nothing
        y = nn.g(w1, w2, c, false)
        argmax(y.data, 1)
    else
        y = nn.g(w1, w2, c, true)
        softmax_crossentropy(t, y)
    end
end
