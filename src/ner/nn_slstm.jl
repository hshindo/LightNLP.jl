struct SLSTM
    nlayers::Int
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
    h1 = ilrfso[1:n]
    h2 = ilrfso[n+1:2n]
    h3 = ilrfso[2n+1:3n]
    h4 = ilrfso[3n+1:4n]
    h5 = ilrfso[4n+1:5n]
    h6 = ilrfso[5n+1:6n]
    h2 .* cminus + h4 .*
    h1 .* u
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
    #h = swish(h,zerograd([T(1)]))

    istrain = Node()
    for i = 1:2
        h = dropout(h, 0.3, istrain)
        h = Conv1D(T,5,dh,dh,2,1)(h)
        h = relu(h)
        #h = swish(h,zerograd([T(1)]))
    end
    h = Linear(T,dh,ntags)(h)
    g = Graph(input=(w1,w2,c,istrain), output=h)
    NN(g)
end
(f::SLSTM)(x::Node, dims) = Node(f, (x,dims))

function slstm_tstep(x::Var, dims)
    h = f.h
    c = f.c
    cg = f.cg
    g = h
    for l = 1:nlayers
        h = conv1d_index(h, dims, window=1)
        ci = conv1d_index(c, dims, window=1)
        cg = repeat(cg)
        h = concat(1, h, x, g)
        h = linear(h, W, b)
        n = length(h) ÷ 7
    end
end
