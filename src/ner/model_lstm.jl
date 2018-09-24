struct NN
    g
end

function NN(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    batchdims_w = Node(name="batchdims_w")
    batchdims_c = Node(name="batchdims_c")
    w = Node(name="w")
    c = Node(name="c")

    w = lookup(param(wordembeds), w)

    c = lookup(param(charembeds), c)
    csize = size(charembeds, 1)
    c = Conv1d(T,5,csize,5csize,padding=2)(c, batchdims_c)
    c = max(c, batchdims_c)

    h = concat(1, w, c)
    h = dropout(h, 0.5)
    hsize = size(wordembeds,1) + 5csize
    h = LSTM(T,hsize,hsize,1,0.0,true)(h, batchdims_w)
    h = dropout(h, 0.5)

    h = Linear(T,2hsize,ntags)(h)
    NN(Graph(h))
end

function NN2(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    batchdims_w = Var(name="batchdims_w")
    batchdims_c = Var(name="batchdims_c")
    w = Var(name="w")
    c = Var(name="c")

    w = lookup(param(wordembeds), w)

    c = lookup(param(charembeds), c)
    csize = size(charembeds, 1)
    c = Conv1d(T,5,csize,5csize,padding=2)(c, batchdims_c)
    c = max(c, batchdims_c)

    h = concat(1, w, c)
    h = dropout(h, 0.5)
    hsize = size(wordembeds,1) + 5csize
    h = LSTM(T,hsize,hsize,1,0.0,true)(h, batchdims_w)
    h = dropout(h, 0.5)

    h = Linear(T,2hsize,ntags)(h)
    NN(Graph(h))
end
