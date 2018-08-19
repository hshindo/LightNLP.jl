struct NN
    g
end

function NN(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
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
    NN(Graph(h))
end
