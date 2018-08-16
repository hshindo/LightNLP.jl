struct NN
    g
end

function NN(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    batchdims_w = Node(name="batchdims_w")
    batchdims_c = Node(name="batchdims_c")

    w = lookup(param(wordembeds), Node(name="w"))

    c = lookup(param(charembeds), Node(name="c"))
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
