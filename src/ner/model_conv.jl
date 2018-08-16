struct NN
    g
end

function NN(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    # batchsize_w = Node(name="batsize_w")
    w = lookup(param(wordembeds), Node(name="w"))

    csize = size(charembeds, 1)
    # batchsize_c = Node("batchsize_c")
    c = lookup(param(charembeds), Node(name="c"))
    c = dropout(c, 0.5)
    c = Conv1d(T,5,csize,5csize,pad=2)(c)
    c = max(c)

    h = concat(1, w, c)
    h = dropout(h, 0.5)
    hsize = size(wordembeds,1) + 5csize
    h = LSTM(T,hsize,hsize,1,0.0,true)(h)
    h = dropout(h, 0.5)
    h = Linear(T,2hsize,ntags)(h)
    NN(Graph(h))
end
