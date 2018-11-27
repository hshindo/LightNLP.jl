function nn_cnn(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    dims_w = Node(name=:dims_w)
    dims_c = Node(name=:dims_c)

    w = lookup(parameter(wordembeds), Node(name=:w))
    c = lookup(parameter(charembeds), Node(name=:c))
    csize = size(charembeds, 1)
    #c = dropout(c, 0.5)
    c = Conv1d(T,5,csize,5csize,padding=2)(c, dims_c)
    c = max(c, dims_c)

    h = concat(1, w, c)
    #h = dropout(h, 0.5)
    hsize = size(wordembeds,1) + 5csize
    h = Conv1d(T,5,hsize,2hsize,padding=2)(h, dims_w)
    #h = dropout(h, 0.5)
    h = Linear(T,2hsize,ntags)(h)
    Graph(h)
end
