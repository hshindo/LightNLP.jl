function model_dep(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
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

    hdep = Linear(T,hsize,hsize)(h)
    hhead = Linear(T,hsize,hsize)(h)
    Linear(T,hsize,hsize)(hdep) * transpose(hhead)

    h = Linear(T,2hsize,ntags)(h)
    Graph(h)
end
