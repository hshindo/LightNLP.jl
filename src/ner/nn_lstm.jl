function setup_nn(embeds_w::Matrix{T}, embeds_c::Matrix{T}, ntags::Int) where T
    embeds_w = zerograd(embeds_w)
    w = lookup(Node(embeds_w), Node(name="w"))

    embeds_c = zerograd(embeds_c)
    c = lookup(Node(embeds_c), Node(name="c"))
    c = dropout(c, 0.5)
    batchdims_c = Node(name="batchdims_c")
    c = window1d(c, 2, batchdims_c)
    csize = size(embeds_c, 1)
    c = Linear(T,5csize,5csize)(c)
    c = maximum(c, 2, batchdims_c)

    h = concat(1, w, c)
    h = dropout(h, 0.5)
    batchdims_w = Node(name="batchdims_w")
    hsize = size(embeds_w,1) + 5csize
    h = LSTM(T,hsize,hsize,1,0.0,true)(h,batchdims_w)
    h = dropout(h, 0.5)

    h = Linear(T,2hsize,ntags)(h)
    Graph(h)
end

#=
struct NN
    g
end

function NN(wordembeds::Vector{Var}, charembeds::Vector{Var}, ntags::Int)
    T = Float32
    istrain = Node(name="train")
    w = lookup(Node(wordembeds), Node(name="w"))

    batchdims_c = Node(name="batchdims_c")
    c = lookup(Node(charembeds), Node(name="c"))
    c = dropout(c, 0.5, istrain)
    d = size(charembeds[1], 1)
    c = Conv1D(T,5,d,5d,2,1)(c,batchdims_c)
    c = max_batch(c, batchdims_c)

    h = concat(1, w, c)
    h = dropout(h, 0.5, istrain)
    batchdims_w = Node(name="batchdims_w")
    d = 100 + 5size(charembeds[1],1)
    h = BiLSTM(T,d,d)(h,batchdims_w)
    h = dropout(h, 0.5, istrain)

    h = Linear(T,2d,ntags)(h)
    g = Graph(h)
    NN(g)
end
=#
