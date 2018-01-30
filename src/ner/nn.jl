struct NN
    g
end

function NN(embeds_w::Matrix{T}, embeds_c::Matrix{T}, ntags::Int) where T
    batchdims_w = Node(name="batchdims_w")
    batchdims_c = Node(name="batchdims_c")
    w = lookup(zerograd(embeds_w), Node(name="w"))
    c = lookup(zerograd(embeds_c), Node(name="c"))

    d = size(embeds_c, 1)
    c = Conv(T,(d,5,1,5d),pads=2)(c)
    c = transpose(c)
    c = reshape(c, size(c,1), size(c,3))
    c = max_batch(c, batchdims_c)

    h = cat(1, w, c)


    c = Conv1D(T,5,d,5d,2,1)(c,batchdims_c)

    w = lookup(wordembeds, Node("w"))
    batchdims_c = Node("batchdims_c")
    c = lookup(charembeds, Node("c"))
    d = size(charembeds[1], 1)
    c = Conv1D(T,5,d,5d,2,1)(c,batchdims_c)
    c = max_batch(c, batchdims_c)

    h = cat(1, w, c)
    batchdims_w = Node("batchdims_w")
    d = 100 + 5size(charembeds[1],1)
    dh = 300
    h = Conv1D(T,5,d,dh,2,1)(h,batchdims_w)
    h = leaky_relu(h)

    istrain = Node("train")
    for i = 1:2
        h = dropout(h, 0.3, istrain)
        h = Conv1D(T,5,dh,dh,2,1)(h,batchdims_w)
        h = leaky_relu(h)
    end
    h = Linear(T,dh,ntags)(h)
    g = Graph(h)
    NN(g)
end


struct NN
    g
end

function NN(wordembeds::Vector{Var}, charembeds::Vector{Var}, ntags::Int)
    T = Float32
    w = lookup(wordembeds, Node("w"))
    batchdims_c = Node("batchdims_c")
    c = lookup(charembeds, Node("c"))
    d = size(charembeds[1], 1)
    c = Conv1D(T,5,d,5d,2,1)(c,batchdims_c)
    c = max_batch(c, batchdims_c)

    h = cat(1, w, c)
    batchdims_w = Node("batchdims_w")
    d = 100 + 5size(charembeds[1],1)
    dh = 300
    h = Conv1D(T,5,d,dh,2,1)(h,batchdims_w)
    h = leaky_relu(h)

    istrain = Node("train")
    for i = 1:2
        h = dropout(h, 0.3, istrain)
        h = Conv1D(T,5,dh,dh,2,1)(h,batchdims_w)
        h = leaky_relu(h)
    end
    h = Linear(T,dh,ntags)(h)
    g = Graph(h)
    NN(g)
end
