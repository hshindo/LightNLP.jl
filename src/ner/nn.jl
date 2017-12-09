struct NN
    g
end

function NN(wordembeds1::Var, wordembeds2::Vector{Var}, charembeds::Vector{Var}, ntags::Int)
    T = Float32
    w1 = lookup(wordembeds1, Node("w1"))
    w2 = lookup(wordembeds2, Node("w2"))

    batchdims_c = Node("batchdims_c")
    c = lookup(charembeds, Node("c"))
    d = size(charembeds[1], 1)
    c = Conv1D(T,5,d,5d,2,1)(c,batchdims_c)
    c = max_batch(c, batchdims_c)

    h = cat(1, w1, w2, c)
    batchdims_w = Node("batchdims_w")
    d = 200 + 5size(charembeds[1],1)
    dh = 300
    h = Conv1D(T,5,d,dh,2,1)(h,batchdims_w)
    h = relu(h)

    #istrain = Node("train")
    #for i = 1:3
    #    h = dropout(h, 0.3, istrain)
    #    h = Conv1D(T,5,dh,dh,2,1)(h,batchdims_w)
    #    h = relu(h)
    #end
    h = Linear(T,dh,ntags)(h)
    g = Graph(h)
    NN(g)
end
