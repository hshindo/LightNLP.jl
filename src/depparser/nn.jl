struct NN
    g
end

function NN(wordembeds::Vector{Var}, posembeds::Vector{Var})
    T = Float32
    istrain = Node("train")
    w = lookup(wordembeds, Node("w"))
    p = lookup(posembeds, Node("p"))

    h = cat(1, w, p)
    h = dropout(h, 0.33, istrain)
    #d = 100 + 5size(posembeds[1],1)
    d = 150
    batchdims_w = Node("batchdims_w")
    h = BiLSTM(T,d,d)(h,batchdims_w)
    h = dropout(h, 0.33, istrain)

    hdep = Linear(T,2d,300)(h)
    hhead = Linear(T,2d,300)(h)
    h = Linear(T,300,300)(hdep)
    h = BLAS.gemm('T', 'N', 1, hhead, h)
    NN(Graph(h))
end
