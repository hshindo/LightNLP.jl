struct NN
    g
end

function NN(wordembeds::Vector{Var}, charembeds::Vector{Var}, ntags::Int)
    T = Float32
    istrain = Var(name="train")
    w = lookup(wordembeds, Var(name="w"))

    batchdims_c = Var(name="batchdims_c")
    c = lookup(charembeds, Var(name="c"))
    c = dropout(c, 0.5, istrain)
    d = size(charembeds[1].data, 1)
    c = Conv1D(T,5,d,5d,2,1)(c,batchdims_c)
    c = max_batch(c, batchdims_c)

    h = concat(1, w, c)
    h = dropout(h, 0.5, istrain)
    batchdims_w = Var(name="batchdims_w")
    d = 100 + 5size(charembeds[1].data,1)
    h = BiLSTM(T,d,d)(h,batchdims_w)
    h = dropout(h, 0.5, istrain)

    h = Linear(T,2d,ntags)(h)
    g = Graph(h)
    NN(g)
end
