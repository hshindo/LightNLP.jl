struct NN
    g
end

function NN(wordembeds::Vector{Var}, posembeds::Vector{Var})
    T = Float32
    w = lookup(wordembeds, Node("w"))
    p = lookup(posembeds, Node("p"))

    h = cat(1, w1, w2, c)
    batchdims_w = Node("batchdims_w")
    d = 200 + 5size(charembeds[1],1)
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
