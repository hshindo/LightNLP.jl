struct NN
    g1
    g2
end

function NN(wordembeds::Vector{Var}, posembeds::Vector{Var})
    T = Float32
    w = lookup(Node(wordembeds), Node(name="w"))
    p = lookup(Node(posembeds), Node(name="p"))

    h = concat(1, w, p)
    h = dropout(h, 0.33)
    d = 300
    batchdims_w = Node(name="batchdims_w")
    h = BiLSTM(T,150,d)(h,batchdims_w)
    h = dropout(h, 0.33, istrain)
    g1 = Graph(h)

    h = Node(name="h")
    hdep = dropout(leaky_relu(Linear(T,2d,300,init_W=OrthoNormal())(h)), 0.33)
    hhead = dropout(leaky_relu(Linear(T,2d,300,init_W=OrthoNormal())(h)), 0.33)
    h = Linear(T,300,300)(hdep)
    h = BLAS.gemm('T', 'N', 1, hhead, h)
    g2 = Graph(h)
    NN(g1, g2)
end

function (nn::NN)(w::Var, p::Var, batchdims_w::Vector{Int}, istrain::Bool)
    h = nn.g1("w"=>w, "p"=>p, "train"=>istrain, "batchdims_w"=>batchdims_w)
    hs = split(h, batchdims_w)
    os = Var[]
    for h in hs
        push!(os, nn.g2("h"=>h,"train"=>istrain))
    end
    os
end
