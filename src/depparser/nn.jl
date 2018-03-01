struct NN
    g1
    g2
end

function NN(wordembeds::Matrix{T}, posembeds::Matrix{T}) where T
    w = lookup(Node(zerograd(wordembeds)), Node(name="w"))
    p = lookup(Node(zerograd(posembeds)), Node(name="p"))

    h = concat(1, w, p)
    #h = dropout(h, 0.33)
    d = 300
    batchdims_w = Node(name="batchdims_w")
    h = LSTM(T,150,d,1,0.0,true)(h,batchdims_w)
    # h = dropout(h, 0.33)
    g1 = BACKEND(Graph(h))

    h = Node(name="h")
    hdep = Linear(T, 2d, 300, init_w=OrthoNormal())(h)
    hdep = relu(hdep)
    #hdep = dropout(hdep, 0.33)
    hhead = Linear(T, 2d, 300, init_w=OrthoNormal())(h)
    hhead = relu(hhead)
    #hhead = dropout(hhead, 0.33)

    h = Linear(T,300,300)(hdep)
    h = BLAS.gemm('T', 'N', 1, hhead, h)
    g2 = BACKEND(Graph(h))
    NN(g1, g2)
end

function (nn::NN)(x::Sample, istrain::Bool)
    Merlin.CONFIG.train = istrain
    h = nn.g1(x.batchdims_w, x.p, x.w)
    hs = split(h, x.batchdims_w)
    os = Var[]
    for h in hs
        push!(os, nn.g2(h))
    end
    if istrain
        ys = Var[]
        for (h,o) in zip(x.heads,os)
            y = softmax_crossentropy(h, o)
            push!(ys, y)
        end
        concat(1, ys)
    else
        cat(1, map(o -> vec(argmax(Array(o.data),1)), os)...)
    end
end
