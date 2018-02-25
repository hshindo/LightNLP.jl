struct NN
    g
end

function NN(embeds_w::Matrix{T}, embeds_c::Matrix{T}, ntags::Int) where T
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
    NN(Graph(h))
end

function (nn::NN)(x::Sample, train::Bool)
    Merlin.CONFIG.train = train
    z = nn.g(x.batchdims_c, x.batchdims_w, Var(x.c), Var(x.w))
    if train
        softmax_crossentropy(Var(x.t), z)
    else
        argmax(z.data, 1)
    end
end
