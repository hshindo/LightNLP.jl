function setup_model(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    csize = size(charembeds, 1)
    conv = Conv1d(T, 5, csize, 5csize, pad=2)
    hsize = size(wordembeds,1) + 5csize
    lstm = LSTM(T, hsize, hsize, 1, 0.0, true)
    linear = Linear(T, 2hsize, ntags)
    FunctorSet(param(wordembeds), param(charembeds), conv, lstm, linear)
end

function forward(f, words, batchdims_w, chars, batchdims_c)
    wordembeds, charembeds, conv, lstm, linear = f.funs
    w = lookup(wordembeds, w)
    c = lookup(charembeds, c)
    c = dropout(c, 0.5)
    c = conv(c, batchdims_c)
    c = max(c, 2, batchdims_c)

    h = concat(1, w, c)
    h = dropout(h, 0.5)
    h = lstm(h, batchdims_w)
    h = dropout(h, 0.5)
    h = linear(h)
    h
end

function NN(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    w = lookup(param(wordembeds), Node(name="w"))

    csize = size(charembeds, 1)
    c = lookup(param(charembeds), Node(name="c"))
    c = dropout(c, 0.5)
    c = Conv1d(T,5,csize,5csize,pad=2)(c)
    c = max(c, 2)

    h = concat(1, w, c)
    h = dropout(h, 0.5)
    hsize = size(wordembeds,1) + 5csize
    h = LSTM(T,hsize,hsize,1,0.0,true)(h)
    h = dropout(h, 0.5)
    h = Linear(T,2hsize,ntags)(h)
    NN(Graph(h))

    #=
    batchdims_c = Node(name="batchdims_c")
    c = Node(window1d, c, 2, batchdims_c)
    csize = size(charembeds, 1)
    c = Node(Linear(T,5csize,5csize), c)
    c = Node(max_batch, c, batchdims_c)

    h = Node(concat, 1, w, c)
    h = Node(dropout, h, 0.5)
    batchdims_w = Node(name="batchdims_w")
    hsize = size(embeds_w,1) + 5csize
    h = Node(LSTM(T,hsize,hsize,1,0.0,true), h, batchdims_w)
    h = Node(dropout, h, 0.5)

    h = Node(Linear(T,2hsize,ntags), h)
    NN(Graph(h))
    =#
end

#=
function (nn::NN)(w, c, t)
    z = nn.g(x.batchdims_c, x.batchdims_w, Var(x.c), Var(x.w))
    if istrain()
        softmax_crossentropy(Var(x.t), z)
    else
        argmax(z.data, 1)
    end
end
=#
