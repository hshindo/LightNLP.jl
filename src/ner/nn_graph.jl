using Merlin.CUDA

mutable struct GatedLinear <: Functor
    l1::Linear
end

function GatedLinear(::Type{T}, insize, outsize) where T
    l1 = Linear(T, insize, 2outsize)
    GatedLinear(l1)
end

function weightdrop!(f::GatedLinear)
    W = f.l1.W
    W = dropout(W, 0.5)
    W = dropout_dim(W, 1, 0.1)
    # W = dropout_dim(W, 2, 0.25)
    # W = dropout(W, 0.5)
    f.W = W
end

function (f::GatedLinear)(x::Var)
    #f.W == nothing && (f.W = f.l1.W)
    #h = linear(x, f.W, f.l1.b)
    h = f.l1(x)
    n = size(h,1) ÷ 2
    a = tanh(h[1:n,:])
    b = sigmoid(h[n+1:2n,:])
    h = a .* b
    h
end

mutable struct NN_Graph <: Functor
    wordembeds
    flair_train
    flair_test
    charembeds
    hsize
    ntags
    conv_char
    conv_word2
    lstm
    linear_out
    crf
end

function NN_Graph(wordembeds::Matrix{T}, flair_train, flair_test, charembeds::Matrix{T}, ntags::Int) where T
    wordembeds = parameter(wordembeds)
    charembeds = parameter(charembeds)
    csize = 2size(charembeds, 1)
    #conv = Conv1d(T, 3, 2csize, 5csize, padding=1)
    conv_char = Conv1d(T, 3, csize, csize, padding=1)
    #wsize = size(wordembeds, 1)
    hsize = 600
    # wsize = size(wordembeds, 1)
    wsize = size(wordembeds, 1)

    lstmsize = wsize + size(flair_train, 1)
    lstm_out = 256
    lstm = LSTM(T, lstmsize, lstm_out, 1, 0.0, true)
    # conv_word2 = GatedLinear(T, 3*(hsize+wsize+csize)+hsize, hsize)
    conv_word2 = GatedLinear(T, 3*(hsize+wsize+csize+2lstm_out)+hsize, hsize)
    #conv_word2 = Conv1d(T, 3, hsize+wsize+csize+ntags, 2hsize, padding=1, ngroups=1)

    linear_out = Linear(T, hsize, ntags)
    #crf = RNNCRF(T, ntags, hsize)
    crf = nothing
    NN_Graph(wordembeds, flair_train, flair_test, charembeds, hsize, ntags, conv_char, conv_word2, lstm, linear_out, crf)
end

function (nn::NN_Graph)(x::NamedTuple)
    w0 = lookup(nn.wordembeds, x.w)
    w = w0
    # w = Var(zero(w0.data))
    if Merlin.istraining()
        fw = lookup(nn.flair_train, x.count)
    else
        fw = lookup(nn.flair_test, x.count)
    end
    h1 = concat(1, w, fw)
    h1 = dropout(h1, 0.5)
    h1, _, _ = nn.lstm(h1, x.dims_w)

    #wc = concat(1, w, fw)
    # w = dropout(w, 0.5)

    c = lookup(nn.charembeds, x.c)
    c = dropout(c, 0.5)
    c = nn.conv_char(c, x.dims_c)
    c = max(c, x.dims_c)
    wc = concat(1, w, c, h1)
    wc = dropout(wc, 0.5)
    wc = dropout_dim(wc, 2, 0.2)

    hsize = nn.hsize
    h = Var(fill!(similar(w0.data,hsize,size(w0,2)),0))
    g = Var(fill!(similar(w0.data,hsize,length(x.dims_w)),0))
    # dropout_g = LockedDropout(0.5)
    # dropout_word = LockedDropout(0.2)
    # o = Var(fill!(similar(w0.data,nn.ntags,size(w0,2)),0))
    hs = Var[]
    for i = 1:4
        gs = Var[]
        for k = 1:length(x.dims_w)
            d = x.dims_w[k]
            gg = repeat(g[:,k:k], 1, d)
            push!(gs, gg)
        end
        g0 = concat(2, gs...)

        h0 = concat(1, h, wc)
        h0 = window1d(h0, x.dims_w, 3, 1, 1, 1)
        h0 = concat(1, h0, g0)
        h = nn.conv_word2(h0)
        push!(hs, h)

        #g = nn.conv_g(h, x.dims_w)
        #g = max(g, x.dims_w)
        g = average(h, x.dims_w)
        # g = dropout_g(g)
        #g = conv_g()
        #g = concat(1, g, g0)
        #g = nn.conv_g(g)

        #g = window1d(h, x.dims_w, 11, 3, 1, 1)
        #g = reshape(g, size(h,1), 11, size(h,2))
        #g = average(g, 2, keepdims=false)
        # h = zoneout(h, h1, 0.1, x.training)
    end
    h = concat(2, hs...)
    h = reshape(h, size(h,1), size(hs[1],2), length(hs))
    h = average(h, 3, keepdims=false)
    # h = nn.crf(o, x.dims_w, 4)
    h = dropout(h, 0.5)
    o = nn.linear_out(h)
    h = o

    if Merlin.istraining()
        # t = flip(x.t, nn.ntags, 0.03)
        softmax_crossentropy(x.t, h)
    else
        # Merlin.printw(nn.crf)
        y = Array{Int}(Array(x.t.data))
        z = Array{Int}(argmax(Array(h.data),1))
        vec(y), vec(z)
    end
end

function maxout(x::Var)
    n = size(x,1) ÷ 2
    a = x[1:n,:]
    b = x[n+1:2n,:]
    c = concat(2, a, b)
    c = pack(c, [size(x,2),size(x,2)], 0)
    c = max(c, 3, keepdims=false)
    tanh(c)
end

function gate(x::Var)
    n = size(x,1) ÷ 5
    f = sigmoid(x[1:n,:])
    i = sigmoid(x[n+1:2n,:])
    o = sigmoid(x[2n+1:3n,:])
    c = x[3n+1:4n,:]
    c = (f .* c) + (i .* tanh(x[4n+1:5n,:]))
    o .* tanh(c)
end

function gate2(x::Var)
    n = size(x,1) ÷ 2
    a = tanh(x[1:n,:])
    b = sigmoid(x[n+1:2n,:])
    a .* b
end

function leftglobal(x::Var, dims::Vector{Int})
    @assert ndims(x) == 2
    idx = Cint[]
    ydims = Int[]
    off = 0
    for d in dims
        for i = 1:d
            append!(idx, off+1:off+i)
            append!(ydims, i)
        end
        off += d
    end
    idx = reshape(idx, 1, length(idx))
    idx = Merlin.todevice!(Var(idx))
    h = lookup(x, idx)
    y = average(h, ydims)
    y
end

function rightglobal(x::Var, dims::Vector{Int})
    @assert ndims(x) == 2
    idx = Cint[]
    ydims = Int[]
    off = 0
    for d in dims
        for i = 1:d
            r = i >= d-1 ? (0:0) : (off+i+2:off+d)
            append!(idx, r)
            append!(ydims, length(r))
        end
        off += d
    end
    idx = reshape(idx, 1, length(idx))
    idx = Merlin.todevice!(Var(idx))
    h = lookup(x, idx)
    y = average(h, ydims)
    y
end

function grugate(x::Var, h::Var)
    n = size(x,1) ÷ 2
    u = sigmoid(x[1:n,:])
    r = tanh(x[n+1:2n,:])
    one = Var(fill!(similar(u.data),1))
    o = (one-u) .* h + u .* r
    o
end

function dgate(x::Var, h::Var)
    xg = linear(x)
    zg = linear(xg .* h)
    zout = linear(zg .* h)
    z = linear(concat(x,h))
    h = (1-z) .* h + z .* zout
end
