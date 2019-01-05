mutable struct GatedLinear <: Functor
    l1::Linear
    W
end

function GatedLinear(::Type{T}, insize, outsize) where T
    l1 = Linear(T, insize, 2outsize)
    GatedLinear(l1, nothing)
end

function weightdrop!(f::GatedLinear)
    f.W = dropout(f.l1.W, 0.5)
end

function (f::GatedLinear)(x::Var)
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
    charembeds
    hsize
    ntags
    conv_char
    linear_word
    conv_word2
    conv_g
    linear_out
end

function NN_Graph(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    wordembeds = parameter(wordembeds)
    charembeds = parameter(charembeds)
    csize = size(charembeds, 1) * 2
    #conv = Conv1d(T, 3, 2csize, 5csize, padding=1)
    conv_char = Conv1d(T, 3, csize, csize, padding=1)
    wsize = size(wordembeds, 1)
    hsize = 500
    linear_word = GatedLinear(T, 2wsize+hsize, wsize)
    conv_word2 = GatedLinear(T, 3*(hsize+wsize+csize)+hsize, hsize)
    conv_g = Conv1d(T, 3, hsize, hsize, padding=1)
    #conv_word2 = Linear(T, 7hsize, 2hsize)
    # attention = AddAttention2(T, hsize)
    linear_out = Linear(T, hsize, ntags)
    NN_Graph(wordembeds, charembeds, hsize, ntags, conv_char, linear_word, conv_word2, conv_g, linear_out)
end

function (nn::NN_Graph)(x::NamedTuple)
    w0 = lookup(nn.wordembeds, x.w)
    w0 = dropout_dim(w0, 0.1)
    w0 = dropout(w0, 0.5)
    w = Var(zero(w0.data))

    c = lookup(nn.charembeds, x.c)
    c = dropout_dim(c, 0.1)
    c = dropout(c, 0.5)
    c = nn.conv_char(c, x.dims_c)
    c = max(c, x.dims_c)
    #wc = concat(1, w, c)

    hsize = nn.hsize
    #weightdrop!(nn.conv_word2)
    h = Var(fill!(similar(w0.data,hsize,size(w0,2)),0))
    g = Var(fill!(similar(w0.data,hsize,length(x.dims_w)),0))
    hs = Var[]
    for i = 1:4
        #w = concat(1, w, w0, h)
        #w = nn.linear_word(w)
        w = w0

        gs = Var[]
        for k = 1:length(x.dims_w)
            d = x.dims_w[k]
            gg = repeat(g[:,k:k], 1, d)
            push!(gs, gg)
        end
        g0 = concat(2, gs...)

        h0 = concat(1, h, w, c)
        h0 = window1d(h0, x.dims_w, 3, 1, 1, 1)
        h0 = concat(1, h0, g0)
        h = nn.conv_word2(h0)
        push!(hs, h)

        #g = nn.conv_g(h, x.dims_w)
        #g = max(g, x.dims_w)
        g = average(h, x.dims_w)

        #g = window1d(h, x.dims_w, 11, 3, 1, 1)
        #g = reshape(g, size(h,1), 11, size(h,2))
        #g = average(g, 2, keepdims=false)
        # h = zoneout(h, h1, 0.1, x.training)
    end
    h = concat(2, hs...)
    h = reshape(h, hsize, size(hs[1],2), length(hs))
    h = average(h, 3, keepdims=false)
    h = dropout(h, 0.5)
    h = nn.linear_out(h)

    if Merlin.istraining()
        # t = flip(x.t, nn.ntags, 0.03)
        softmax_crossentropy(x.t, h)
    else
        y = Array{Int}(Array(x.t.data))
        z = Array{Int}(Array(argmax(h.data,1)))
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
