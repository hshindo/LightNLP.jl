mutable struct NN_Graph <: Functor
    wordembeds
    charembeds
    tagembeds
    ntags
    conv_char
    conv_word1
    conv_word2
    conv_g
    linear_out
end

function NN_Graph(wordembeds::Matrix{T}, charembeds::Matrix{T}, tagembeds::Matrix{T}) where T
    wordembeds = parameter(wordembeds)
    charembeds = parameter(charembeds)
    tagembeds = parameter(tagembeds)
    ntags = size(tagembeds, 2)
    csize = size(charembeds, 1) * 2
    #conv = Conv1d(T, 3, 2csize, 5csize, padding=1)
    conv_char = Conv1d(T, 3, csize, csize, padding=1)
    #conv_char = Linear(T, csize, csize)
    #hsize = size(wordembeds,1) + 5csize
    hsize = size(wordembeds,1) + csize
    # conv_word = Conv1d(T, 3, 3hsize, 2hsize, padding=1)
    conv_word1 = Conv1d(T, 3, hsize, 2hsize, padding=1)
    conv_word2 = Linear(T, 4hsize, 2hsize)
    conv_g = Linear(T, hsize, hsize)
    #conv_word2 = Linear(T, 7hsize, 2hsize)
    # attention = AddAttention2(T, hsize)
    tsize = size(tagembeds, 1)
    linear_out = Linear(T, 5hsize, ntags)
    NN_Graph(wordembeds, charembeds, tagembeds, ntags, conv_char, conv_word1, conv_word2,
    conv_g, linear_out)
end

function (nn::NN_Graph)(x::NamedTuple)
    w = lookup(nn.wordembeds, x.w)
    c = lookup(nn.charembeds, x.c)
    c = nn.conv_char(c, x.dims_c)
    c = max(c, x.dims_c)
    wc = concat(1, w, c)
    wc = dropout(wc, 0.5)

    #g = Var(fill!(similar(wc.data,size(wc,1),length(x.dims_w)),0))
    hbase = nn.conv_word1(wc, x.dims_w)
    h = gate2(hbase)
    hs = Var[]
    for i = 1:5
        g = average(h, x.dims_w)
        gs = Var[]
        for k = 1:length(x.dims_w)
            d = x.dims_w[k]
            gg = repeat(g[:,k:k], 1, d)
            push!(gs, gg)
        end
        g = concat(2, gs...)

        h0 = window1d(h, x.dims_w, 3, 1, 1, 1)
        h0 = concat(1, h0, g)
        h0 = nn.conv_word2(h0)
        h = h0 + hbase
        h = gate2(h)
        #g = window1d(h, x.dims_w, 11, 3, 1, 1)
        #g = reshape(g, size(h,1), 11, size(h,2))
        #g = average(g, 2, keepdims=false)
        # h = zoneout(h, h1, 0.1, x.training)
        push!(hs, h)
    end
    h = concat(1, hs...)
    h = dropout(h, 0.5)
    h = nn.linear_out(h)

    if x.training
        # t = flip(x.t, nn.ntags, 0.03)
        softmax_crossentropy(x.t, h)
    else
        y = Array{Int}(Array(x.t.data))
        z = Array{Int}(Array(argmax(h.data,1)))
        vec(y), vec(z)
    end
end

#=
function (nn::NN_Graph)(x::NamedTuple)
    w = lookup(nn.wordembeds, x.w)
    c = lookup(nn.charembeds, x.c)
    #c = dropout(c, 0.5, x.training)

    #=
    h = repeat(nn.c0, 1, size(c,2))
    g = max(h, x.dims_c)
    hs = Var[]
    for i = 1:3
        gs = Var[]
        for k = 1:length(x.dims_c)
            d = x.dims_c[k]
            gg = repeat(g[:,k:k], 1, d)
            push!(gs, gg)
        end
        gg = concat(2, gs...)
        h0 = concat(1, h, c)
        h0 = window1d(h0, x.dims_c, 3, 1, 1, 1)
        h0 = concat(1, h0, gg)
        h0 = nn.conv_char(h0)
        h = gate2(h0)
        g = max(h, x.dims_c)
        push!(hs, g)
        #h = concat(1, h, c)
        #h = nn.conv_char(h, x.dims_c)
        #h = gate2(h)
    end
    h = concat(1, hs...)
    c = h
    =#
    c = nn.conv_char(c, x.dims_c)
    c = max(c, x.dims_c)
    wc = concat(1, w, c)
    wc = dropout(wc, 0.5, x.training)

    #h = repeat(nn.w0, 1, size(wc,2))
    h = Var(fill!(similar(wc.data),0))
    g = Var(fill!(similar(wc.data,size(wc,1),length(x.dims_w)),0))
    hs = Var[]
    for i = 1:3
        gs = Var[]
        for k = 1:length(x.dims_w)
            d = x.dims_w[k]
            gg = repeat(g[:,k:k], 1, d)
            push!(gs, gg)
        end
        gg = concat(2, gs...)
        h0 = concat(1, h, wc)
        h0 = window1d(h0, x.dims_w, 3, 1, 1, 1)
        h0 = concat(1, h0, gg)
        h1 = nn.conv_word(h0)
        h = gate2(h1)
        # h = zoneout(h, h1, 0.1, x.training)
        push!(hs, h)

        #h2 = nn.conv_word2(h0)
        #h2 = gate2(h2)
        g = max(h, x.dims_w)
    end
    h = concat(1, hs...)
    h = dropout(h, 0.5, x.training)

    h = nn.linear_out(h)
    if x.training
        # t = flip(x.t, nn.ntags, 0.03)
        softmax_crossentropy(x.t, h)
    else
        y = Array{Int}(Array(x.t))
        z = Array{Int}(Array(argmax(h.data,1)))
        vec(y), vec(z)
    end
end
=#

#=
function (nn::NN_Graph)(x::NamedTuple)
    w = lookup(nn.wordembeds, x.w)
    c = lookup(nn.charembeds, x.c)
    c = nn.conv(c, x.dims_c)
    c = max(c, x.dims_c)
    wc = concat(1, w, c)
    wc = dropout(wc, 0.5, x.training)

    h = Var(fill!(similar(wc.data),0))
    for i = 1:5
        h = concat(1, h, wc)
        h = nn.linear_conv(h, x.dims_w)
        h = gate(h, 2)
    end

    h = dropout(h, 0.5, x.training)
    h = nn.linear_out(h)
    if x.training
        softmax_crossentropy(x.t, h)
    else
        y = Array{Int}(Array(x.t.data))
        z = Array{Int}(Array(argmax(h.data,1)))
        vec(y), vec(z)
    end
end
=#

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
