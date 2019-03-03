"""
    Recurrent CNN
"""
mutable struct NN_RCNN <: Functor
    wordembeds
    charembeds
    hsize
    conv_char
    conv_h
    l_out
    l_categ
end

function NN_RCNN(wordembeds::Matrix{T}, charembeds::Matrix{T}, ncategs) where T
    wordembeds = parameter(wordembeds)
    charembeds = parameter(charembeds)
    wsize = size(wordembeds, 1)
    csize = 3*size(charembeds, 1)
    # dropword = parameter(Uniform(-0.01,0.01)(T,wsize+csize))

    hsize = 700
    conv_char = Conv1d(T, 3, csize, csize, padding=1)
    conv_h = Conv1d(T, 3, wsize+csize+2hsize, 2hsize, padding=1)
    l_out = Linear(T, hsize, 5)
    l_categ = Linear(T, hsize, ncategs)
    NN_RCNN(wordembeds, charembeds, hsize, conv_char, conv_h, l_out, l_categ)
end

function (nn::NN_RCNN)(x::NamedTuple)
    w = lookup(nn.wordembeds, x.w)
    c = lookup(nn.charembeds, x.c)
    c = nn.conv_char(c, x.dims_c)
    c = max(c, x.dims_c)
    wc = concat(1, w, c)
    wc = dropout(wc, 0.5)
    # wc = replace1d(wc, 0.33, nn.dropword)
    wc = dropout_dim(wc, 2, 0.2) # word-level dropout

    h = zero(w, nn.hsize, size(w,2))
    g = zero(w, nn.hsize, length(x.dims_w))
    hs = Var[]
    for i = 1:5
        g = expand(g, x.dims_w)
        h = concat(1, wc, h, g)
        h = nn.conv_h(h, x.dims_w)
        h = gate(h)
        push!(hs, h)
        g = average(h, x.dims_w)
    end
    h = concat(2, hs...)
    h = reshape(h, size(h,1), size(hs[1],2), length(hs))
    h = average(h, dims=3, keepdims=false)
    h = dropout(h, 0.5)
    o = nn.l_out(h)

    if Merlin.istraining()
        l1 = softmax_crossentropy(x.t, o)
        if isempty(x.s.data)
            l1
        else
            s = lookup(h, x.s)
            s = average(s, x.dims_s)
            s = dropout(s, 0.5)
            s = nn.l_categ(s)
            l2 = softmax_crossentropy(x.categ, s)
            l1 + l2
        end
    else
        o = Array{Int}(Array(argmax(o,dims=1)))
        spans = bioes2span(vec(o))
        if isempty(spans)
            z = []
        else
            s, dims_s = expand_span(spans)
            s = todevice(Var(s))
            s = lookup(h, s)
            s = average(s, dims_s)
            s = dropout(s, 0.5)
            s = nn.l_categ(s)
            s = Array{Int}(Array(argmax(s,dims=1)))
            z = map(x -> tuple(x[1]...,x[2]), zip(spans,vec(s)))
        end
        o = Array{Int}(Array(x.t.data))
        spans = bioes2span(vec(o))
        s = Array{Int}(Array(x.categ.data))
        y = map(x -> tuple(x[1]...,x[2]), zip(spans,vec(s)))
        y, z
    end
end

function gate(x::Var)
    n = size(x,1) ÷ 2
    a = tanh(x[1:n,:])
    b = sigmoid(x[n+1:2n,:])
    a .* b
end

function expand(x::Var, dims)
    ys = Var[]
    for i = 1:length(dims)
        d = dims[i]
        y = repeat(x[:,i:i], 1, d)
        push!(ys, y)
    end
    concat(2, ys...)
end

function expand_span(spans::Vector{Tuple{Int,Int}})
    s = Int[]
    dims_s = Int[]
    for (i,j) in spans
        append!(s, i:j)
        push!(dims_s, j-i+1)
    end
    s = reshape(s, 1, length(s))
    s, dims_s
end
