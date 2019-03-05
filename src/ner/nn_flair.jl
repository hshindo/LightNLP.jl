mutable struct NN_Flair <: Functor
    wordembeds
    charembeds
    hsize
    conv_char
    conv_h
    l_out
    l_categ
end

function NN_Flair(wordembeds::Matrix{T}, charembeds::Matrix{T}, ncategs, flairdim) where T
    wordembeds = parameter(wordembeds)
    charembeds = parameter(charembeds)
    wsize = size(wordembeds, 1) + flairdim
    csize = 3*size(charembeds, 1)

    hsize = 700
    conv_char = Conv1d(T, 3, csize, csize, padding=1)
    conv_h = Conv1d(T, 3, wsize+csize+2hsize, 2hsize, padding=1)
    l_out = Linear(T, hsize, 5)
    l_categ = Linear(T, hsize, ncategs)
    NN_Flair(wordembeds, charembeds, hsize, conv_char, conv_h, l_out, l_categ)
end

function (nn::NN_Flair)(x::NamedTuple)
    w = lookup(nn.wordembeds, x.w)
    c = lookup(nn.charembeds, x.c)
    c = nn.conv_char(c, x.dims_c)
    c = max(c, x.dims_c)
    wc = concat(1, w, c, x.flair)
    # wc = dropout(wc, 0.5)
    wc = dropout_dim(wc, 2, 0.33) # word-level dropout

    h = zero(w, nn.hsize, size(w,2))
    g = zero(w, nn.hsize, length(x.dims_w))
    drop = LockedDropout(0.5)
    hs = Var[]
    for i = 1:4
        g = expand(g, x.dims_w)
        h = concat(1, wc, h, g)
        h = drop(h)
        h = nn.conv_h(h, x.dims_w)
        h = gate(h)
        push!(hs, h)
        g = average(h, x.dims_w)
    end
    h = concat(2, hs...)
    h = reshape(h, size(h,1), size(hs[1],2), length(hs))
    h = dropout(h, 0.5)
    h = average(h, dims=3, keepdims=false)
    o = nn.l_out(h)

    if Merlin.istraining()
        l1 = softmax_crossentropy(x.t, o)
        if isempty(x.s.data)
            l1
        else
            s = lookup(h, x.s)
            s = dropout(s, 0.5)
            s = average(s, x.dims_s)
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
            s = dropout(s, 0.5)
            s = average(s, dims_s)
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
