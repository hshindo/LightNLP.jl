mutable struct NN_RANN <: Functor
    wordembeds
    charembeds
    hsize
    conv_char
    l_h
    l_out
    l_categ
end

function NN_RANN(wordembeds::Matrix{T}, charembeds::Matrix{T}, ncategs) where T
    wordembeds = parameter(wordembeds)
    charembeds = parameter(charembeds)
    wsize = size(wordembeds, 1)
    csize = 3size(charembeds, 1)
    # dropword = parameter(Uniform(-0.01,0.01)(T,wsize+csize))

    hsize = 700
    conv_char = Conv1d(T, 3, csize, csize, padding=1)
    l_h = Linear(T, 2(hsize+wsize+csize), 2hsize)
    # conv_h = Conv1d(T, 3, wsize+csize+2hsize, 2hsize, padding=1)
    l_out = Linear(T, hsize, 5)
    l_categ = Linear(T, hsize, ncategs)
    NN_RANN(wordembeds, charembeds, hsize, conv_char, l_h, l_out, l_categ)
end

function (nn::NN_RANN)(samples::Vector{Sample}, indexes::Vector{Int})
    x = Sample(samples, indexes)
    w = lookup(nn.wordembeds, x.word)
    c = lookup(nn.charembeds, x.char)
    c = nn.conv_char(c, x.dims_char)
    c = max(c, x.dims_char)
    wc = concat(1, w, c)
    wc = dropout(wc, 0.5)
    # wc = replace1d(wc, 0.33, nn.dropword)
    wc = dropout_dim(wc, 2, 0.2) # word-level dropout

    h = zero(w, nn.hsize, size(w,2))
    hs = Var[]
    for i = 1:5
        h = concat(1, wc, h)
        h, dims_h = attention(h, x.dims_word)
        h = nn.l_h(h)
        # h = nn.conv_h(h, x.dims_word)
        h = gate(h)
        h = average(h, dims_h)
        push!(hs, h)
    end
    # h = concat(2, hs...)
    # h = reshape(h, size(h,1), size(hs[1],2), length(hs))
    h = dropout(h, 0.5)
    # h = average(h, dims=3, keepdims=false)
    o = nn.l_out(h)

    if Merlin.istraining()
        l1 = softmax_crossentropy(x.tag, o)
        if isempty(x.span.data)
            l1
        else
            s = lookup(h, x.span)
            s = dropout(s, 0.5)
            s = average(s, x.dims_span)
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
        o = Array{Int}(Array(x.tag.data))
        spans = bioes2span(vec(o))
        s = Array{Int}(Array(x.categ.data))
        y = map(x -> tuple(x[1]...,x[2]), zip(spans,vec(s)))
        y, z
    end
end

function attention(x::Var, dims::Vector{Int})
    indexes = Int[]
    off = 0
    dims_p = Int[]
    for k = 1:length(dims)
        d = dims[k]
        for i = 1:d
            # for i = 1:dims[k]
            n = min(i+1,d) - max(1,i-1)
            @assert n > 0
            push!(dims_p, n)
            for j = max(1,i-1):min(i+1,d)
                i == j && continue
                push!(indexes, off+i, off+j)
            end
        end
        off += dims[k]
    end
    indexes = reshape(indexes, 2, length(indexes)÷2)
    indexes = todevice(indexes)
    lookup(x, Var(indexes)), dims_p
end
