"""
    Recurrent CNN
"""
mutable struct NN_RCNN <: Functor
    wordembeds
    charembeds
    hsize
    ntags
    conv_char
    conv_h
    linear_out
end

function NN_RCNN(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    wordembeds = parameter(wordembeds)
    charembeds = parameter(charembeds)
    wsize = size(wordembeds, 1)
    csize = 2size(charembeds, 1)
    hsize = 700

    conv_char = Conv1d(T, 3, csize, csize, padding=1)
    conv_h = Conv1d(T, 3, wsize+csize+2hsize, 2hsize, padding=1)
    linear_out = Linear(T, hsize, ntags)
    NN_RCNN(wordembeds, charembeds, hsize, ntags, conv_char, conv_h, linear_out)
end

function (nn::NN_RCNN)(x::NamedTuple)
    w = lookup(nn.wordembeds, x.w)
    c = lookup(nn.charembeds, x.c)
    c = dropout(c, 0.5)
    c = nn.conv_char(c, x.dims_c)
    c = max(c, x.dims_c)
    wc = concat(1, w, c)
    wc = dropout(wc, 0.5)
    wc = dropout_dim(wc, 2, 0.2) # word-level dropout

    h = zero(w, nn.hsize, size(w,2))
    g = zero(w, nn.hsize, length(x.dims_w))
    hs = Var[]
    for t = 1:4
        g = expand(g, x.dims_w)
        h = concat(1, wc, h, g)
        h = nn.conv_h(h, x.dims_w)
        h = gate(h)
        push!(hs, h)
        g = average(h, x.dims_w)
    end
    h = concat(2, hs...)
    h = reshape(h, size(h,1), size(hs[1],2), length(hs))
    h = average(h, 3, keepdims=false)
    h = dropout(h, 0.5)
    h = nn.linear_out(h)

    if Merlin.istraining()
        softmax_crossentropy(x.t, h)
    else
        y = Array{Int}(Array(x.t.data))
        z = Array{Int}(argmax(Array(h.data),1))
        vec(y), vec(z)
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
