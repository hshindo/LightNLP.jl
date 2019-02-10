mutable struct NN_SLSTM <: Functor
    wordembeds
    charembeds
    conv
    slstm
    linear
end

function NN_SLSTM(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    wordembeds = parameter(wordembeds)
    charembeds = parameter(charembeds)
    csize = size(charembeds, 1)
    conv = Conv1d(T, 5, csize, 5csize, padding=2)
    hsize = size(wordembeds,1) + 5csize
    slstm = SLSTM(T, hsize, hsize)
    linear = Linear(T, 2hsize, ntags)
    NN_SLSTM(wordembeds, charembeds, conv, slstm, linear)
end

function (nn::NN_SLSTM)(x::NamedTuple)
    w = lookup(nn.wordembeds, x.w)
    c = lookup(nn.charembeds, x.c)
    c = nn.conv(c, x.dims_c)
    c = max(c, x.dims_c)

    h = concat(1, w, c)
    h = dropout(h, 0.5, x.training)
    h = nn.slstm(h, x.dims_w)
    h = dropout(h, 0.5, x.training)
    h = nn.linear(h)
    if x.training
        softmax_crossentropy(x.t, h)
    else
        y = Array{Int}(Array(x.t.data))
        z = Array{Int}(Array(argmax(h.data,1)))
        vec(y), vec(z)
    end
end
