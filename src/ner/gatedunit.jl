mutable struct GatedUnit <: Functor
    W
    b
end

function GatedUnit(::Type{T}, insize::Int, outsize::Int) where T
    W = Normal(0,0.001)(T, insize, 2outsize)
    b = Fill(0)(T, 2outsize)
    GatedUnit(parameter(W), parameter(b))
end

function (f::GatedUnit)(head::Var, dep::Var)
    head = repeat(head, 1, size(dep,2))
    h = linear(concat(1,head,dep), f.W, f.b)
    n = size(h,1) ÷ 2
    a = tanh(h[1:n,:])
    b = sigmoid(h[n+1:2n,:])
    a .* b
end

mutable struct NN_Gate <: Functor
    wordembeds
    charembeds
    conv
    slstm
    g
    gate1
    gate2
    linear
end

function NN_Gate(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    wordembeds = parameter(wordembeds)
    charembeds = parameter(charembeds)
    csize = size(charembeds, 1)
    conv = Conv1d(T, 5, csize, 5csize, padding=2)
    hsize = size(wordembeds,1) + 5csize
    slstm = SLSTM(T, hsize, hsize)
    g = parameter(Normal(0,0.001)(T,hsize))
    gate1 = GatedUnit(T, 2hsize, hsize)
    gate2 = GatedUnit(T, 2hsize, hsize)
    linear = Linear(T, 2hsize, ntags)
    NN_Gate(wordembeds, charembeds, conv, slstm, g, gate1, gate2, linear)
end

function (nn::NN_Gate)(x::NamedTuple)
    w = lookup(nn.wordembeds, x.w)
    c = lookup(nn.charembeds, x.c)
    c = nn.conv(c, x.dims_c)
    c = max(c, x.dims_c)

    h = concat(1, w, c)
    h = dropout(h, 0.5, x.training)
    h = nn.slstm(h, x.dims_w)
    h = dropout(h, 0.5, x.training)
    #=
    g = nn.g
    for i = 1:2
        off = 0
        hs = Var[]
        for d in x.dims_w
            hh = h[:,off+1:off+d]
            g = average(nn.gate1(g,hh), 2)
            hh = nn.gate2(g, hh)
            push!(hs, hh)
            off += d
        end
        h = concat(2, hs...)
        #h = dropout(h, 0.5, x.training)
    end
    =#
    h = nn.linear(h)
    if x.training
        softmax_crossentropy(x.t, h)
    else
        y = Array{Int}(Array(x.t.data))
        z = Array{Int}(Array(argmax(h.data,1)))
        vec(y), vec(z)
    end
end
