mutable struct NN
    embeds_w
    embeds_c
    conv_c
    conv_hs
    out
end

function NN(embeds_w::Var, embeds_c::Var, ntags::Int)
    T = Float32
    hsize = 300
    conv_c = Conv(T, 5,d,5d,2,1)
    conv_h = Conv(T)
    conv_hs = [Conv(T) for i=1:2]
    out = Linear(T, hsize, ntags)
    NN(embeds_w, embeds_c, conv_c, conv_hs, out)
end

function (nn::NN)(w, c, y=nothing)
    w = nn.embeds_w(w)
    c = nn.embeds_c(c)
    c = nn.conv_c(c)
    c = maximum_batch(c)

    h = concat(1, w, c)
    d = 100 + 5size(charembeds[1],1)
    dh = 300
    h = nn.conv_h(h,batchdims_w)
    h = relu(h)
    for i = 1:2
        h = dropout(h, y == nothing ? 0.0 : 0.3)
        h = nn.conv_hs[i](h)
        h = relu(h)
    end
    h = nn.linear_out(h)
    h
end

struct NN
    g
end

function NN(wordembeds::Vector{Var}, charembeds::Vector{Var}, ntags::Int)
    T = Float32
    w = lookup(wordembeds, Node("w"))
    batchdims_c = Node("batchdims_c")
    c = lookup(charembeds, Node("c"))
    d = size(charembeds[1], 1)
    c = Conv1D(T,5,d,5d,2,1)(c,batchdims_c)
    c = max_batch(c, batchdims_c)

    h = cat(1, w, c)
    batchdims_w = Node("batchdims_w")
    d = 100 + 5size(charembeds[1],1)
    dh = 300
    h = Conv1D(T,5,d,dh,2,1)(h,batchdims_w)
    h = leaky_relu(h)

    istrain = Node("train")
    for i = 1:2
        h = dropout(h, 0.3, istrain)
        h = Conv1D(T,5,dh,dh,2,1)(h,batchdims_w)
        h = leaky_relu(h)
    end
    h = Linear(T,dh,ntags)(h)
    g = Graph(h)
    NN(g)
end
