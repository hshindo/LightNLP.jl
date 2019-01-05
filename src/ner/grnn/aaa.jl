mutable struct NN_BE <: Functor
end

function B2E(x::Var, dims::Vector{Int})
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

function (nn::NN_BE)(x::NamedTuple)
    B = x.h


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
    htag = nn.linear_out(h)

    if x.training
        # t = flip(x.t, nn.ntags, 0.03)
        softmax_crossentropy(x.t, h)
    else
        y = Array{Int}(Array(x.t.data))
        z = Array{Int}(Array(argmax(h.data,1)))
        vec(y), vec(z)
    end
end
