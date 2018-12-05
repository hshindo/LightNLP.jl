mutable struct SLSTM <: Functor
    insize::Int
    hsize::Int
    W1
    b1
    W2
    b2
    h1
    h2
end

function SLSTM(::Type{T}, insize::Int, hsize::Int;
    init_W=Normal(0,0.001), init_U=Normal(0,0.001), init_b=Fill(0)) where T

    W1 = init_W(T, insize+hsize, 2hsize)
    b1 = init_b(T, 2hsize)
    W2 = init_W(T, insize+hsize, 2hsize)
    b2 = init_b(T, 2hsize)
    h1 = Fill(0)(T, hsize)
    h2 = Fill(0)(T, hsize)
    SLSTM(insize, hsize, parameter(W1), parameter(b1), parameter(W2), parameter(b2), parameter(h1), parameter(h2))
end

function (f::SLSTM)(x::Var, dims)
    @assert ndims(x) == 2
    @assert sum(dims) == size(x,2)
    @assert issorted(dims, rev=true)
    hx1 = repeat(f.h1, 1, length(dims))
    hx2 = repeat(f.h2, 1, length(dims))
    #hx = similar(x.data, f.hsize, length(dims))
    #fill!(hx, 0)
    #hx = Var(hx)

    y = x
    y1 = slstm_tstep(y, dims, f.hsize, f.W1, f.b1, hx1, false)
    y2 = slstm_tstep(y, dims, f.hsize, f.W2, f.b2, hx2, true)
    y = concat(1, y1, y2)
    y
end

function slstm_tstep(x::Var, dims, hsize::Int, W::Var, b::Var, hx::Var, rev::Bool)
    cumdims = Array{Int}(undef, length(dims)+1)
    cumdims[1] = 1
    for i = 1:length(dims)
        cumdims[i+1] = cumdims[i] + dims[i]
    end

    ht = hx
    hts = Array{Var}(undef, size(x,2))
    for t = 1:dims[1]
        xts = Var[]
        for j = 1:length(dims)
            d = dims[j]
            t > d && break
            k = cumdims[j]
            k += rev ? d-t : t-1
            push!(xts, x[:,k:k])
        end
        xt = concat(2, xts...)
        if size(ht,2) > size(xt,2)
            ht = ht[:,1:size(xt,2)]
        end
        ht = slstm_onestep(hsize, xt, W, b, ht)
        for j = 1:length(dims)
            d = dims[j]
            t > d && break
            k = cumdims[j]
            k += rev ? d-t : t-1
            hts[k] = ht[:,j:j]
        end
    end
    y = concat(2, hts...)
    y
end

function slstm_onestep(hsize::Int, xt::Var, W::Var, b::Var, ht::Var)
    a = linear(concat(1,xt,ht), W, b)
    n = size(a,1) ÷ 2
    h = tanh(a[1:n,:])
    g = sigmoid(a[n+1:2n,:])
    ht = h .* g
    ht

    #i = sigmoid(a[1:n,:])
    #f = sigmoid(a[n+1:2n,:])
    #ct = (f .* ct) + (i .* tanh(a[2n+1:3n,:]))
    #o = sigmoid(a[3n+1:4n,:])
    #ht = o .* tanh(ct)
    #ht, ct
end
