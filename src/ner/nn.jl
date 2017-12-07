struct NN
    g
end

function NN(wordembeds1::Var, wordembeds2::Vector{Var}, charembeds::Vector{Var}, ntags::Int)
    T = Float32
    w1 = Node("w1")
    w1 = lookup(wordembeds1, w1)
    w2 = Node("w2")
    w2 = lookup(wordembeds2, w2)

    c = Node("c")
    batchdims_c = Node("batchdims_c")
    c = lookup(charembeds, c)
    d = size(charembeds[1], 1)
    c = Conv1D(T,5,d,5d,2,1)(c,batchdims_c)
    c = max_batch(c, batchdims_c)

    h = cat(1, w1, w2, c)
    batchdims = Node("batchdims")
    d = 200 + 5size(charembeds[1],1)
    dh = 300
    h = Conv1D(T,5,d,dh,2,1)(h,batchdims)
    h = relu(h)

    istrain = Node("train")
    for i = 1:2
        h = dropout(h, 0.3, istrain)
        h = Conv1D(T,5,dh,dh,2,1)(h,batchdims)
        h = relu(h)
    end
    h = Linear(T,dh,ntags)(h)
    g = Graph(h)
    NN(g)
end

function (nn::NN)(w1::Var, w2::Var, c::Vector{Vector{Var}}, t=nothing)
    batchdims = mao(length, c)
    batchdims_c = Int[]
    for x in c
        append!(batchdims_c, map(length,x))
    end
    if t == nothing
        y = nn.g("w1"=>w1, "w2"=>w2, "c"=>c, "batchdims_c"=>batchdims_c, "batchdims"=>batchdims, "train"=>false)
        argmax(y.data, 1)
    else
        y = nn.g("w1"=>w1, "w2"=>w2, "c"=>c, "batchdims_c"=>batchdims_c, "batchdims"=>batchdims, "train"=>true)
        softmax_crossentropy(t, y)
    end
end

#=
function (nn::NN)(w1::Var, w2::Var, c::Var, t=nothing)
    if t == nothing
        y = nn.g("w1"=>w1, "w2"=>w2, "c"=>c, "train"=>false)
        argmax(y.data, 1)
    else
        y = nn.g("w1"=>w1, "w2"=>w2, "c"=>c, "train"=>true)
        softmax_crossentropy(t, y)
    end
end
=#
