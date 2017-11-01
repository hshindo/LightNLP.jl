struct NN
    g
end

function NN(nchars::Int, ntags::Int)
    T = Float32
    x = Node()
    embeds = embeddings(T, nchars, 20, init_w=Normal(0,0.05))
    h = lookup(embeds, x)
    for i = 1:2
        h = Conv1D(T,5,20,20,2,1)(h)
        h = relu(h)
    end
    h = Linear(T,20,ntags)(h)
    Graph(input=x, output=h)
end

function (nn::NN)(x, y=nothing)
    if y == nothing
        z = nn.g(x, false)
        argmax(z.data, 1)
    else
        z = nn.g(x, true)
        softmax_crossentropy(y, z)
    end
end
