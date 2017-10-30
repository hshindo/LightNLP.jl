function setup_nn(ntags::Int)
    T = Float32
    x = Node()
    h = Linear(T,10,10)(x)
    h = relu(h)
    h = Conv1D(T,5,d,dh,2,1)(h)
    h = relu(h)
    h = Linear(T,10,ntags)(h)
    Graph(input=x, output=h)
end
