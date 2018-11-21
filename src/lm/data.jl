mutable struct Data
    data
    train::Bool
end

function Data(seq::Vector{Int}, batchsize::Int, train::Bool)
    n = length(seq) ÷ batchsize
    data = reshape(seq[1:n*batchsize], n, batchsize)
    Data(data, train)
end

Base.length(data::Data) = size(data.data,1) - 1

function Merlin.todevice(data::Data, dev::Int)
    Data(todevice(data.data,dev), data.train)
end

function Base.getindex(data::Data, indexes::Vector{Int})
    s, e = indexes[1], indexes[end]
    x = data.data[s:e,:]
    dims_x = [size(x,1) for _=1:size(x,2)]
    x = reshape(x, 1, length(x))
    y = data.data[s+1:e+1,:]
    y = reshape(y, length(y))
    (x=Var(x), y=Var(y), dims_x=dims_x, train=data.train)
end

function readdata(config)
    worddict = Dict("<eos>"=>1)
    function _readdata!(path::String)
        data = Int[]
        for line in open(readlines,path)
            elems = collect(split(chomp(line)))
            ids = map(elems) do e
                get!(worddict, e, length(worddict)+1)
            end
            append!(data, ids)
            push!(data, 1) # <eos>
        end
        batchsize = config["batchsize"]
        n = length(data) ÷ batchsize
        data = reshape(data[1:n*batchsize], n, batchsize)
        Data(data)
    end
    traindata = _readdata!(config["train_file"])
    testdata = _readdata!(config["test_file"])
    @info "#Training examples:\t$(length(traindata))"
    @info "#Testing examples:\t$(length(testdata))"
    @info "#Words:\t$(length(worddict))"
    traindata, testdata, worddict
end
