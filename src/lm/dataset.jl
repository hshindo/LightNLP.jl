mutable struct Dataset
    data
    training::Bool
end

function Dataset(seq::Vector{Int}, batchsize::Int, training::Bool)
    n = length(seq) ÷ batchsize
    data = reshape(seq[1:n*batchsize], n, batchsize)
    Dataset(data, training)
end

Base.length(dataset::Dataset) = size(dataset.data,1) - 1

function Merlin.todevice(dataset::Dataset, dev::Int)
    Dataset(todevice(dataset.data,dev), dataset.training)
end

function Base.getindex(dataset::Dataset, indexes::Vector{Int})
    s, e = indexes[1], indexes[end]
    x = dataset.data[s:e,:]
    dims_x = [size(x,1) for _=1:size(x,2)]
    x = reshape(x, 1, length(x))
    y = dataset.data[s+1:e+1,:]
    y = reshape(y, length(y))
    (x=Var(x), y=Var(y), dims_x=dims_x, training=dataset.training)
end

function readdata!(path::String, batchsize, worddict, training)
    data = Int[]
    for line in open(readlines,path)
        elems = collect(split(chomp(line)))
        ids = map(elems) do e
            get!(worddict, e, length(worddict)+1)
        end
        append!(data, ids)
        push!(data, 1) # <eos>
    end
    n = length(data) ÷ batchsize
    data = reshape(data[1:n*batchsize], n, batchsize)
    Dataset(data, training)
end
