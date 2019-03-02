export IntDict
export countfilter!

"""
    IntDict{T}

A dictionary to convert key::T into integer id.

## 👉 Example
```julia
dict = IntDict{String}()
push!(dict, "abc") == 1
push!(dict, "def") == 2
push!(dict, "abc") == 1

dict["abc"] == 1
getkey(dict, id1) == "abc"
```
"""
struct IntDict{T}
    key2id::Dict{T,Int}
    id2key::Vector{T}
    id2count::Vector{Int}
end
IntDict{T}() where T = IntDict(Dict{T,Int}(), T[], Int[])
function IntDict(keys::T...) where T
    d = IntDict{T}()
    for key in keys
        get!(d, key)
    end
    d
end

Base.count(dict::IntDict, id::Int) = dict.id2count[id]
Base.getkey(dict::IntDict, id::Int) = dict.id2key[id]
Base.getindex(dict::IntDict, key) = dict.key2id[key]
Base.get(dict::IntDict, key, default) = get(dict.key2id, key, default)
Base.length(d::IntDict) = length(d.key2id)

function Base.get!(dict::IntDict, key)
    if haskey(dict.key2id, key)
        id = dict.key2id[key]
        dict.id2count[id] += 1
    else
        id = length(dict.id2key) + 1
        dict.key2id[key] = id
        push!(dict.id2key, key)
        push!(dict.id2count, 1)
    end
    id
end

function Base.replace!(dict::IntDict, newid::Int; count::Int)
    for id = 1:length(dict)
        Base.count(dict,id) > count && push!(keys,dict[id])
        if Base.count(dict,id) <= count
            delete!(dict.key2id, dict[id])
            dict.id2key[id] = dict.id2key[newid]
            dict.id2count[id] = 0
        end
    end
end

function save(dict::IntDict)
    throw("Not implemented yet.")
end

function load(::Type{IntDict{T}}, path::String) where T
    throw("Not implemented yet.")
end
