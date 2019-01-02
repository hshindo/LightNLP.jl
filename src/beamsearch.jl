mutable struct Node{T}
    state::T
    score::Float64
    prev::Node{T}

    Node(state) = new(state, 0.0)
end

lessthan(x::Node{T}, y::Node{T}) where T = x.score > y.score

function toseq(finalstate::T) where T
    seq = T[]
    s = finalstate
    # while s != nothing
    while s.step > 1
        unshift!(seq, s)
        s = s.prev
    end
    unshift!(seq, s)
    seq
end

"""
* next: state -> score, state
"""
function beamsearch(initstate::T, beamsize::Int) where T
    chart = Vector{T}[]
    push!(chart, [initstate])

    k = 1
    while k <= length(chart)
        states = chart[k]
        length(states) > beamsize && sort!(states, lt=lessthan)
        for i = 1:min(beamsize,length(states))
            for (s,score) in next(states[i])
                while s.step > length(chart)
                    push!(chart, T[])
                end
                push!(chart[s.step], s)
            end
        end
        k += 1
    end
    sort!(chart[end], lt=lessthan)
    chart
end

function search(initstate, beamsize::Int)

end

public class BeamSearch[S] where S: IState[S]
    type Node = Node[S]
    BeamSize: int
    WeightVec: array[double]

    public Search(input: S): Node
        mutable kbest = List(BeamSize)
        Node(-1, input, 0.0, array[], null) |> kbest.Add
        while (kbest.Count > 0 && !kbest[0].St.IsFinal)
            kbest = Expand(kbest)
        kbest[0]

    Expand(source: List[Node]): List[Node]
        def temp = List()
        foreach (node in source)
            foreach (act in node.St.NextActs())
                def fs = node.St.GetFeats(act)
                def w = fs.Sum(f => WeightVec[f])
                (node, act, fs, w) |> temp.Add
        temp.Sort((x, y) => y.Weight.CompareTo(x.Weight))

        def dest = List(BeamSize)
        foreach ((node, act, fs, w) in temp)
            node.Next(act, fs, w) |> dest.Add
            when (dest.Count == BeamSize) break
        dest
