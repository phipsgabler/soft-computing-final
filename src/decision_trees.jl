using StaticArrays
import Base: show

"""Variable for a `DecisionTree{N, C}`."""
struct Variable{N}
    n::Int
    Variable{N}(m) where N = (1 <= m <= N) ? new{N}(m) : error("Illegal construction")
end

"""Decision tree for `N` features and `C` classes, consisting of `Classification`s and `Desicion`s."""
abstract type DecisionTree{N, C} end

struct Classification{N, C} <: DecisionTree{N, C}
    index::Int
    # label
    Classification{N, C}(m) where {N, C} =
        (1 <= m <= C) ? new{N, C}(m) : error("Illegal construction")
end

struct Decision{N, C} <: DecisionTree{N, C}
    conditions::Dict{Variable{N}, Float64}
    threshold::Float64
    iftrue::DecisionTree{N, C}
    iffalse::DecisionTree{N, C}
    
    Decision{N, C}(c, t, ift::DecisionTree{N, C}, iff::DecisionTree{N, C}) where {N, C} =
        new{N, C}(c, t, ift, iff)
    Decision(c, t, ift::DecisionTree{N, C}, iff::DecisionTree{N, C}) where {N, C} =
        new{N, C}(c, t, ift, iff)
end

function Base.show(io::IO, v::Variable, indent = 0)
    print(io, "x")
    join(io, map(d -> '\u2080' + d, digits(v.n)) |> reverse)
end

function Base.show(io::IO, c::Classification, indent = 0)
    print(io, "{", c.index, "}")
end

function Base.show(io::IO, d::Decision, indent = 0; digits = 2)
    print(io, "if ")
    if !isempty(d.conditions)
        sorted_conditions = sort(collect(d.conditions), by = c -> c.first.n)
        join(io, ("$(round(f, digits)) × $v" for (v, f) in sorted_conditions), " + ")
    else
        print(io, "0.0")
    end
    print(io, " ≤ ", round(d.threshold, digits), "\n")
    print(io, " " ^ (indent + 2), "then ")
    show(io, d.iftrue, indent + 2)
    print(io, "\n", " " ^ (indent + 2), "else ")
    show(io, d.iffalse, indent + 2)
end


treedepth(::Classification) = 1
treedepth(d::Decision) = 1 + max(treedepth(d.iftrue), treedepth(d.iffalse))

treesize(::Classification) = 1
treesize(d::Decision) = 1 + treesize(d.iftrue) + treesize(d.iffalse)


"""
    decide{N}(value::StaticVector, t::DecisionTree)

Evaluate the decision tree `t` on a data point, returning a `Variable` indicating the result.
"""
function decide(datapoint::AbstractVector{Float64}, d::Decision)
    combination = 0.0
    for (v, λ) in d.conditions
        combination += λ * datapoint[v.n]
    end
        
    if combination <= d.threshold
        return decide(datapoint, d.iftrue)
    else
        return decide(datapoint, d.iffalse)
    end
end

function decide(::AbstractVector{Float64}, c::Classification)
    return c
end



δ{T}(i, j, ::Type{T} = Float64) = ifelse(i == j, one(T), zero(T))

function basisvector_impl(n, i)
    ee = [δ(i, j) for j in 1:n]
    Expr(:call, :SVector, ee...)
end

macro basisvector(n, i)
    @assert n isa Integer
    @assert i isa Integer
    basisvector_impl(n, i)
end

@generated function basisvector{N, I}(::Type{Val{N}}, ::Type{Val{I}})
    basisvector_impl(N, I)
end

onehot{N}(v::Variable{N}) = basisvector(Val{N}, Val{v.n})

"""
    normalize_conditions{N}(conditions::Dict{Variable{N}, Float64})

Convert a condition dictionary into a vector suitable for linear decision.
"""
function normalize_conditions{N}(conditions::Dict{Variable{N}, Float64})
    z = @SVector zeros(N)
    reduce((v, cond) -> v + onehot(cond[1]) * cond[2], z, conditions)
end
