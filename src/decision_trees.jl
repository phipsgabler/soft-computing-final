using StaticArrays
import Base: show

"""Decision tree for `N` variables, consisting of `Variable{N}`s and `Branch{N}`s."""
abstract type DecisionTree{N} end

struct Variable{N} <: DecisionTree{N}
    n::Int
    Variable{N}(m) where N = (1 <= m <= N) ? new{N}(m) : error("Illegal construction")
end

struct Branch{N} <: DecisionTree{N}
    conditions::Dict{Variable{N}, Float64}
    threshold::Float64
    iftrue::DecisionTree{N}
    iffalse::DecisionTree{N}
    
    Branch{N}(c, t, ift::DecisionTree{N}, iff::DecisionTree{N}) where N = new{N}(c, t, ift, iff)
    Branch(c, t, ift::DecisionTree{N}, iff::DecisionTree{N}) where N = new{N}(c, t, ift, iff)
end

function unicode_subscript(i)
    @assert 0 ≤ i ≤ 9
    '\u2080' + i
end

function Base.show(io::IO, v::Variable, indent = 0)
    print(io, "x", unicode_subscript(v.n))
end

function Base.show(io::IO, b::Branch, indent = 0; digits = 2)
    print(io, "if ")
    if !isempty(b.conditions)
        sorted_conditions = sort(collect(b.conditions), by = c -> c.first.n)
        join(io, ("$(round(f, digits)) × $v" for (v, f) in sorted_conditions), " + ")
    else
        print(io, "0.0")
    end
    print(io, " ≤ ", round(b.threshold, digits), "\n")
    print(io, " " ^ (indent + 2), "then ")
    show(io, b.iftrue, indent + 2)
    print(io, "\n", " " ^ (indent + 2), "else ")
    show(io, b.iffalse, indent + 2)
end


treedepth(::Variable) = 1
treedepth(b::Branch) = 1 + max(treedepth(b.iftrue), max(treedepth(b.iffalse)))

treesize(::Variable) = 1
treesize(b::Branch) = 1 + treesize(b.iftrue) + treesize(b.iffalse)


"""
    decide{N}(value::StaticVector, t::DecisionTree)

Evaluate the decision tree `t` on a data point, returning a `Variable` indicating the result.
"""
function decide(value::AbstractVector{Float64}, b::Branch)
    if dot(normalize_conditions(b.conditions), value) <= b.threshold
        return decide(value, b.iftrue)
    else
        return decide(value, b.iffalse)
    end
end

function decide(value::AbstractVector{Float64}, v::Variable)
    return v
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
