using StaticArrays
import Base: show, promote_rule

abstract type DecisionTree{N} end

struct Variable{N} <: DecisionTree{N}
    n::Integer
    Variable{N}(m) where N = (1 <= m <= N) ? new{N}(m) : error("Illegal construction")
end

struct Branch{N}
    conditions::Dict{Variable{N}, Float64}
    threshold::Float64
    iftrue::DecisionTree{N}
    iffalse::DecisionTree{N}
end

function Base.show(io::IO, v::Variable, indent = 0)
    print(io, "{", v.n, "}")
end

function Base.show(io::IO, b::Branch, indent = 0)
    print(io, " " ^ indent, "if ")
    join(io, ("$f × $v" for (v, f) in b.conditions), " + ")
    print(io, " ≤ ", b.threshold, "\n")
    print(io, " " ^ (indent + 2), "then ")
    show(io, b.iftrue, indent + 2)
    print(io, "\n", " " ^ (indent + 2), "else ")
    show(io, b.iffalse, indent + 2)
end

function decide{N}(value::StaticVector{N, Float64}, v::Variable{N})
    return v
end

function decide{N}(value::StaticVector{N, Float64}, b::Branch{N})
    if dot(normalize_conditions(b.conditions), value) <= b.threshold
        return decide(value, b.iftrue)
    else
        return decide(value, b.iffalse)
    end
end

function normalize_conditions{N}(conditions::AbstractVector{Tuple{Variable{N}, Float64}})
    result = @MVector zeros(N)
    for (v, f) in conditions
        result[v.n] += f
    end

    result
end

δ{T}(i, j, ::Type{T} = Float64) = ifelse(i == j, one(T), zero(T)) 

macro ε(n, i, T = Float64)
    vals = Expr(:vect, [δ(i, j, T) for j in 1:n]...)
    :(@SVector $vals)
end

