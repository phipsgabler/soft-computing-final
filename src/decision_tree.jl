using StaticArrays
import Base: show, promote_rule

abstract type DecisionTree{N} end

struct Variable{N} <: DecisionTree{N}
    n::Int
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
    sorted_conditions = sort(collect(b.conditions), by = c -> c.first.n)
    
    print(io, " " ^ indent, "if ")
    join(io, ("$f × $v" for (v, f) in sorted_conditions), " + ")
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

δ{T}(i, j, ::Type{T} = Float64) = ifelse(i == j, one(T), zero(T)) 

@generated function basisvector{N, I}(::Type{Val{N}}, ::Type{Val{I}})
    ee = [δ(I, j) for j in 1:N]
    Expr(:call, :SVector, ee...)
end

onehot{N}(v::Variable{N}) = basisvector(Val{N}, Val{v.n})

function normalize_conditions{N}(conditions::Dict{Variable{N}, Float64})
    z = @SVector zeros(N)
    reduce((v, cond) -> v + onehot(cond[1]) * cond[2], z, conditions)
end



