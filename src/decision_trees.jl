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
function decide{N}(value::StaticVector{N, Float64}, b::Branch{N})
    if dot(normalize_conditions(b.conditions), value) <= b.threshold
        return decide(value, b.iftrue)
    else
        return decide(value, b.iffalse)
    end
end

function decide{N}(value::StaticVector{N, Float64}, v::Variable{N})
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


# Extracting this into a type improves type stability a lot.
abstract type Context{N} end


struct LeftContext{N} <: Context{N}
    conditions::Dict{Variable{N}, Float64}
    threshold::Float64
    right::DecisionTree{N}
    parent::Context{N}
end

function (context::LeftContext{N})(t::DecisionTree{N}) where N
    context.parent(Branch{N}(context.conditions, context.threshold, t, context.right))
end


struct RightContext{N} <: Context{N}
    conditions::Dict{Variable{N}, Float64}
    threshold::Float64
    left::DecisionTree{N}
    parent::Context{N}
end

function (context::RightContext{N})(t::DecisionTree{N}) where N
    context.parent(Branch{N}(context.conditions, context.threshold, context.left, t))
end


struct NoContext{N} <: Context{N} end
    
function (context::NoContext{N})(t::DecisionTree{N}) where N
    t
end


struct Cont{N}
    context::Context{N}
    chunk::DecisionTree{N}
end

function (continuation::Cont{N})(k::Function) where N
    newchunk = k(continuation.chunk)::DecisionTree{N}
    tree = continuation.context(newchunk)::DecisionTree{N}
    return tree, continuation.chunk
end


function randomsplit_impl{N}(rng::AbstractRNG, t::Variable{N}, n::Int,
                             parent_context::Context{N}, continuation::Cont{N})
    n += 1
    if rand() ≤ 1/n
        return Cont{N}(parent_context, t), n
    else    
        return continuation, n
    end
end

function randomsplit_impl{N}(rng::AbstractRNG, t::Branch{N}, n,
                             parent_context::Context{N}, continuation::Cont{N})
    c1, n = randomsplit_impl(rng, t.iftrue, n,
                             LeftContext{N}(t.conditions, t.threshold, t.iffalse, parent_context),
                             continuation)
    c2, n = randomsplit_impl(rng, t.iffalse, n,
                             RightContext{N}(t.conditions, t.threshold, t.iftrue, parent_context),
                             c1)
    
    n += 1
    if rand() ≤ 1/n
        return Cont{N}(parent_context, t), n
    else    
        return c2, n
    end
end

"""
    randomsplit(action, t::DecisionTree)

Select uniformly at random a point to split `t`, then replace the subtree `s` by `action(s)`. 
Returns both the new tree and the replaced subtree.
"""
function randomsplit{N}(action, rng::AbstractRNG, t::DecisionTree{N})
    # Uses reservoir samling of continuations, see: https://stackoverflow.com/a/3272490/1346276.
    default_context = NoContext{N}()
    default_continuation = Cont{N}(default_context, t)
    cont, _ = randomsplit_impl(rng, t, 0, default_context, default_continuation)
    cont(action)
end

randomsplit{N}(action, t::DecisionTree{N}) = randomsplit(action, Base.GLOBAL_RNG, t)

"""
    randomchild(t::DecisionTree)

Select uniformly at random a subtree of `t`.
"""
randomchild{N}(t::DecisionTree{N}) = randomsplit(identity, t)[2]
