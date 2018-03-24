# Extracting these closures into types improves type stability a lot.
# Essentially, I have reinveted [zippers](https://wiki.haskell.org/Zipper) ;)

abstract type Context{N, C} end


struct LeftContext{N, C} <: Context{N, C}
    conditions::Dict{Variable{N}, Float64}
    threshold::Float64
    right::DecisionTree{N, C}
    parent::Context{N, C}
end

function (context::LeftContext{N, C})(t::DecisionTree{N, C}) where {N, C}
    context.parent(Decision{N, C}(context.conditions, context.threshold, t, context.right))
end


struct RightContext{N, C} <: Context{N, C}
    conditions::Dict{Variable{N}, Float64}
    threshold::Float64
    left::DecisionTree{N, C}
    parent::Context{N, C}
end

function (context::RightContext{N, C})(t::DecisionTree{N, C}) where {N, C}
    context.parent(Decision{N, C}(context.conditions, context.threshold, context.left, t))
end


struct NoContext{N, C} <: Context{N, C} end
    
function (context::NoContext{N, C})(t::DecisionTree{N, C}) where {N, C}
    t
end


struct Cont{N, C}
    context::Context{N, C}
    chunk::DecisionTree{N, C}
end

function (continuation::Cont{N, C})(k::Function) where {N, C}
    newchunk = k(continuation.chunk)::DecisionTree{N, C}
    tree = continuation.context(newchunk)::DecisionTree{N, C}
    return tree, continuation.chunk
end


function randsplit_impl{N, C}(rng::AbstractRNG, t::Classification{N, C}, n::Int,
                                parent_context::Context{N, C}, continuation::Cont{N, C})
    n += 1
    if rand() ≤ 1/n
        return Cont{N, C}(parent_context, t), n
    else    
        return continuation, n
    end
end

function randsplit_impl{N, C}(rng::AbstractRNG, t::Decision{N, C}, n,
                                parent_context::Context{N, C}, continuation::Cont{N, C})
    c1, n = randsplit_impl(rng, t.iftrue, n,
                             LeftContext{N, C}(t.conditions, t.threshold, t.iffalse, parent_context),
                             continuation)
    c2, n = randsplit_impl(rng, t.iffalse, n,
                             RightContext{N, C}(t.conditions, t.threshold, t.iftrue, parent_context),
                             c1)
    
    n += 1
    if rand() ≤ 1/n
        return Cont{N, C}(parent_context, t), n
    else    
        return c2, n
    end
end

"""
    randsplit(action,[ rng,] t::DecisionTree)

Select uniformly at random a point to split `t`, then replace the subtree `s` by `action(s)`. 
Returns both the new tree and the replaced subtree.
"""
function randsplit{N, C}(action, rng::AbstractRNG, t::DecisionTree{N, C})
    # Uses reservoir samling of continuations, see: https://stackoverflow.com/a/3272490/1346276.
    default_context = NoContext{N, C}()
    default_continuation = Cont{N, C}(default_context, t)
    cont, _ = randsplit_impl(rng, t, 0, default_context, default_continuation)
    cont(action)
end

randsplit{N, C}(action, t::DecisionTree{N, C}) = randsplit(action, Base.GLOBAL_RNG, t)

"""
    randchild([rng,] t::DecisionTree)

Select uniformly at random a subtree of `t`.
"""
randchild{N, C}(rng::AbstractRNG, t::DecisionTree{N, C}) = randsplit(identity, rng, t)[2]
randchild{N, C}(t::DecisionTree{N, C}) = randchild(Base.GLOBAL_RNG, t)
