# Extracting these closures into types improves type stability a lot.
# Essentially, I have reinveted [zippers](https://wiki.haskell.org/Zipper) ;)

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
