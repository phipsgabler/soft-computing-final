# see: https://stackoverflow.com/a/3272490/1346276

abstract type Tree{N} end

struct Leave{N} <: Tree{N}
    label::Int
end

struct Branch{N} <: Tree{N}
    left::Tree{N}
    right::Tree{N}
end


abstract type Context{N} end


struct LeftContext{N} <: Context{N}
    right::Tree{N}
    parent::Context{N}
end

function (context::LeftContext{N})(t::Tree{N}) where N
    context.parent(Branch(t, context.right))
end


struct RightContext{N} <: Context{N}
    left::Tree{N}
    parent::Context{N}
end

function (context::RightContext{N})(t::Tree{N}) where N
    context.parent(Branch(context.left, t))
end


struct NoContext{N} <: Context{N} end
    
function (context::NoContext{N})(t::Tree{N}) where N
    t
end


struct Cont{N}
    context::Context{N}
    chunk::Tree{N}
end

function (continuation::Cont{N})(k::Function) where N
    newchunk = k(continuation.chunk)::Tree{N}
    tree = continuation.context(newchunk)::Tree{N}
    return tree, continuation.chunk
end

function randomsplit_impl{N}(t::Leave{N}, n::Int, context::Context{N}, continuation::Cont{N})
    n += 1
    if rand() ≤ 1/n
        return Cont{N}(context, t), n
    else
        return continuation, n
    end
end

function randomsplit_impl{N}(t::Branch{N}, n::Int, context::Context{N}, continuation::Cont{N})
    c1, n = randomsplit_impl(t.left, n, LeftContext{N}(t.right, context), continuation)
    c2, n = randomsplit_impl(t.right, n, RightContext{N}(t.left, context), c1)
    
    n += 1
    if rand() ≤ 1/n
        return Cont{N}(context, t), n
    else    
        return c2, n
    end
end

function randomsplit{N}(action, t::Tree{N})
    default_context = NoContext{N}()
    cont, _ = randomsplit_impl(t, 0, default_context, Cont{N}(default_context, t))
    cont(action)
end
randomchild{N}(t::Tree{N}) = randomsplit(identity, t)[2]


const t = Branch{1}(Branch{1}(Leave{1}(1), Leave{1}(2)), Leave{1}(4))
const t2 = Branch{1}(Leave{1}(1), Leave{1}(2))


# to test for a uniform distribution of splits, we use some hackish hashing by summation:
sumtree(t::Leave) = t.label
sumtree(t::Branch) = sumtree(t.left) + sumtree(t.right)

# function testvalues(N)
#     zd = [sumtree(randomsplit(t, x -> (Leave(64), nothing))[1]) for _ in 1:N]
#     [(i, sum(zd .== i)) for i in unique(zd)]
# end

# k -> Branch(k(Leave()), Leave())
# k -> Branch(Leave(), k(Leave()))
# k -> k(Branch(Leave(), Leave()))
