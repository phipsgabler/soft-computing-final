# see: https://stackoverflow.com/a/3272490/1346276

abstract type Tree{N} end

struct Leave{N} <: Tree{N}
    label::Int
end

struct Branch{N} <: Tree{N}
    left::Tree{N}
    right::Tree{N}
end

struct Cont{N}
    context::Function
    chunk::Tree{N}
end

function (continuation::Cont{N})(k::Function) where N
    newchunk = k(continuation.chunk)::Tree{N}
    tree = continuation.context(newchunk)::Tree{N}
    return tree, continuation.chunk
end

function randomsplit_impl{N}(t::Leave{N}, n::Int, context::Function, continuation::Cont{N})
    n += 1
    if rand() ≤ 1/n
        return Cont{N}(context, t), n
    else
        return continuation, n
    end
end

function randomsplit_impl{N}(t::Branch{N}, n::Int, context::Function, continuation::Cont{N})
    c1, n = randomsplit_impl(t.left, n, b -> context(Branch(b, t.right)), continuation)
    c2, n = randomsplit_impl(t.right, n, b -> context(Branch(t.left, b)), c1)

    n += 1
    if rand() ≤ 1/n
        return Cont{N}(context, t), n
    else    
        return c2, n
    end
end

function randomsplit{N}(action, t::Tree{N})
    cont, _ = randomsplit_impl(t, 0, identity, Cont{N}(identity, t))
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
