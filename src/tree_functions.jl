# Generate_Tree( max_depth, generation_method )
# begin
# if max_depth = 1 then
# set the root of the tree to a randomly selected terminal;
# else if generation_method = full then
# set the root of the tree to a randomly selected non-terminal;
# else
# set the root to a randomly selected element which is either
# terminal or non-terminal;
# for each argument of the root, generate a subtree with the call
# Generate_Tree( max_depth - 1, generation_method );
# end;

import Base: rand


Base.rand{N}(rng::AbstractRNG, ::Type{Variable{N}}) = Variable{N}(rand(rng, 1:N))

function randconditions{N}(rng::AbstractRNG, ::Type{DecisionTree{N}}; crange = (-10, 10))
    a, b = (crange[2] - crange[1]), crange[1]
    
    # Choose k ∈ {1..N} variables with replacement at random.  Since these are dicts, duplicate
    # entries are removed.
    k = rand(rng, 1:N)
    Dict{Variable{N}, Float64}(zip(rand(rng, Variable{N}, k), a * rand(rng, k) + b))
end

randconditions(treetype::Type{DecisionTree{N}} where {N}; crange = (-10, 10)) =
    randconditions(Base.GLOBAL_RNG, treetype; crange = crange)
randconditions(treetype::Type{DecisionTree{N}} where {N}, n; crange = (-10, 10)) =
    [randconditions(Base.GLOBAL_RNG, treetype; crange = crange) for i = 1:n]

function randtree{N}(rng::AbstractRNG, treetype::Type{DecisionTree{N}}, maxdepth, ramp_probability;
                          crange = (-10, 10))
    a, b = (crange[2] - crange[1]), crange[1]

    if maxdepth == 1 || rand(rng) ≤ ramp_probability
        rand(rng, Variable{N})
    else
        conditions = randconditions(rng, treetype, crange = crange)
        threshold = a * rand(rng) + b
        true_children = randtree(rng, treetype, maxdepth - 1, ramp_probability)
        false_children = randtree(rng, treetype, maxdepth - 1, ramp_probability)
        Branch{N}(conditions, threshold, true_children, false_children)
    end
end

randtree(treetype::Type{DecisionTree{N}} where N, maxdepth, ramp_probability; crange = (-10, 10)) =
    randtree(Base.GLOBAL_RNG, treetype, maxdepth, ramp_probability, crange = crange)
randtree(treetype::Type{DecisionTree{N}} where N, maxdepth, ramp_probability, n;
         crange = (-10, 10)) =
    [randtree(Base.GLOBAL_RNG, treetype, maxdepth, ramp_probability, crange = crange) for i = 1:n]

# function randtree_ramped(rng::AbstractRNG, ::Type{DecisionTree{N}}, maxdepth::Int;
#                          full_proportion = 0.5)
#     @assert 0.0 ≤ full_proportion ≤ 1.0
#     randtree(rng, DecisionTree{N})
# end







# Alternative: using Boltzmann sampling to generate trees of expected size. 
# See: https://byorgey.wordpress.com/2013/04/25/random-binary-trees-with-a-size-limited-critical-boltzmann-sampler-2/

# (Most of the) original Haskell code:
# genTreeUB :: GenM Tree
# genTreeUB = do
#   r <- getRandom
#   atom
#   if r <= (1/2 :: Double)
#     then return Leaf
#     else Branch <$> genTreeUB <*> genTreeUB
# 
# atom :: GenM ()
# atom = do
#   (_, maxSize) <- ask
#   curSize <- get
#   when (curSize >= maxSize) mzero
#   put (curSize + 1)

# genTreeLB :: GenM Tree
# genTreeLB = do
#   put 0
#   t <- genTreeUB
#   tSize <- get
#   (minSize, _) <- ask
#   guard $ tSize >= minSize
#   return t

# genTree :: GenM Tree
# genTree = genTreeLB `mplus` genTree


function boltzmann_ub{N}(rng::AbstractRNG, treetype::Type{DecisionTree{N}},
                         minsize, maxsize, cursize, crange)
    a, b = (crange[2] - crange[1]), crange[1]

    if cursize > maxsize
        return Nullable{DecisionTree{N}}(), cursize
    elseif rand(rng) ≤ 0.5
        return Nullable{DecisionTree{N}}(rand(rng, Variable{N})), cursize
    else
        true_children, cursize = boltzmann_ub(rng, treetype, minsize, maxsize, cursize + 1, crange)
        isnull(true_children) && @goto bound_exceeded

        false_children, cursize = boltzmann_ub(rng, treetype, minsize, maxsize, cursize + 1, crange)
        isnull(false_children) && @goto bound_exceeded

        conditions = randconditions(rng, treetype, crange = crange)
        threshold = a * rand(rng) + b
        t = Branch{N}(conditions, threshold, get(true_children), get(false_children))
        return Nullable{DecisionTree{N}}(t), cursize
        
        @label bound_exceeded
        return Nullable{DecisionTree{N}}(), cursize
    end
end

function randtree_boltzmann{N}(rng::AbstractRNG, treetype::Type{DecisionTree{N}},
                               minsize, maxsize; crange = (-10, 10))
    candidate, size = boltzmann_ub(rng, treetype, minsize, maxsize, 0, crange)
    while isnull(candidate) || size < minsize
        candidate, size = boltzmann_ub(rng, treetype, minsize, maxsize, 0, crange)
    end

    return get(candidate)
end

function randtree_boltzmann{N}(treetype::Type{DecisionTree{N}}, minsize, maxsize; crange = (-10, 10))
    randtree_boltzmann(Base.GLOBAL_RNG, treetype, minsize, maxsize, crange = crange)
end

function randtree_boltzmann{N}(treetype::Type{DecisionTree{N}}, minsize, maxsize, n;
                               crange = (-10, 10))
    [randtree_boltzmann(Base.GLOBAL_RNG, treetype, minsize, maxsize, crange = crange) for i = 1:n]
end

