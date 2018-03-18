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


abstract type TreeSampler end

function randtree end

randtree(sampler::TreeSampler) = randtree(Base.GLOBAL_RNG, sampler)
randtree(sampler::TreeSampler, n) = [randtree(Base.GLOBAL_RNG, sampler) for _ = 1:n]


struct SplitSampler{N} <: TreeSampler
    maxdepth::Int
    split_probability::Float64
    crange::Tuple{Float64, Float64}
    
    SplitSampler{N}(m, r, c = (-10, 10)) where N =
        (0.0 ≤ r ≤ 1.0) ? new{N}(m, r, c) : error(r, " is not a probability!")
end

decreasedepth{N}(r::SplitSampler{N}) = SplitSampler{N}(r.maxdepth - 1, r.split_probability, r.crange)

function randtree{N}(rng::AbstractRNG, sampler::SplitSampler{N})
    maxdepth = sampler.maxdepth
    split_probability = sampler.split_probability
    crange = sampler.crange
    a, b = (crange[2] - crange[1]), crange[1]

    if maxdepth == 1 || rand(rng) ≥ split_probability
        rand(rng, Variable{N})
    else
        conditions = randconditions(rng, DecisionTree{N}, crange = sampler.crange)
        threshold = a * rand(rng) + b
        true_children = randtree(rng, decreasedepth(sampler))
        false_children = randtree(rng, decreasedepth(sampler))
        Branch{N}(conditions, threshold, true_children, false_children)
    end
end


struct RampedSplitSampler{N} <: TreeSampler
    maxdepth::Int
    split_probability::Float64
    rand_portion::Float64
    crange::Tuple{Float64, Float64}
    
    function RampedSplitSampler{N}(m, r, p, c = (-10, 10)) where N
        @assert (0.0 ≤ r ≤ 1.0) "$r is not a probability!"
        @assert (0.0 ≤ p ≤ 1.0) "$p is not a valid percentage!"
        new{N}(m, r, p, c)
        
    end
end

decreasedepth{N}(r::RampedSplitSampler{N}) =
    RampedSplitSampler{N}(r.maxdepth - 1, r.split_probability, r.rand_portion, r.crange)

function randtree{N}(rng::AbstractRNG, sampler::RampedSplitSampler{N})
    maxdepth = sampler.maxdepth
    split_probability = sampler.split_probability
    rand_portion = sampler.rand_portion
    crange = sampler.crange
    a, b = (crange[2] - crange[1]), crange[1]

    if rand(rng) ≥ rand_portion
        split_probability = 0.0
    end
    
    if maxdepth == 1 || rand(rng) ≤ split_probability
        rand(rng, Variable{N})
    else
        conditions = randconditions(rng, DecisionTree{N}, crange = sampler.crange)
        threshold = a * rand(rng) + b
        true_children = randtree(rng, decreasedepth(sampler))
        false_children = randtree(rng, decreasedepth(sampler))
        Branch{N}(conditions, threshold, true_children, false_children)
    end
end







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

struct BoltzmannSampler{N} <: TreeSampler
    minsize::Int
    maxsize::Int
    crange::Tuple{Float64, Float64}
    
    BoltzmannSampler{N}(m, n, c = (-10, 10)) where N = new{N}(m, n, c)
end

function boltzmann_ub{N}(rng::AbstractRNG, sampler::BoltzmannSampler{N}, cursize)
    minsize = sampler.minsize
    maxsize = sampler.maxsize
    crange = sampler.crange
    a, b = (crange[2] - crange[1]), crange[1]

    if cursize > maxsize
        return Nullable{DecisionTree{N}}(), cursize
    elseif rand(rng) ≤ 0.5
        return Nullable{DecisionTree{N}}(rand(rng, Variable{N})), cursize
    else
        true_children, cursize = boltzmann_ub(rng, sampler, cursize + 1)
        isnull(true_children) && @goto bound_exceeded

        false_children, cursize = boltzmann_ub(rng, sampler, cursize + 1)
        isnull(false_children) && @goto bound_exceeded

        conditions = randconditions(rng, DecisionTree{N}, crange = crange)
        threshold = a * rand(rng) + b
        t = Branch{N}(conditions, threshold, get(true_children), get(false_children))
        return Nullable{DecisionTree{N}}(t), cursize
        
        @label bound_exceeded
        return Nullable{DecisionTree{N}}(), cursize
    end
end

function randtree{N}(rng::AbstractRNG, sampler::BoltzmannSampler{N})
    candidate, size = boltzmann_ub(rng, sampler, 0)
    while isnull(candidate) || size < sampler.minsize
        candidate, size = boltzmann_ub(rng, sampler, 0)
    end

    return get(candidate)
end

