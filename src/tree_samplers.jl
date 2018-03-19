import Base: rand


Base.rand{N}(rng::AbstractRNG, ::Type{Variable{N}}) = Variable{N}(rand(rng, 1:N))


"""
    randconditions{N}(rng::AbstractRNG, ::Type{DecisionTree{N}}, crange)
    
Generate a random condition dictionary for a `DecisionTree{N}`, consisting of 1--N `Variable{N}`s
and their factors.
"""
function randconditions{N}(rng::AbstractRNG, ::Type{DecisionTree{N}}, crange)
    a, b = (crange[2] - crange[1]), crange[1]
    
    # Choose k ∈ {1..N} variables with replacement at random.  Since these are dicts, duplicate
    # entries are removed.
    k = rand(rng, 1:N)
    Dict{Variable{N}, Float64}(zip(rand(rng, Variable{N}, k), a * rand(rng, k) + b))
end

"""
    randconditions{N}(::Type{DecisionTree{N}} [, n];crange)

Generate one or `n` random condition dictionary for a `DecisionTree{N}`, consisting of 1--N
`Variable{N}`s and their factors.
"""
randconditions(treetype::Type{DecisionTree{N}} where {N}; crange = (-10, 10)) =
    randconditions(Base.GLOBAL_RNG, treetype; crange = crange)
randconditions(treetype::Type{DecisionTree{N}} where {N}, n; crange = (-10, 10)) =
    [randconditions(Base.GLOBAL_RNG, treetype; crange = crange) for i = 1:n]


abstract type TreeSampler{N} end

Base.rand(sampler::TreeSampler) = rand(Base.GLOBAL_RNG, sampler)
Base.rand(rng::AbstractRNG, sampler::TreeSampler, n) = [rand(rng, sampler) for _ = 1:n]
Base.rand(sampler::TreeSampler, n) = rand(Base.GLOBAL_RNG, sampler, n)


struct SplitSampler{N} <: TreeSampler{N}
    maxdepth::Int
    split_probability::Float64
    crange::Tuple{Float64, Float64}

    """
        SplitSampler{N}(m, s [, c = (-10, 10)])

    Configuration for sampling `DecisionTree{N}`s with maximum depth `m`, and probability of 
    branching `s` (1.0 corresponds to "full sampling", 0.5 to "grow sampling".)  `c` is the 
    range from which literals are sampled.
    """
    SplitSampler{N}(m, s, c = (-10, 10)) where N =
        (0.0 ≤ s ≤ 1.0) ? new{N}(m, s, c) : error(s, " is not a probability!")
end

decreasedepth{N}(r::SplitSampler{N}) = SplitSampler{N}(r.maxdepth - 1, r.split_probability, r.crange)

function Base.rand{N}(rng::AbstractRNG, sampler::SplitSampler{N})
    maxdepth = sampler.maxdepth
    split_probability = sampler.split_probability
    crange = sampler.crange
    a, b = (crange[2] - crange[1]), crange[1]

    if maxdepth == 1 || rand(rng) ≥ split_probability
        rand(rng, Variable{N})
    else
        conditions = randconditions(rng, DecisionTree{N}, crange)
        threshold = a * rand(rng) + b
        true_children = rand(rng, decreasedepth(sampler))
        false_children = rand(rng, decreasedepth(sampler))
        Branch{N}(conditions, threshold, true_children, false_children)
    end
end


struct RampedSplitSampler{N} <: TreeSampler{N}
    maxdepth::Int
    split_probability::Float64
    rand_portion::Float64
    crange::Tuple{Float64, Float64}

    """
        RampedSplitSampler{N}(m, s, p [, c = (-10, 10)])

    Configuration for sampling `DecisionTree{N}`s with maximum depth `m`, and probability of 
    branching `s`.  Ramped sampling means that with probability `rand_portion`, we use grow sampling
    with `s`, and otherwise full sampling (split probability 1.0). `c` is the range from which
    literals are sampled.
    """
    function RampedSplitSampler{N}(m, s, p, c = (-10, 10)) where N
        @assert (0.0 ≤ s ≤ 1.0) "$s is not a probability!"
        @assert (0.0 ≤ p ≤ 1.0) "$p is not a valid percentage!"
        new{N}(m, s, p, c)
        
    end
end

decreasedepth{N}(r::RampedSplitSampler{N}) =
    RampedSplitSampler{N}(r.maxdepth - 1, r.split_probability, r.rand_portion, r.crange)

function Base.rand{N}(rng::AbstractRNG, sampler::RampedSplitSampler{N})
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
        conditions = randconditions(rng, DecisionTree{N}, crange)
        threshold = a * rand(rng) + b
        true_children = rand(rng, decreasedepth(sampler))
        false_children = rand(rng, decreasedepth(sampler))
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

struct BoltzmannSampler{N} <: TreeSampler{N}
    minsize::Int
    maxsize::Int
    crange::Tuple{Float64, Float64}

    """
        BoltzmannSampler{N}(m, n, [, c = (-10, 10)])

    Configuration for Boltzmann sampling `DecisionTree{N}`s with minimum size `m` and maximum size
    `n`.  `c` is the range from which literals are sampled.
    """
    BoltzmannSampler{N}(m, n, c = (-10, 10)) where N = new{N}(m, n, c)
end

function boltzmann_ub{N}(rng::AbstractRNG, sampler::BoltzmannSampler{N}, cursize)
    minsize = sampler.minsize
    maxsize = sampler.maxsize
    crange = sampler.crange
    a, b = (crange[2] - crange[1]), crange[1]

    if cursize > maxsize        # maximum bound exceeded
        return Nullable{DecisionTree{N}}(), cursize
    elseif rand(rng) ≤ 0.5      # generate leaf
        return Nullable{DecisionTree{N}}(rand(rng, Variable{N})), cursize
    else                        # try generating a branch -- if it's size is small enough
        # try generating children, or immediately fail
        true_children, cursize = boltzmann_ub(rng, sampler, cursize + 1)
        isnull(true_children) && return Nullable{DecisionTree{N}}(), cursize

        false_children, cursize = boltzmann_ub(rng, sampler, cursize + 1)
        isnull(false_children) && return Nullable{DecisionTree{N}}(), cursize

        # `cursize` is now the size of the whole thing, and less than maxsize
        conditions = randconditions(rng, DecisionTree{N}, crange)
        threshold = a * rand(rng) + b
        result = Branch{N}(conditions, threshold, get(true_children), get(false_children))
        return Nullable{DecisionTree{N}}(result), cursize        
    end
end

function Base.rand{N}(rng::AbstractRNG, sampler::BoltzmannSampler{N})
    candidate, size = boltzmann_ub(rng, sampler, 0)
    while isnull(candidate) || size < sampler.minsize
        candidate, size = boltzmann_ub(rng, sampler, 0)
    end

    return get(candidate)
end
