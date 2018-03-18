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

function randconditions{N}(rng::AbstractRNG, ::Type{DecisionTree{N}}; range = (-10, 10))
    a, b = (range[2] - range[1]), range[1]
    
    # Choose k ∈ {1..N} variables for each tree
    # Since these are dicts, duplicate variables are removed, which leads to about
    # about N - N/e elements in expectation (see https://math.stackexchange.com/a/41781/31127)
    # when N gets larger.
    k = rand(rng, 1:N)
    Dict(zip(rand(rng, Variable{N}, k), a * rand(rng, k) + b))
end

randconditions(treetype::Type{DecisionTree{N}} where {N}; range = (-10, 10)) =
    randconditions(Base.GLOBAL_RNG, treetype; range = range)
randconditions(treetype::Type{DecisionTree{N}} where {N}, n; range = (-10, 10)) =
    [randconditions(Base.GLOBAL_RNG, treetype; range = range) for i = 1:n]

function randtree_full{N}(rng::AbstractRNG, treetype::Type{DecisionTree{N}}, maxdepth;
                          range = (-10, 10))
    a, b = (range[2] - range[1]), range[1]

    if maxdepth == 1
        rand(rng, Variable{N}) 
    else
        conditions = randconditions(rng, treetype, range = range)
        threshold = a * rand(rng) + b
        true_children = randtree_full(rng, treetype, maxdepth - 1)
        false_children = randtree_full(rng, treetype, maxdepth - 1)
        Branch{N}(conditions, threshold, true_children, false_children)
    end
end

randtree_full(treetype::Type{DecisionTree{N}} where N, maxdepth) =
    randtree_full(Base.GLOBAL_RNG, treetype, maxdepth)
randtree_full(treetype::Type{DecisionTree{N}} where N, maxdepth, n) =
    [randtree_full(Base.GLOBAL_RNG, treetype, maxdepth) for i = 1:n]

# function randtree_grow(rng::AbstractRNG, ::Type{DecisionTree{N}}, maxdepth, n)
#     if maxdepth == 1
#         rand(rng, Variable{N}, n)
#     else
#         if rand(rng) < 0.5
#             rand(rng, Branch{N}, maxdepth - 1, n)
#         else
#             rand(rng, Variable{N}, n)
#         end
#     end
# end

# function randtree_ramped(rng::AbstractRNG, ::Type{DecisionTree{N}}, maxdepth::Int;
#                          full_proportion = 0.5)
#     @assert 0.0 ≤ full_proportion ≤ 1.0
#     if rand() < full_proportion
#         randtree_grow(rng, DecisionTree{N})
# end
