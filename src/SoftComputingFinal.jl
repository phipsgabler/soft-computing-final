module SoftComputingFinal

include("decision_trees.jl")
include("data.jl")
include("tree_samplers.jl")
include("gp.jl")

# see: https://github.com/bensadeghi/DecisionTree.jl

# EXPORTS
export
    # Basic tree representation etc.
    DecisionTree, Variable, Branch, treedepth, treesize, decide, normalize_conditions,
    randomsplit,
    # Sampling
    SplitSampler, RampedSplitSampler, BoltzmannSampler,
    # Genetic programming
    rungp

end # module


# workspace(); include("SoftComputingFinal.jl"); using SoftComputingFinal
