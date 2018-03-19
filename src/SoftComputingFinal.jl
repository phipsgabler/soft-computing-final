module SoftComputingFinal

include("decision_trees.jl")
include("data.jl")

# see: https://github.com/bensadeghi/DecisionTree.jl

# EXPORTS
export DecisionTree,
    Variable,
    Branch,
    treedepth,
    treesize,
    decide,
    normalize_conditions,
    SplitSampler,
    RampedSplitSampler,
    BoltzmannSampler

end # module
