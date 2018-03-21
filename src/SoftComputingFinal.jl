module SoftComputingFinal

include("decision_trees.jl")
include("evaluation.jl")
include("data.jl")
include("tree_samplers.jl")
include("splitting.jl")
include("gp.jl")

# see: https://github.com/bensadeghi/DecisionTree.jl

# EXPORTS
export
    # Basic tree representation etc.
    DecisionTree, Decision, Classification, Variable,
    treedepth, treesize, decide, normalize_conditions,
    randomsplit, randomchild,
    # Sampling
    SplitSampler, RampedSplitSampler, BoltzmannSampler,
    # Data stuff
    load_glass, load_ionosphere, load_image_segmentation, load_testdata, create_fitness,
    # Genetic programming
    rungp

end # module


# workspace(); include("SoftComputingFinal.jl"); using SoftComputingFinal
