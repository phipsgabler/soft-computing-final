module SoftComputingFinal

include("decision_trees.jl")
include("evaluation.jl")
include("data.jl")
include("tree_samplers.jl")
include("splitting.jl")
include("gp.jl")

# For comparison, see: https://github.com/bensadeghi/DecisionTree.jl

# EXPORTS
export
    # Basic tree representation etc.
    DecisionTree, Decision, Classification, Variable,
    treedepth, treesize, decide, normalize_conditions,
    randsplit, randchild,
    # Sampling
    SplitSampler, RampedSplitSampler, BoltzmannSampler,
    # Data stuff
    load_glass, load_ionosphere, load_segmentation, load_testdata, create_fitness,
    # Genetic programming
    rungp, runssgp

end # module
