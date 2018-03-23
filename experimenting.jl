# addprocs(4)

# using DataFrames
using LearningStrategies
# @everywhere using SoftComputingFinal
using SoftComputingFinal

function testfitness()
    glass = load_glass();
    fitness = create_fitness(glass, Val{9}, Val{7})
    tree = rand(BoltzmannSampler{9, 7}(10, 20));
    fitness(tree)
end


function testgp(N)
    data = load_testdata()
    fitness, accuracy = create_fitness(data, Val{2}, Val{2}, depth_penalty = 0.1)

    tracer = Tracer(DecisionTree{2, 2}, (m, i) -> m.population[indmax(m.population_fitnesses)])
    # initializer = BoltzmannSampler{2, 2}(1, 10, (-2.0, 2.0))
    initializer = RampedSplitSampler{2, 2}(4, 0.5, 0.5, crange = (-2.0, 2.0))
    pop, trace = rungp(fitness, 20, initializer, N, tracer = tracer);

    max_trees = collect(trace)
    # display(accuracy.(max_trees))
    best_tree = select(max_trees, 1, by = fitness, rev = true)
    display(best_tree)
    display(accuracy(best_tree))
    Dict(:pop => pop, :poptrace => max_trees, :accuracytrace => accuracy.(max_trees))
end

testgp(20);

