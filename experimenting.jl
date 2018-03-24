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
    popsize = 250
    
    data = load_testdata()
    fitness, accuracy = create_fitness(data, Val{2}, Val{2},
                                       depth_penalty = 2.0, size_penalty = 0.5)

    # tracer = Tracer(DecisionTree{2, 2}, (m, i) -> m.population[indmax(m.population_fitnesses)])
    tracer = Tracer(Float64, (m, i) -> accuracy(m.population[indmax(m.population_fitnesses)]))

    # initializer = BoltzmannSampler{2, 2}(1, 10, (-2.0, 2.0))
    initializer = RampedSplitSampler{2, 2}(3, 0.5, 0.5, crange = (-2.0, 2.0))
    pop, trace = runssgp(fitness, popsize, initializer, N, tracer = tracer, verbose = false)

    println(collect(trace))
    
    # max_trees = collect(trace)
    # best_tree = select(max_trees, 1, by = fitness, rev = true)
    # println(best_tree)
    # println(accuracy(best_tree))
    # println(accuracy.(max_trees))
    # Dict(:pop => pop, :poptrace => max_trees, :accuracytrace => accuracy.(max_trees))
end

# testgp(100)

function testglass(N)
    popsize = 250
    
    data = load_glass()
    fitness, accuracy = create_fitness(data, Val{9}, Val{7},
                                       depth_penalty = 0.0, size_penalty = 0.0)
    println("Loaded data, created fitness function")

    tracer = Tracer(Float64, (m, i) -> accuracy(m.population[indmax(m.population_fitnesses)]))
    # initializer = RampedSplitSampler{9, 7}(12, 0.5, 0.5,
    #                                        crange = (-10.0, 10.0),
    #                                        maxvars = 3)
    initializer = BoltzmannSampler{9, 7}(1, 100,
                                         crange = (-10.0, 10.0),
                                         maxvars = 3)

    pop, trace = runssgp(fitness, popsize, initializer, N,
                         max_depth = 25,
                         tracer = tracer,
                         verbose = false,
                         debug = true)

    println(collect(trace))
end

testglass(1000)
