# using DataFrames
using LearningStrategies
using SoftComputingFinal

const ValType = Type{Val{T}} where T
const datasets = Dict{String, Tuple{Function, ValType, ValType}}(
    "testdata" => (load_testdata, Val{2}, Val{2}),
    "glass" => (load_glass, Val{9}, Val{7}),
    "ionosphere" => (load_ionosphere, Val{34}, Val{2}),
    "segmentation" => (load_segmentation, Val{19}, Val{7}))

getsampler(::Type{Val{M}}, ::Type{Val{N}}) where {M, N} =
    RampedSplitSampler{M, N}(6, 0.5, 0.5,
                             crange = (-10.0, 10.0),
                             maxvars = min(3, M))

function testdataset(name, N)
    popsize = 250

    load_data, nvars, nclasses = datasets[name]
    data = load_data()
    fitness, accuracy = create_fitness(data, nvars, nclasses,
                                       depth_factor = 0.0, size_factor = 0.0)

    tracer = Tracer(Float64, (m, i) -> accuracy(m.population[indmax(m.population_fitnesses)]))
    initializer = getsampler(nvars, nclasses)

    pop, trace = runssgp(fitness, popsize, initializer, N,
                         max_depth = 30,
                         tracer = tracer,
                         verbose = false,
                         debug = false);

    println(collect(trace))
end


function main()
    if length(ARGS) == 2
        testdataset(ARGS[1], ARGS[2])
    elseif length(ARGS) == 1
        testdataset(ARGS[1], 1000)
    else
        error("Usage: $PROGRAM_FILE <dataset> [<generations>]")
    end
end


main()
