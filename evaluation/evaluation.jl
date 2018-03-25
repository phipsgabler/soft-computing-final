using DataFrames
using LearningStrategies
using SoftComputingFinal

using Plots
pyplot()


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

kbest(model, k) = view(model.population, selectperm(model.population_fitnesses, 1:k))


function evaluatedataset(name, N; folds = 10, repetitions = 3, pareto_sample = 50)
    popsize = 250

    load_data, nvars_t, nclasses_t = datasets[name]
    data = load_data()
    D = size(data)[1]
    shuffled = randperm(D)
    errortraces = DataFrame(generation = Int[],
                           error = Float64[])

    for (train_indices, val_indices) in kfolds(D, folds)
        training_data = view(data, shuffled[train_indices])
        validation_data = view(data, shuffled[val_indices])
        
        fitness, xa = create_fitness(training_data, nvars_t, nclasses_t,
                                     depth_factor = 0.0,
                                     size_factor = 0.0)
        xb, validation_accuracy = create_fitness(validation_data, nvars_t, nclasses_t,
                                                 depth_factor = 0.0,
                                                 size_factor = 0.0)

        for r = 1:repetitions
            tracer = Tracer(Float64, (m, i) -> mean(validation_accuracy.(kbest(m, pareto_sample))))
            initializer = getsampler(nvars_t, nclasses_t)

            pop, trace = runssgp(fitness, popsize, initializer, N,
                                 max_depth = 30,
                                 tracer = tracer,
                                 verbose = false,
                                 debug = false)

            append!(errortraces, DataFrame(generation = 1:N, error = collect(trace)))
        end
    end

    return by(errortraces, :generation) do df
        x̄ = mean(df[:error])
        σ̂ = std(df[:error], mean = x̄)
        n = length(df[:error])
        DataFrame(mean = x̄,
                  ci_l = x̄ - 1.96σ̂ / √n,
                  ci_u = x̄ + 1.96σ̂ / √n)
    end
end


function main()
    if length(ARGS) == 2
        dataset, N = ARGS[1], parse(Int, ARGS[2])
    elseif length(ARGS) == 1
        dataset, N = ARGS[1], 1000
    else
        error("Usage: $PROGRAM_FILE <dataset> [<generations>]")
    end

    results = evaluatedataset(dataset, N)
    # display(results)
    
    plt = plot(results[:generation], results[:mean],
               ribbon = (results[:ci_l], results[:ci_u]),
               fillalpha = 0.2,
               xlabel = "Generation",
               ylabel = "Validation accuracy", ylims = (0, 1),
               title = "Average validation accuracy\n and 95% confidence interval\n over time",
               legend = :none)
    png(plt, "$dataset.png")
end


main()
