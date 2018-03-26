using DataFrames
using CSV
using LearningStrategies: Tracer
using SoftComputingFinal
using ProgressMeter

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
best(model) = model.population[indmax(model.population_fitnesses)]


function evaluatedataset(name, N; folds = 10, repetitions = 3, pareto_sample = 50)
    popsize = 250

    load_data, nvars_t, nclasses_t = datasets[name]
    data = load_data()
    D = size(data)[1]
    shuffled = randperm(D)
    accuracies = DataFrame(generation = Int[],
                           accuracy = Float64[])
    sizes = DataFrame(size = Int[], fold = Int[], run = Int[])

    p = Progress(folds * repetitions, desc = "Runs: ")
    f = 1
    
    for (train_indices, val_indices) in zip(kfolds(D, folds)...)
        training_data = view(data, shuffled[train_indices])
        validation_data = view(data, shuffled[val_indices])
        
        fitness, _ = create_fitness(training_data, nvars_t, nclasses_t,
                                     depth_factor = 0.0,
                                     size_factor = 0.5)
        _, validation_accuracy = create_fitness(validation_data, nvars_t, nclasses_t,
                                                 depth_factor = 0.0,
                                                 size_factor = 0.5)

        for r = 1:repetitions
            tracer = Tracer(Float64, (m, i) -> validation_accuracy(best(m)))
            initializer = getsampler(nvars_t, nclasses_t)

            pop, trace = runssgp(fitness, popsize, initializer, N,
                                 max_depth = 30,
                                 tracer = tracer,
                                 verbose = false,
                                 debug = false)

            append!(accuracies, DataFrame(generation = 1:N, accuracy = collect(trace)))
            append!(sizes, DataFrame(size = treesize.(pop), fold = f, run = r))

            next!(p, showvalues = [(:fold, f)]) # update progress bar
        end
        
        f += 1
    end

    finish!(p)
    #println(STDERR) # newline after progress bar

    mean_accuracies = by(accuracies, :generation) do df
        x̄ = mean(df[:accuracy])
        σ̂ = std(df[:accuracy], mean = x̄)
        n = length(df[:accuracy])
        DataFrame(mean = x̄, ci = 1.96σ̂ / √n)
    end

    return mean_accuracies, sizes
end


function main()
    if length(ARGS) == 2
        dataset, N = ARGS[1], parse(Int, ARGS[2])
    elseif length(ARGS) == 1
        dataset, N = ARGS[1], 1000
    else
        error("Usage: $PROGRAM_FILE <dataset> [<generations>]")
    end

    mean_accuracies, sizes = evaluatedataset(dataset, N)
    # display(mean_accuracies)
    
    plt_acc = plot(mean_accuracies[:generation], mean_accuracies[:mean],
                   ribbon = mean_accuracies[:ci],
                   fillalpha = 0.2,
                   xlabel = "Generation",
                   ylabel = "Validation accuracy", ylims = (0, 1),
                   title = "Average validation accuracy\n and 95% confidence interval\n over time",
                   legend = :none)
    png(plt_acc, "$dataset-accuracies.png")

    plt_sizes = histogram(sizes[:size], normalize = true,
                          xlabel = "Size", ylabel = "Frequency",
                          title = "Size distribution in last generation",
                          legend = :none)
    png(plt_sizes, "$dataset-sizes.png")

    CSV.write("$dataset-accuracies.csv", mean_accuracies)
    CSV.write("$dataset-sizes.csv", sizes)
end


main()
