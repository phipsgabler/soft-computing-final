using LearningStrategies
import LearningStrategies: update!

struct GPModel{N}
    population::Vector{DecisionTree{N}}
    fitness
    fitness_values::Vector{Float64}
end

GPModel(population, fitness) = GPModel(population, fitness, fitness.(population))

struct GPModelSolver <: LearningStrategy end

function update!(model, s::GPModelSolver)
    population = model.population
    fitness_values = model.fitness_values
    fitness = model.fitness

    # selection

    # crossover
    
    # mutation
    for i in eachindex(population)
        
    end

    # fitness update
    fitness_values .= fitness.(population)
end



function rungp{N}(fitness, psize::Int, sampler::TreeSampler{N}, maxiter::Int)
    population = rand(sampler, psize)
    tracing_max = Tracer(DecisionTree{N}, (m, i) -> m.population[indmax(m.fitness_values)])
    
    learn!(GPModel(population, fitness),
           strategy(GPModelSolver(),
                    Verbose(MaxIter(maxiter)),
                    tracing_max))

    return population, collect(tracing_max)
end

