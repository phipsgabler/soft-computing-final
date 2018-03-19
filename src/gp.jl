using LearningStrategies
import LearningStrategies: update!

struct GPModel{N}
    population
    fitness
    fitness_values

    GPModel{N}(population::Vector{T} where T<:DecisionTree{N}, fitness) where N =
        new{N}(population, fitness, fitness.(population))
end



struct GPModelSolver <: LearningStrategy
    tournament_size::Int
    crossover_probability::Float64
    mutation_probability::Float64
end


function selection(parents, parent_fitnesses, k)
    candidates = rand(eachindex(parents, parent_fitnesses), k)
    parents[indmax(parent_fitnesses[candidates])]
end

function crossover(parent₁, parent₂, pc)
    rand([parent₁, parent₂])
end

function mutate(individual, pm)
    individual
end

function update!(model, s::GPModelSolver)
    parents = model.population
    fitness_values = model.fitness_values
    fitness = model.fitness
    k = s.tournament_size
    pc = s.crossover_probability
    pm = s.mutation_probability

    # breed
    children = similar(parents)
    for i in eachindex(children)
        p₁ = selection(parents, fitness_values, k)
        p₂ = selection(parents, fitness_values, k)
        child = crossover(p₁, p₂, pc)
        children[i] = mutate(child, pm)
    end

    # update
    parents .= children
    fitness_values .= fitness.(parents)
end



function rungp{N}(fitness, psize::Int, sampler::TreeSampler{N}, maxiter::Int)
    population = rand(sampler, psize)
    tracing_max = Tracer(DecisionTree{N}, (m, i) -> m.population[indmax(m.fitness_values)])
    
    learn!(GPModel{N}(population, fitness),
           strategy(GPModelSolver(7, 0.2, 0.8),
                    Verbose(MaxIter(maxiter)),
                    tracing_max))

    return population, collect(tracing_max)
end

# workspace(); include("SoftComputingFinal.jl"); using SoftComputingFinal
