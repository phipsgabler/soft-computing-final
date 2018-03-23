using LearningStrategies
import LearningStrategies: update!

mutable struct GPModel{N, C}
    fitness::Function
    population::Vector{DecisionTree{N, C}}
    cache::Vector{DecisionTree{N, C}}
    population_fitnesses::Vector{Float64}
    cache_fitnesses::Vector{Float64}
end

function GPModel(fitness, population)
    fitnesses = fitness.(population)
    GPModel(fitness, population, similar(population), fitnesses, similar(fitnesses))
end


struct GPModelSolver{N, C} <: LearningStrategy
    tournament_size::Int
    mutation_probability::Float64
    mutation_sampler::TreeSampler{N, C}
end


function selection(parents, parent_fitnesses, k)
    candidates = rand(eachindex(parents, parent_fitnesses), k)
    parents[indmax(parent_fitnesses[candidates])]
end

function crossover(parent₁, parent₂)
    # choosing a random `chunk` from `parent₂` and splice it somewhere into `parent₁`
    newtree, _ = randsplit(parent₁) do _
        randchild(parent₂)
    end

    return newtree
end

function mutate(individual, pₘ, mutation_sampler)
    if rand() ≤ pₘ
        newtree, chunk = randsplit(individual) do _
            # ignore chunk and replace by random stuff
            rand(mutation_sampler)
        end

        return newtree
    else
        return individual
    end
end

function update!(model, s::GPModelSolver)
    parents = model.population
    children = model.cache
    parent_fitnesses = model.population_fitnesses
    child_fitnesses = model.cache_fitnesses
    fitness = model.fitness
    
    k = s.tournament_size
    pₘ = s.mutation_probability
    ms = s.mutation_sampler

    # breed
    for i in eachindex(children, child_fitnesses)
        p₁ = selection(parents, parent_fitnesses, k)
        p₂ = selection(parents, parent_fitnesses, k)
        child = crossover(p₁, p₂)
        children[i] = mutate(child, pₘ, ms)
        child_fitnesses[i] = fitness(child)
    end

    # update: swap actual values with caches
    model.population = children
    model.population_fitnesses = child_fitnesses
    model.cache = parents
    model.cache_fitnesses = parent_fitnesses
end


function rungp{N, C}(fitness, psize::Int, sampler::TreeSampler{N, C}, maxiter::Int;
                     tracer = Tracer(Void, (m, i) -> nothing, typemax(Int)),
                     tournament_size = 7, mutation_rate = 0.5)
    initial_population = rand(sampler, psize)
    model = GPModel(float ∘ fitness, initial_population)
    solver = GPModelSolver(tournament_size, mutation_rate, sampler)
    
    learn!(model, strategy(solver, Verbose(MaxIter(maxiter)), tracer))

    return model.population, tracer
end
