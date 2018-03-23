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
    max_depth::Int
    tournament_size::Int
    crossover_probability::Float64
    mutation_probability::Float64
    mutation_sampler::TreeSampler{N, C}
    rng::AbstractRNG
end


function selection(parents, parent_fitnesses, s)
    candidates = rand(eachindex(parents, parent_fitnesses), s.tournament_size)
    parents[indmax(parent_fitnesses[candidates])]
end

function mate(p, q, s)
    # Choose a random subtree from `p` and replace it by a random child of `q`
    newtree, _ = randsplit((subtree) -> randchild(q), s.rng, p)
    newtree

    newtree
end

function crossover(parent₁, parent₂, s)
    if rand(s.rng) ≤ s.crossover_probability
        candidate = mate(parent₁, parent₂, s)
        while treedepth(candidate) > s.max_depth
            candidate = mate(parent₁, parent₂, s)
        end

        return candidate
    else
        return rand(s.rng, [parent₁, parent₂])
    end
end

function splice(i, s)
    # Choose a random subtree from `p` and put in a random tree
    newtree, _ = randsplit((subtree) -> rand(s.rng, s.mutation_sampler), i)
    newtree
end

function mutate(individual, s)
    if rand(s.rng) ≤ s.mutation_probability
        candidate = splice(individual, s)
        while treedepth(candidate) > s.max_depth
            candidate = splice(individual, s)
        end

        return candidate
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

    # breed
    for i in eachindex(children, child_fitnesses)
        p₁ = selection(parents, parent_fitnesses, s)
        p₂ = selection(parents, parent_fitnesses, s)
        child = crossover(p₁, p₂, s)
        children[i] = mutate(child, s)
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
                     breaker = Breaker((m, i) -> false),
                     max_depth = 20, tournament_size = 7,
                     mutation_rate = 0.5, crossover_rate = 0.5,
                     depth_penalty = 2.0, size_penalty = 0.5,
                     rng = Base.GLOBAL_RNG)
    initial_population = rand(sampler, psize)
    model = GPModel(float ∘ fitness, initial_population)
    solver = GPModelSolver(max_depth, tournament_size, crossover_rate, mutation_rate, sampler, rng)
    
    learn!(model, strategy(solver, Verbose(MaxIter(maxiter)), tracer, breaker))

    return model.population, tracer
end
