using LearningStrategies
import LearningStrategies: update!

struct GPModel{N}
    population::Vector{DecisionTree{N}}
    fitness::Function
    fitness_values::Vector
end

GPModel(population, fitness) = GPModel(population, fitness, fitness.(population))


struct GPModelSolver{N} <: LearningStrategy
    tournament_size::Int
    mutation_probability::Float64
    mutation_sampler::TreeSampler{N}
end


function selection(parents, parent_fitnesses, k)
    candidates = rand(eachindex(parents, parent_fitnesses), k)
    parents[indmax(parent_fitnesses[candidates])]
end

function crossover(parent₁, parent₂)
    # choosing a random `chunk` from `parent₂` and splice it somewhere into `parent₁`
    newtree, _ = randomsplit(parent₁) do _
        randomchild(parent₂)
    end

    return newtree
end

function mutate(individual, pₘ, mutation_sampler)
    if rand() ≤ pₘ
        newtree, chunk = randomsplit(individual) do _
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
    fitness_values = model.fitness_values
    fitness = model.fitness
    k = s.tournament_size
    pₘ = s.mutation_probability
    ms = s.mutation_sampler

    # breed
    # TODO paralleize this
    children = similar(parents)
    for i in eachindex(children)
        p₁ = selection(parents, fitness_values, k)
        p₂ = selection(parents, fitness_values, k)
        child = crossover(p₁, p₂)
        children[i] = mutate(child, pₘ, ms)
    end

    # update
    parents .= children
    fitness_values .= fitness.(parents) # TODO: pmap this!
end


function rungp{N}(fitness, psize::Int, sampler::TreeSampler{N}, maxiter::Int;
                  tracer = Tracer(Void, (m, i) -> nothing, typemax(Int)))
    population = rand(sampler, psize)
    
    learn!(GPModel(population, fitness),
           strategy(GPModelSolver(7, 0.5, sampler), # use log(size) / 2 for mutation sampler?
                    Verbose(MaxIter(maxiter)),
                    tracer))

    return population, tracer
end


function testgp(N)
    tracer = Tracer(Int, (m, i) -> maximum(m.fitness_values))
    pop, tracer = rungp(treesize, 20, BoltzmannSampler{4}(10, 20), N, tracer = tracer)
    collect(tracer)
end
