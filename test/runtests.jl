push!(LOAD_PATH,"../src/")

using SoftComputingFinal
using Base.Test

const TEST_RNG = MersenneTwister(12345678)

@testset "DecisionTree" begin
    forest = [Classification{3, 3}(1),
              Decision(Dict(Variable{3}(1) => 0.1, Variable{3}(2) => 3.0), 1.0,
                       Classification{3, 3}(2),
                       Classification{3, 3}(1)),
              Decision(Dict(Variable{3}(1) => -4.2, Variable{3}(3) => 9.93), 1.16,
                       Decision(Dict(Variable{3}(1) => -0.79, Variable{3}(2) => 2.87), 5.96,
                                Classification{3, 3}(1),
                                Classification{3, 3}(2)),
                       Decision(Dict(Variable{3}(1) => -7.41, Variable{3}(3) => -3.03), 5.16,
                                Classification{3, 3}(2),
                                Classification{3, 3}(1)))]
    
    printed = ["""{1}""",
               """
               if 0.1 × x₁ + 3.0 × x₂ ≤ 1.0
                 then {2}
                 else {1}""",
               """
               if -4.2 × x₁ + 9.93 × x₃ ≤ 1.16
                 then if -0.79 × x₁ + 2.87 × x₂ ≤ 5.96
                   then {1}
                   else {2}
                 else if -7.41 × x₁ + -3.03 × x₃ ≤ 5.16
                   then {2}
                   else {1}"""]

    depths = [1, 2, 3]
    sizes = [1, 3, 7]
    conditions = [nothing, [0.1, 3.0, 0.0], [-4.2, 0.0, 9.93]]
    
    for i in eachindex(forest)
        if forest[i] isa Decision
            @test normalize_conditions(forest[i].conditions) == conditions[i]
        end
        
        @test string(forest[i]) == printed[i]
        @test treedepth(forest[i]) == depths[i]
        s = treesize(forest[i])
        @test s == sizes[i]

        
        newt, chunk = randsplit(TEST_RNG, forest[i]) do chunk
            Classification{3, 3}(3)
        end
        @test treesize(newt) - 1 == s - treesize(chunk)
    end
end

@testset "Sampling" begin
    # SplitSampler
    split_samples = rand(SplitSampler{10, 3}(5, 1.0, minvars = 2, maxvars = 4), 20)
    @test all(treedepth(t) ≤ 5 for t in split_samples)
    @test all(t isa Classification || 2 ≤ length(t.conditions) ≤ 4 for t in split_samples)

    # RampedSplitSampler
    ramped_split_samples = rand(RampedSplitSampler{10, 3}(5, 0.5, 0.5, minvars = 2, maxvars = 4), 20)
    @test all(treedepth(t) ≤ 5 for t in ramped_split_samples)
    @test all(t isa Classification || 2 ≤ length(t.conditions) ≤ 4 for t in ramped_split_samples)

    #BoltzmannSampler
    boltzmann_samples = rand(BoltzmannSampler{10, 3}(5, 20, minvars = 2, maxvars = 4), 20)
    @test all(5 ≤ treesize(t) ≤ 20 for t in boltzmann_samples)
    @test all(t isa Classification || 2 ≤ length(t.conditions) ≤ 4 for t in boltzmann_samples)
end

@testset "SSGP Integration" begin
    N = 500
    data = load_testdata()
    fitness, accuracy = create_fitness(data, Val{2}, Val{2},
                                       depth_penalty = 2.0, size_penalty = 0.5)

    initializer = RampedSplitSampler{2, 2}(4, 0.5, 0.5, crange = (-2.0, 2.0))
    population, _ = runssgp(fitness, 20, initializer, N, verbose = false)

    best_tree = select(population, 1, by = fitness, rev = true)
    @show α = accuracy(best_tree)
    @test α > 0.7
end

