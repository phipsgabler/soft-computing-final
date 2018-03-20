include("../src/SoftComputingFinal.jl")

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

        
        newt, chunk = randomsplit(TEST_RNG, forest[i]) do chunk
            Classification{3, 3}(3)
        end
        @test treesize(newt) - 1 == s - treesize(chunk)
    end

    
end

