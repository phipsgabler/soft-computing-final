include("../src/SoftComputingFinal.jl")

using SoftComputingFinal
using Base.Test

const TEST_RNG = MersenneTwister(12345678)

@testset "DecisionTree" begin
    forest = [Variable{3}(1),
              Branch(Dict(Variable{3}(1) => 0.1, Variable{3}(2) => 3.0), 1.0,
                     Variable{3}(2), Variable{3}(1)),
              Branch(Dict(Variable{3}(1) => -4.2, Variable{3}(3) => 9.93), 1.16,
                     Branch(Dict(Variable{3}(1) => -0.79, Variable{3}(2) => 2.87), 5.96,
                            Variable{3}(1), Variable{3}(2)),
                     Branch(Dict(Variable{3}(1) => -7.41, Variable{3}(3) => -3.03), 5.16,
                            Variable{3}(2), Variable{3}(1)))]
    
    printed = ["""x₁""",
               """
               if 0.1 × x₁ + 3.0 × x₂ ≤ 1.0
                 then x₂
                 else x₁""",
               """
               if -4.2 × x₁ + 9.93 × x₃ ≤ 1.16
                 then if -0.79 × x₁ + 2.87 × x₂ ≤ 5.96
                   then x₁
                   else x₂
                 else if -7.41 × x₁ + -3.03 × x₃ ≤ 5.16
                   then x₂
                   else x₁"""]

    depths = [1, 2, 3]
    sizes = [1, 3, 7]
    conditions = [nothing, [0.1, 3.0, 0.0], [-4.2, 0.0, 9.93]]
    
    for i in eachindex(forest)
        if forest[i] isa Branch
            @test normalize_conditions(forest[i].conditions) == conditions[i]
        end
        
        @test string(forest[i]) == printed[i]
        @test treedepth(forest[i]) == depths[i]
        s = treesize(forest[i])
        @test s == sizes[i]

        
        newt, chunk = randomsplit(TEST_RNG, forest[i]) do chunk
            Variable{3}(3)
        end
        @test treesize(newt) - 1 == s - treesize(chunk)
    end

    
end

