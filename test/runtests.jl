include("../src/SoftComputingFinal.jl")

using SoftComputingFinal
using Base.Test

@testset "DecisionTree" begin
    t = Branch{2}(Dict(Variable{2}(1) => 0.1, Variable{2}(2) => 3),
                  1.0,
                  Variable{2}(2),
                  Variable{2}(1))
    printed = """
              if 0.1 × x₁ + 3.0 × x₂ ≤ 1.0
                then x₂
                else x₁"""

    @test normalize_conditions(t.conditions) == [0.1, 3.0]
    @test string(t) == printed

    @test treedepth(Variable{10}(1)) == 1
    @test treedepth(t) == 2
    @test treesize(Variable{10}(1)) == 1
    @test treesize(t) == 3
end

