using SoftComputingFinal
using Base.Test

@testset "DecisionTree" begin
    t = Branch{2}(Dict(Variable{2}(1) => 0.1, Variable{2}(2) => 3),
                  1.0,
                  Variable{2}(2),
                  Variable{2}(1))
    printed = """
              if 0.1 × {1} + 3.0 × {2} ≤ 1.0
                then {2}
                else {1}"""

    @test normalize_conditions(t.conditions) == [0.1, 3.0]
    @test string(t) == printed
end

