using SoftComputingFinal
using Base.Test

# write your own tests here
t = Branch{2}([(Variable{2}(1), 0.1), (Variable{2}(2), 3)],
              1.0,
              Variable{2}(2),
              Variable{2}(1))
