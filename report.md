## Choices made

- Representation of trees: the paper in its description stays close to the "strongly typed
  GP" formalism of Montana (1995).  I interpreted this more loosely and represented trees as types
  in Julia, with all operators formulated in the host language to ensure well-formedness
  (eg. random generation produces well-formed trees in Julia automatically).
- Initialization/random generation: the `RampedSplitSampler` implements the ramped-half-and-half
  sampling described in the paper by Koza (1992), with the tweak that the proportion grow and full
  sampling happens stochastically; the reason for this was to get a meaningful implementation for 
  sampling a single tree.
  
  Furthermore, the paper didn't specify the maximum depths or sizes used
  for generating the population or the mutation subtrees.  I therefore tried to choose a
  reasonable default.
- Implementation of the fitness function:
- Implementation of genetic operators:
    * The paper states that a "steady-state GA with elitism" had been used.
      I used steady-state, ie., only one individual per generation is changed
      and always replaced the individual with the least fitness.
    * Crossover and mutation both happen with probability 0.5.  Since crossover
      only returns one individual, a random parent is chosen in the case no actual 
      mating happens.
    * Selection is stated to use tournament selection of size 7.  I therefore choose
      7 random individuals with replacement and return the best of those.


## Outcome

