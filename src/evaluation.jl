using DataFrames

"""
    accuracy(target, output; normalize = true)

If `normalize` is `true`, the fraction of correctly classified observations is returned.  Otherwise
the total number is returned.

Adapted from https://github.com/JuliaML/MLMetrics.jl/blob/c03435006ab64f84376835ad588d9cbf546da506/src/classification/multiclass.jl#L194.
"""
function accuracy(target::AbstractVector,
                  output::AbstractVector;
                  normalize = true)
    @assert length(target) == length(output)
    correct = 0
    
    @inbounds for i in eachindex(target, output)
        correct += target[i] == output[i]
    end
    
    normalize ? Float64(correct / length(target)) : Float64(correct)
end


"""
    kfolds(n::Integer, [k = 5]) -> Tuple

Compute the train/validation assignments for `k` repartitions of `n` observations, and return them
in the form of two vectors. The first vector contains the index-vectors for the training subsets,
and the second vector the index-vectors for the validation subsets respectively.

From https://github.com/JuliaML/MLDataPattern.jl/blob/d9fcafe1cdd7230e249d3ae85163eef088e3d659/src/folds.jl#L226.
"""
function kfolds(n::Integer, k::Integer = 5)
    2 <= k <= n || throw(ArgumentError("n must be positive and k must to be within 2:$(max(2,n))"))
    # Compute the size of each fold. This is important because
    # in general the number of total observations might not be
    # divideable by k. In such cases it is custom that the remaining
    # observations are divided among the folds. Thus some folds
    # have one more observation than others.
    sizes = fill(floor(Int, n / k), k)
    for i = 1:(n % k)
        sizes[i] = sizes[i] + 1
    end
    # Compute start offset for each fold
    offsets = cumsum(sizes) .- sizes .+ 1
    # Compute the validation indices using the offsets and sizes
    val_indices = map((o,s)->(o:o+s-1), offsets, sizes)
    # The train indices are then the indicies not in validation
    train_indices = map(idx->setdiff(1:n,idx), val_indices)
    # We return a tuple of arrays
    train_indices, val_indices
end


function prepare_dataset{N, C}(df::DataFrame, ::Type{Val{N}}, ::Type{Val{C}})
    features = convert(Matrix{Float64}, df[1:end-1])
    classes = recode(df[end], map(=>,
                                  levels(df[end]),
                                  Classification{N, C}.(1:C))...)

    features, classes
end


function create_fitness{N, C}(data, Vn::Type{Val{N}}, Vc::Type{Val{C}};
                              size_penalty::Float64 = 0.0, depth_penalty::Float64 = 0.0)
    features, classes = prepare_dataset(data, Vn, Vc)
    
    
    function fitness(tree::DecisionTree{N, C})
        classify(d) = decide(d, tree)
        predictions = squeeze(mapslices(classify, features, 2), 2) # TODO: work on transposed matrix?
        penalty = size_penalty * treesize(tree) + depth_penalty * treedepth(tree)
        accuracy(predictions, data[N+1]) + penalty
    end

    fitness
end

# glass = SoftComputingFinal.load_glass();
# t = rand(SoftComputingFinal.BoltzmannSampler{9}(10, 20));
# fitness = SoftComputingFinal.create_fitness(glass, Val{9})
