using Test
using Random

function suppress_info(f::Function)
    logger = Base.CoreLogging.SimpleLogger(IOBuffer(), Base.CoreLogging.Error)
    Base.CoreLogging.with_logger(logger) do
        f()
    end
end

const CROSSVAL_X_MATRIX = Float64[
    1.1 0.5 1.7 2.3
    2.2 1.1 0.4 3.6
    3.3 1.8 2.5 4.9
    4.4 2.6 1.1 6.1
    5.5 3.3 3.2 7.4
    6.6 4.0 1.8 8.8
    7.7 4.6 2.9 9.9
    8.8 5.3 0.7 11.2
    9.9 6.1 3.6 12.5
    11.0 6.8 2.2 13.7
    12.1 7.5 4.3 15.0
    13.2 8.2 1.5 16.3
    14.3 9.0 3.8 17.5
    15.4 9.7 2.0 18.9
    16.5 10.4 4.6 20.1
    17.6 11.1 1.2 21.4
]

const CROSSVAL_X_TENSOR = reshape(copy(CROSSVAL_X_MATRIX), size(CROSSVAL_X_MATRIX, 1), 2, 2)

const CROSSVAL_CLASSES = repeat(["A", "B"], inner = size(CROSSVAL_X_MATRIX, 1) ÷ 2)
const CROSSVAL_CLASSES_CAT = NCPLS.categorical(CROSSVAL_CLASSES)
const CROSSVAL_Y = NCPLS.onehot(CROSSVAL_CLASSES)[1]
const CROSSVAL_Y_REG = reshape(
    CROSSVAL_X_MATRIX[:, 1] .+ 0.5 .* CROSSVAL_X_MATRIX[:, 2],
    :,
    1,
)

function crossval_spec(; multilinear::Bool=true)
    NCPLSModel(ncomponents = 1, multilinear = multilinear, multilinear_seed = 2)
end

@testset "encoding helpers" begin
    encoded = NCPLS.onehot([1, 3, 2, 3, 1], 3)
    @test encoded == [
        1 0 0
        0 0 1
        0 1 0
        0 0 1
        1 0 0
    ]
    @test_throws ArgumentError NCPLS.onehot([1, 4], 3)
    @test_throws ArgumentError NCPLS.onehot([0, 1], 3)
    @test_throws ArgumentError NCPLS.onehot([1, 2], -1)

    encoded_labels, uniques = NCPLS.onehot(["cat", "dog", "cat", "owl", "dog"])
    @test uniques == ["cat", "dog", "owl"]
    @test encoded_labels == [
        1 0 0
        0 1 0
        1 0 0
        0 0 1
        0 1 0
    ]

    @test NCPLS.sampleclasses([0 1 0; 1 0 0; 0 0 1; 0 1 0]) == [2, 1, 3, 2]
    @test_throws ArgumentError NCPLS.sampleclasses([1 1 0; 0 1 0])
    @test_throws ArgumentError NCPLS.sampleclasses([0 0 0; 0 1 0])
    @test_throws ArgumentError NCPLS.sampleclasses([2 0 0; 0 1 0])
end

@testset "metric helpers" begin
    samples = ["A", "A", "B", "C", "C", "C"]
    w = NCPLS.invfreqweights(samples)
    @test length(w) == length(samples)
    @test sum(w) ≈ 1.0
    @test w[1] == w[2]
    @test w[3] > w[1] > w[4]
    @test sum(w[[1, 2]]) ≈ sum(w[[3]])
    @test sum(w[[3]]) ≈ sum(w[[4, 5, 6]])

    Y_true = [
        1 0 0
        0 1 0
        0 1 0
        0 1 0
        0 0 1
    ]
    Y_pred = [
        0 1 0
        0 1 0
        0 0 1
        0 1 0
        0 0 1
    ]
    @test NCPLS.nmc(Y_true, Y_pred, false) == 4 / 15
    @test NCPLS.nmc(Y_true, Y_pred, true) == 4 / 9
    @test_throws DimensionMismatch NCPLS.nmc(Y_true, Y_pred[1:1, :], false)
    @test_throws ArgumentError NCPLS.nmc(Y_true[1:0, :], Y_pred[1:0, :], true)

    perms = [0.4, 0.6, 0.5, 0.55]
    observed = 0.5
    @test NCPLS.pvalue(perms, observed) == 4 / (length(perms) + 1)
    @test NCPLS.pvalue(perms, observed; tail = :lower) == 3 / (length(perms) + 1)
    @test_throws ArgumentError NCPLS.pvalue(perms, observed; tail = :sideways)
end

@testset "classification prediction helpers use last NCPLS component slice" begin
    mf = NCPLS.fit(
        NCPLSModel(ncomponents = 2, multilinear = false),
        CROSSVAL_X_MATRIX,
        CROSSVAL_Y;
        responselabels = ["A", "B"],
    )

    preds = NCPLS.predict(mf, CROSSVAL_X_MATRIX, 2)
    expected = zeros(Int, size(preds, 1), size(preds, 3))
    for (i, cls) in enumerate(argmax.(eachrow(preds[:, end, :])))
        expected[i, cls] = 1
    end

    @test NCPLS.onehot(mf, preds) == expected
    @test NCPLS.onehot(mf, CROSSVAL_X_MATRIX, 2) == expected
    @test NCPLS.predictclasses(mf, preds) == NCPLS.responselabels(mf)[NCPLS.sampleclasses(expected)]
end

@testset "classification helpers isolate class columns in mixed response fits" begin
    X = CROSSVAL_X_MATRIX[1:4, :]
    class_labels = ["A", "B", "A", "B"]
    Yclass, _ = NCPLS.onehot(class_labels)
    Ymixed = hcat(Yclass, reshape([10.0, 20.0, 30.0, 40.0], :, 1))

    mf = NCPLS.fit(
        NCPLSModel(ncomponents = 2, multilinear = false),
        X,
        Ymixed;
        sampleclasses = class_labels,
        responselabels = ["A", "B", "trait"],
    )

    preds = zeros(Float64, 4, 2, 3)
    preds[:, 2, :] = [
        0.9 0.1 10.0
        0.2 0.8 11.0
        0.6 0.4 12.0
        0.1 0.9 13.0
    ]
    expected = [
        1 0
        0 1
        1 0
        0 1
    ]

    @test NCPLS.onehot(mf, preds) == expected
    @test NCPLS.predictclasses(mf, preds) == ["A", "B", "A", "B"]
    @test_throws MethodError NCPLS.sampleclasses(mf, preds)
end

@testset "random_batch_indices builds stratified folds" begin
    strata = vcat(fill(1, 6), fill(2, 6))
    folds = NCPLS.random_batch_indices(strata, 3, MersenneTwister(1))

    @test length(folds) == 3
    @test sort!(reduce(vcat, folds)) == collect(1:length(strata))
    @test all(length(batch) == 4 for batch in folds)
    @test_throws ArgumentError NCPLS.random_batch_indices(strata, 4)
    @test_throws ArgumentError NCPLS.random_batch_indices(strata, 0)
    @test_throws ArgumentError NCPLS.random_batch_indices(strata, length(strata) + 1)

    uneven = vcat(fill(1, 5), fill(2, 4))
    @test_logs (:info, r"Stratum 1 .* not evenly divisible") begin
        NCPLS.random_batch_indices(uneven, 2, MersenneTwister(2))
    end
end

@testset "default callback bundles" begin
    cfg = NCPLS.cv_classification()
    @test haskey(cfg, :score_fn)
    @test haskey(cfg, :predict_fn)
    @test haskey(cfg, :select_fn)
    @test haskey(cfg, :flag_fn)
    @test cfg.select_fn([0.1, 0.2]) == 2
    @test cfg.flag_fn([1 0; 0 1], [1 0; 1 0]) == [false, true]

    reg = NCPLS.cv_regression()
    @test reg.score_fn(reshape([1.0, 2.0], :, 1), reshape([1.0, 3.0], :, 1)) ≈ sqrt(0.5)
    @test reg.select_fn([0.3, 0.2]) == 2

    mf = NCPLS.fit_ncpls_light(
        crossval_spec(multilinear = false),
        CROSSVAL_X_MATRIX,
        CROSSVAL_Y_REG,
    )
    @test reg.predict_fn(mf, CROSSVAL_X_MATRIX[1:3, :], 1) ≈ NCPLS.predict(
        mf,
        CROSSVAL_X_MATRIX[1:3, :],
        1,
    )[:, end, :]
end

@testset "resolve_obs_weights combines fixed and fold-local weights" begin
    resolved = NCPLS.resolve_obs_weights(
        (; obs_weights = [1.0, 2.0]),
        (X, Y; kwargs...) -> [0.5, 0.25],
        CROSSVAL_X_MATRIX[1:2, :],
        CROSSVAL_Y[1:2, :],
        [3, 7],
        crossval_spec(multilinear = false),
    )
    @test resolved.obs_weights ≈ [0.5, 0.5]
end

@testset "nestedcv returns scores and components for tensor predictors" begin
    cfg = NCPLS.cv_classification()
    weight_calls = Ref(0)

    scores, components = suppress_info() do
        NCPLS.nestedcv(
            CROSSVAL_X_TENSOR,
            CROSSVAL_Y;
            spec = crossval_spec(),
            fit_kwargs = (; responselabels = ["A", "B"]),
            obs_weight_fn = (X, Y; kwargs...) -> begin
                weight_calls[] += 1
                ones(size(X, 1))
            end,
            score_fn = cfg.score_fn,
            predict_fn = cfg.predict_fn,
            select_fn = cfg.select_fn,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            strata = NCPLS.sampleclasses(CROSSVAL_Y),
            rng = MersenneTwister(123),
            verbose = false,
        )
    end

    @test length(scores) == 2
    @test all(0.0 ≤ acc ≤ 1.0 for acc in scores)
    @test components == [1, 1]
    @test weight_calls[] == 6
end

@testset "nestedcvperm shuffles tensor-response pairings" begin
    cfg = NCPLS.cv_classification()
    perms = suppress_info() do
        NCPLS.nestedcvperm(
            CROSSVAL_X_TENSOR,
            CROSSVAL_Y;
            spec = crossval_spec(),
            fit_kwargs = (; responselabels = ["A", "B"]),
            obs_weight_fn = (X, Y; kwargs...) -> ones(size(X, 1)),
            score_fn = cfg.score_fn,
            predict_fn = cfg.predict_fn,
            select_fn = cfg.select_fn,
            num_permutations = 2,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            max_components = 1,
            strata = NCPLS.sampleclasses(CROSSVAL_Y),
            rng = MersenneTwister(321),
            verbose = false,
        )
    end

    @test length(perms) == 2
    @test all(0.0 ≤ acc ≤ 1.0 for acc in perms)
end

@testset "cvreg and permreg support tensor predictors" begin
    scores, components = suppress_info() do
        NCPLS.cvreg(
            CROSSVAL_X_TENSOR,
            CROSSVAL_Y_REG;
            spec = crossval_spec(),
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            rng = MersenneTwister(666),
            verbose = false,
        )
    end
    @test length(scores) == 2
    @test components == [1, 1]
    @test all(isfinite, scores)

    scores_vec, components_vec = suppress_info() do
        NCPLS.cvreg(
            CROSSVAL_X_TENSOR,
            vec(CROSSVAL_Y_REG);
            spec = crossval_spec(),
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            rng = MersenneTwister(666),
            verbose = false,
        )
    end
    @test scores_vec == scores
    @test components_vec == components

    permutation_scores = suppress_info() do
        NCPLS.permreg(
            CROSSVAL_X_TENSOR,
            vec(CROSSVAL_Y_REG);
            spec = crossval_spec(),
            num_permutations = 2,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            rng = MersenneTwister(777),
            verbose = false,
        )
    end
    @test length(permutation_scores) == 2
    @test all(isfinite, permutation_scores)
end

@testset "cvda and permda support matrix and categorical-label inputs" begin
    scores, components = suppress_info() do
        NCPLS.cvda(
            CROSSVAL_X_TENSOR,
            CROSSVAL_CLASSES_CAT;
            spec = crossval_spec(),
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            rng = MersenneTwister(444),
            verbose = false,
        )
    end
    @test length(scores) == 2
    @test components == [1, 1]
    @test all(0.0 ≤ acc ≤ 1.0 for acc in scores)

    scores_matrix, components_matrix = suppress_info() do
        NCPLS.cvda(
            CROSSVAL_X_TENSOR,
            CROSSVAL_Y;
            spec = crossval_spec(),
            fit_kwargs = (; responselabels = ["A", "B"]),
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            rng = MersenneTwister(444),
            verbose = false,
        )
    end
    @test scores_matrix == scores
    @test components_matrix == components
    @test_throws ArgumentError NCPLS.cvda(CROSSVAL_X_TENSOR, CROSSVAL_CLASSES; spec = crossval_spec())
    @test_throws ArgumentError NCPLS.cvda(CROSSVAL_X_TENSOR, collect(1:size(CROSSVAL_X_TENSOR, 1)); spec = crossval_spec())

    permutation_scores = suppress_info() do
        NCPLS.permda(
            CROSSVAL_X_TENSOR,
            CROSSVAL_CLASSES_CAT;
            spec = crossval_spec(),
            num_permutations = 2,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            rng = MersenneTwister(555),
            verbose = false,
        )
    end
    @test length(permutation_scores) == 2
    @test all(0.0 ≤ acc ≤ 1.0 for acc in permutation_scores)
    @test_throws ArgumentError NCPLS.permda(CROSSVAL_X_TENSOR, CROSSVAL_CLASSES; spec = crossval_spec())
    @test_throws ArgumentError NCPLS.permda(CROSSVAL_X_TENSOR, collect(1:size(CROSSVAL_X_TENSOR, 1)); spec = crossval_spec())
end

@testset "outlierscan returns per-sample counts for tensor predictors" begin
    out = suppress_info() do
        NCPLS.outlierscan(
            CROSSVAL_X_TENSOR,
            CROSSVAL_Y;
            spec = crossval_spec(),
            fit_kwargs = (; responselabels = ["A", "B"]),
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            rng = MersenneTwister(111),
            verbose = false,
        )
    end

    @test length(out.n_tested) == size(CROSSVAL_X_TENSOR, 1)
    @test length(out.n_flagged) == size(CROSSVAL_X_TENSOR, 1)
    @test all(0.0 ≤ r ≤ 1.0 for r in out.rate)
    @test sum(out.n_tested) == 2 * (size(CROSSVAL_X_TENSOR, 1) ÷ 2)
    @test all(out.n_flagged .≤ out.n_tested)

    out_unweighted = suppress_info() do
        NCPLS.outlierscan(
            CROSSVAL_X_TENSOR,
            CROSSVAL_Y;
            spec = crossval_spec(),
            fit_kwargs = (; responselabels = ["A", "B"]),
            obs_weight_fn = nothing,
            weighted = false,
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            rng = MersenneTwister(111),
            verbose = false,
        )
    end
    @test length(out_unweighted.n_tested) == size(CROSSVAL_X_TENSOR, 1)
    @test all(out_unweighted.n_flagged .≤ out_unweighted.n_tested)

    out_labels = suppress_info() do
        NCPLS.outlierscan(
            CROSSVAL_X_TENSOR,
            CROSSVAL_CLASSES_CAT;
            spec = crossval_spec(),
            num_outer_folds = 2,
            num_outer_folds_repeats = 2,
            num_inner_folds = 2,
            num_inner_folds_repeats = 2,
            rng = MersenneTwister(111),
            verbose = false,
        )
    end
    @test length(out_labels.n_tested) == size(CROSSVAL_X_TENSOR, 1)
    @test all(out_labels.n_flagged .≤ out_labels.n_tested)
    @test_throws ArgumentError NCPLS.outlierscan(CROSSVAL_X_TENSOR, CROSSVAL_CLASSES; spec = crossval_spec())
end
