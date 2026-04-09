"""
    analyze_synthetic_gcms_with_ncpls(
        data;
        model::Union{NCPLSModel, Nothing}=nothing,
        ncomponents::Integer=2,
        multilinear::Bool=true,
        orthogonalize_mode_weights::Bool=false,
        center_X::Bool=true,
        scale_X::Bool=false,
        center_Yprim::Bool=true,
        multilinear_maxiter::Int=500,
        multilinear_tol::Float64=1e-10,
        multilinear_init::Symbol=:hosvd,
        multilinear_seed::Int=1,
        test_fraction::Real=0.3,
        rng::AbstractRNG=MersenneTwister(1),
        train_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
        test_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
        verbose::Bool=false,
    )

Analyze synthetic GC-MS regression data with NCPLS.

The predictor tensor is kept in tensor form, `data.Yadd` is passed as additional
responses when present, and the result contains the fitted NCPLS model together with
train/test scores, predictions, and simple component-wise RMSE and R² summaries.
"""
function analyze_synthetic_gcms_with_ncpls(
    data;
    model::Union{NCPLSModel, Nothing}=nothing,
    ncomponents::Integer=2,
    multilinear::Bool=true,
    orthogonalize_mode_weights::Bool=false,
    center_X::Bool=true,
    scale_X::Bool=false,
    center_Yprim::Bool=true,
    multilinear_maxiter::Int=500,
    multilinear_tol::Float64=1e-10,
    multilinear_init::Symbol=:hosvd,
    multilinear_seed::Int=1,
    test_fraction::Real=0.3,
    rng::AbstractRNG=MersenneTwister(1),
    train_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    test_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    verbose::Bool=false,
)
    ncomponents > 0 || throw(ArgumentError("ncomponents must be greater than zero"))

    X = data.X
    Yprim = data.Yprim
    Yadd = data.Yadd

    ndims(X) ≥ 2 || throw(ArgumentError(
        "data.X must have at least 2 dimensions: samples × variables[/modes]"))
    size(X, 1) == size(Yprim, 1) || throw(DimensionMismatch(
        "data.X and data.Yprim must have the same number of samples"))
    isnothing(Yadd) || size(Yadd, 1) == size(X, 1) || throw(DimensionMismatch(
        "data.Yadd must have the same number of samples as data.X"))

    train_idx, test_idx = split_synthetic_gcms_samples(
        size(X, 1);
        test_fraction = test_fraction,
        rng = rng,
        train_idx = train_idx,
        test_idx = test_idx,
    )

    tail = ntuple(_ -> Colon(), ndims(X) - 1)
    Xtrain = X[train_idx, tail...]
    Xtest = X[test_idx, tail...]
    Ytrain = Yprim[train_idx, :]
    Ytest = Yprim[test_idx, :]
    Yadd_train = isnothing(Yadd) ? nothing : Yadd[train_idx, :]
    Yadd_test = isnothing(Yadd) ? nothing : Yadd[test_idx, :]

    predictorlabels, responselabels = synthetic_gcms_labels(data)

    ncpls_model = isnothing(model) ? NCPLSModel(
        ncomponents = ncomponents,
        center_X = center_X,
        scale_X = scale_X,
        center_Yprim = center_Yprim,
        multilinear = multilinear,
        orthogonalize_mode_weights = orthogonalize_mode_weights,
        multilinear_maxiter = multilinear_maxiter,
        multilinear_tol = multilinear_tol,
        multilinear_init = multilinear_init,
        multilinear_seed = multilinear_seed,
    ) : model

    mf = fit(
        ncpls_model,
        Xtrain,
        Ytrain;
        Yadd = Yadd_train,
        obs_weights = nothing,
        verbose = verbose,
    )

    Yhat_train_raw = predict(mf, Xtrain)
    Yhat_test_raw = predict(mf, Xtest)
    scores_train = project(mf, Xtrain)
    scores_test = project(mf, Xtest)

    Yhat_train = components_last_predictions(Yhat_train_raw, 2)
    Yhat_test = components_last_predictions(Yhat_test_raw, 2)

    train_metrics = componentwise_regression_metrics(Yhat_train, Ytrain)
    test_metrics = componentwise_regression_metrics(Yhat_test, Ytest)

    (
        ncpls_model = ncpls_model,
        ncplsfit = mf,
        train_idx = train_idx,
        test_idx = test_idx,
        Xtrain = Xtrain,
        Xtest = Xtest,
        Ytrain = Ytrain,
        Ytest = Ytest,
        Yadd_train = Yadd_train,
        Yadd_test = Yadd_test,
        predictorlabels = predictorlabels,
        responselabels = responselabels,
        scores_train = scores_train,
        scores_test = scores_test,
        Yhat_train = Yhat_train,
        Yhat_test = Yhat_test,
        rmse_train = train_metrics.rmse,
        rmse_test = test_metrics.rmse,
        r2_train = train_metrics.r2,
        r2_test = test_metrics.r2,
        rmse_train_overall = train_metrics.rmse_overall,
        rmse_test_overall = test_metrics.rmse_overall,
        r2_train_overall = train_metrics.r2_overall,
        r2_test_overall = test_metrics.r2_overall,
    )
end
