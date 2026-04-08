"""
    analyze_synthetic_gcms_with_cppls(
        data;
        model::Union{CPPLS.CPPLSModel, Nothing}=nothing,
        ncomponents::Integer=2,
        test_fraction::Real=0.3,
        center_X::Bool=true,
        scale_X::Bool=false,
        scale_Yprim::Bool=false,
        rng::AbstractRNG=MersenneTwister(1),
        train_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
        test_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    )

Analyze synthetic GC-MS regression data with unfolded CPPLS.

The predictor tensor is unfolded along the sample mode, `CPPLS` is fit with
`gamma = 0.5`, and `data.Yadd` is passed as `Yaux` when present. The result contains
the fitted CPPLS model together with train/test scores, predictions, and simple
component-wise RMSE and R² summaries.
"""
function analyze_synthetic_gcms_with_cppls(
    data;
    model::Union{CPPLS.CPPLSModel, Nothing}=nothing,
    ncomponents::Integer=2,
    test_fraction::Real=0.3,
    center_X::Bool=true,
    scale_X::Bool=false,
    scale_Yprim::Bool=false,
    rng::AbstractRNG=MersenneTwister(1),
    train_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    test_idx::Union{AbstractVector{<:Integer}, Nothing}=nothing,
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
    Xtrain_tensor = X[train_idx, tail...]
    Xtest_tensor = X[test_idx, tail...]
    Xtrain = float64(reshape(Xtrain_tensor, length(train_idx), :))
    Xtest = float64(reshape(Xtest_tensor, length(test_idx), :))

    Ytrain = Yprim[train_idx, :]
    Ytest = Yprim[test_idx, :]
    Yaux_train = isnothing(Yadd) ? nothing : Yadd[train_idx, :]
    Yaux_test = isnothing(Yadd) ? nothing : Yadd[test_idx, :]

    predictorlabels, responselabels = synthetic_gcms_labels(data)

    cppls_model = isnothing(model) ? CPPLS.CPPLSModel(
        ncomponents = ncomponents,
        gamma = 0.5,
        center_X = center_X,
        scale_X = scale_X,
        scale_Yprim = scale_Yprim,
        mode = :regression,
    ) : model

    cpplsfit = CPPLS.fit(
        cppls_model,
        Xtrain,
        Ytrain;
        Yaux = Yaux_train,
        predictorlabels = predictorlabels,
        responselabels = responselabels,
    )

    Yhat_train = CPPLS.predict(cpplsfit, Xtrain)
    Yhat_test = CPPLS.predict(cpplsfit, Xtest)
    scores_train = CPPLS.project(cpplsfit, Xtrain)
    scores_test = CPPLS.project(cpplsfit, Xtest)

    train_metrics = componentwise_regression_metrics(Yhat_train, Ytrain)
    test_metrics = componentwise_regression_metrics(Yhat_test, Ytest)

    (
        cppls_model = cppls_model,
        cpplsfit = cpplsfit,
        train_idx = train_idx,
        test_idx = test_idx,
        Xtrain_tensor = Xtrain_tensor,
        Xtest_tensor = Xtest_tensor,
        Xtrain = Xtrain,
        Xtest = Xtest,
        Ytrain = Ytrain,
        Ytest = Ytest,
        Yaux_train = Yaux_train,
        Yaux_test = Yaux_test,
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
