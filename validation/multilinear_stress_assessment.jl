"""
    assess_multilinear_stress(;
        nsamples::Integer=80,
        mode_dims::Tuple{Vararg{Int}}=(30, 24, 6),
        ncomponents::Integer=2,
        nresponses::Integer=1,
        stress_settings::AbstractVector{<:NamedTuple}=[
            (label = :mild, x_noise_scale = 0.10, y_noise_scale = 0.10, component_strength_scale = 1.0),
            (label = :moderate, x_noise_scale = 0.25, y_noise_scale = 0.25, component_strength_scale = 0.75),
            (label = :hard, x_noise_scale = 0.40, y_noise_scale = 0.40, component_strength_scale = 0.50),
        ],
        random_seeds::AbstractVector{<:Integer}=[1, 2, 3, 4, 5],
        convergence_settings::AbstractVector{<:NamedTuple}=[
            (maxiter = 2, tol = 1e-1),
            (maxiter = 10, tol = 1e-3),
            (maxiter = 50, tol = 1e-6),
        ],
        reference_setting::NamedTuple=(maxiter = 500, tol = 1e-10),
        integer_counts::Bool=false,
        test_fraction::Real=0.25,
        rng::AbstractRNG=MersenneTwister(1),
        verbose::Bool=false,
    )

Assess multilinear NCPLS under progressively stronger noise and weaker predictor signal.

For each stress setting, this utility generates synthetic `d ≥ 3` multilinear data,
compares `multilinear_init = :hosvd` against repeated `:random` starts, and compares
looser PARAFAC convergence settings against a stricter reference fit. The returned
object is intentionally compact so that stress trends can be inspected directly.
"""
function assess_multilinear_stress(;
    nsamples::Integer=80,
    mode_dims::Tuple{Vararg{Int}}=(30, 24, 6),
    ncomponents::Integer=2,
    nresponses::Integer=1,
    stress_settings::AbstractVector{<:NamedTuple}=[
        (label = :mild, x_noise_scale = 0.10, y_noise_scale = 0.10, component_strength_scale = 1.0),
        (label = :moderate, x_noise_scale = 0.25, y_noise_scale = 0.25, component_strength_scale = 0.75),
        (label = :hard, x_noise_scale = 0.40, y_noise_scale = 0.40, component_strength_scale = 0.50),
    ],
    random_seeds::AbstractVector{<:Integer}=[1, 2, 3, 4, 5],
    convergence_settings::AbstractVector{<:NamedTuple}=[
        (maxiter = 2, tol = 1e-1),
        (maxiter = 10, tol = 1e-3),
        (maxiter = 50, tol = 1e-6),
    ],
    reference_setting::NamedTuple=(maxiter = 500, tol = 1e-10),
    integer_counts::Bool=false,
    test_fraction::Real=0.25,
    rng::AbstractRNG=MersenneTwister(1),
    verbose::Bool=false,
)
    length(mode_dims) ≥ 3 || throw(ArgumentError(
        "assess_multilinear_stress is intended for d ≥ 3 predictor modes"))
    !isempty(stress_settings) || throw(ArgumentError("stress_settings must not be empty"))

    cases = Vector{NamedTuple}(undef, length(stress_settings))

    for i in eachindex(stress_settings)
        setting = stress_settings[i]
        validate_stress_setting(setting)

        data_seed = rand(rng, 1:typemax(Int))
        split_seed = rand(rng, 1:typemax(Int))

        data = synthetic_multilinear_regression_data(
            nsamples = nsamples,
            mode_dims = mode_dims,
            ncomponents = ncomponents,
            nresponses = nresponses,
            component_strength_scale = setting.component_strength_scale,
            integer_counts = integer_counts,
            x_noise_scale = setting.x_noise_scale,
            y_noise_scale = setting.y_noise_scale,
            rng = MersenneTwister(data_seed),
        )

        init_cmp = compare_multilinear_init(
            data;
            ncomponents = 1,
            random_seeds = random_seeds,
            test_fraction = test_fraction,
            rng = MersenneTwister(split_seed),
            verbose = verbose,
        )

        conv_cmp = compare_multilinear_convergence(
            data;
            ncomponents = 1,
            settings = convergence_settings,
            reference_setting = reference_setting,
            multilinear_init = :random,
            multilinear_seed = Int(first(random_seeds)),
            test_fraction = test_fraction,
            rng = MersenneTwister(split_seed),
            verbose = verbose,
        )

        cases[i] = (
            label = setting.label,
            x_noise_scale = Float64(setting.x_noise_scale),
            y_noise_scale = Float64(setting.y_noise_scale),
            component_strength_scale = Float64(setting.component_strength_scale),
            init = (
                hosvd_rmse_test = init_cmp.hosvd.rmse_test_overall,
                hosvd_r2_test = init_cmp.hosvd.r2_test_overall,
                hosvd_score_cor_test = init_cmp.hosvd.true_score_abs_cor_test,
                hosvd_mode_cor = init_cmp.hosvd.mode_abs_cor,
                hosvd_mz_overlap = init_cmp.hosvd.mz_overlap,
                random_rmse_test_mean = init_cmp.random_test_rmse_mean,
                random_rmse_test_std = init_cmp.random_test_rmse_std,
                random_r2_test_mean = init_cmp.random_test_r2_mean,
                random_r2_test_std = init_cmp.random_test_r2_std,
                random_score_cor_test_mean = init_cmp.random_score_cor_test_mean,
                random_score_cor_test_std = init_cmp.random_score_cor_test_std,
                random_mode_cor_mean = dropdims(init_cmp.random_mode_cor_mean; dims = 3),
                random_mode_cor_std = dropdims(init_cmp.random_mode_cor_std; dims = 3),
                random_pred_rmse_vs_hosvd_max = componentwise_max(init_cmp.random_pred_rmse_vs_hosvd),
                random_score_abs_cor_vs_hosvd_min = componentwise_min(init_cmp.random_score_abs_cor_vs_hosvd),
            ),
            convergence = (
                reference_rmse_test = conv_cmp.reference.rmse_test_overall,
                reference_r2_test = conv_cmp.reference.r2_test_overall,
                reference_score_cor_test = conv_cmp.reference.true_score_abs_cor_test,
                reference_mode_cor = conv_cmp.reference.mode_abs_cor,
                reference_parafac_relerr = conv_cmp.reference.parafac_relerr,
                settings = [
                    (
                        maxiter = s.maxiter,
                        tol = s.tol,
                        rmse_test = s.rmse_test_overall,
                        r2_test = s.r2_test_overall,
                        score_cor_test = s.true_score_abs_cor_test,
                        mode_cor = s.mode_abs_cor,
                        parafac_relerr = s.parafac_relerr,
                        pred_rmse_vs_reference = s.pred_rmse_vs_reference,
                        score_abs_cor_vs_reference = s.score_abs_cor_vs_reference,
                    ) for s in conv_cmp.settings
                ],
            ),
        )
    end

    (
        mode_dims = mode_dims,
        nsamples = nsamples,
        cases = cases,
    )
end

function validate_stress_setting(setting::NamedTuple)
    hasproperty(setting, :label) || throw(ArgumentError(
        "Each stress setting must define label"))
    hasproperty(setting, :x_noise_scale) || throw(ArgumentError(
        "Each stress setting must define x_noise_scale"))
    hasproperty(setting, :y_noise_scale) || throw(ArgumentError(
        "Each stress setting must define y_noise_scale"))
    hasproperty(setting, :component_strength_scale) || throw(ArgumentError(
        "Each stress setting must define component_strength_scale"))
    setting.x_noise_scale ≥ 0 || throw(ArgumentError(
        "x_noise_scale must be non-negative"))
    setting.y_noise_scale ≥ 0 || throw(ArgumentError(
        "y_noise_scale must be non-negative"))
    setting.component_strength_scale > 0 || throw(ArgumentError(
        "component_strength_scale must be greater than zero"))
    nothing
end

function componentwise_max(vectors::AbstractVector{<:AbstractVector{<:Real}})
    vec(maximum(hcat(vectors...); dims = 2))
end

function componentwise_min(vectors::AbstractVector{<:AbstractVector{<:Real}})
    vec(minimum(hcat(vectors...); dims = 2))
end
