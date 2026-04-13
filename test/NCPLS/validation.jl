if !isdefined(NCPLS, :synthetic_gcms_regression_data)
    validation_dir = normpath(joinpath(@__DIR__, "..", "..", "validation"))
    for file in (
        "synthetic_gcms.jl",
        "synthetic_multilinear.jl",
        "analysis_helpers.jl",
        "ncpls_analysis.jl",
        "multilinear_comparison.jl",
        "mode_weight_orthogonalization.jl",
        "multilinear_init_assessment.jl",
        "multilinear_convergence_assessment.jl",
        "multilinear_stress_assessment.jl",
        "yadd_effect.jl",
        "obs_weight_effect.jl",
    )
        Base.include(NCPLS, joinpath(validation_dir, file))
    end
end

import Random
import LinearAlgebra: norm
import Statistics: mean

@testset "synthetic_multilinear_regression_data covers vector, matrix, and tensor branches" begin
    data1 = NCPLS.synthetic_multilinear_regression_data(
        nsamples = 10,
        mode_dims = (12,),
        ncomponents = 2,
        nresponses = 1,
        y_noise_scale = 0.0,
        rng = Random.MersenneTwister(21),
    )

    @test size(data1.X) == (10, 12)
    @test size(data1.Yprim) == (10, 1)
    @test size(data1.T) == (10, 2)
    @test size(data1.Qtrue) == (2, 1)
    @test length(data1.mode_weights) == 1
    @test size(data1.mode_weights[1]) == (12, 2)
    @test data1.rt_mode === nothing
    @test data1.mz_mode == 1
    @test all(norm(data1.mode_weights[1][:, a]) ≈ 1.0 for a in 1:2)
    @test data1.Yprim ≈ data1.Yclean

    data2 = NCPLS.synthetic_multilinear_regression_data(
        nsamples = 8,
        mode_dims = (9, 7),
        ncomponents = 3,
        nresponses = 2,
        rng = Random.MersenneTwister(22),
    )

    @test size(data2.X) == (8, 9, 7)
    @test size(data2.Yprim) == (8, 2)
    @test length(data2.mode_weights) == 2
    @test size(data2.templates) == (9, 7, 3)
    @test data2.rt_mode == 1
    @test data2.mz_mode == 2
    @test size(data2.rt_peak_centers) == (2, 3)
    @test length(data2.active_mz_channels) == 3
    @test all(ch -> all(1 .≤ ch .≤ 7), data2.active_mz_channels)

    data3 = NCPLS.synthetic_multilinear_regression_data(
        nsamples = 6,
        mode_dims = (7, 5, 4),
        ncomponents = 2,
        nresponses = 1,
        integer_counts = false,
        rng = Random.MersenneTwister(23),
    )

    @test size(data3.X) == (6, 7, 5, 4)
    @test eltype(data3.X) == Float64
    @test size(data3.lambda) == (6, 7, 5, 4)
    @test length(data3.mode_weights) == 3
    @test data3.rt_mode == 1
    @test data3.mz_mode == 3
end

@testset "synthetic_multilinear_regression_data validates settings" begin
    @test_throws ArgumentError NCPLS.synthetic_multilinear_regression_data(mode_dims = ())
    @test_throws ArgumentError NCPLS.synthetic_multilinear_regression_data(mode_dims = (1, 5))
    @test_throws ArgumentError NCPLS.synthetic_multilinear_regression_data(ncomponents = 0)
    @test_throws ArgumentError NCPLS.synthetic_multilinear_regression_data(nresponses = 0)
    @test_throws ArgumentError NCPLS.synthetic_multilinear_regression_data(component_strength_scale = 0.0)
    @test_throws ArgumentError NCPLS.synthetic_multilinear_regression_data(mz_mode = 3, mode_dims = (5, 4))
    @test_throws ArgumentError NCPLS.synthetic_multilinear_regression_data(rt_mode = 2, mz_mode = 2, mode_dims = (5, 4))
    @test_throws ArgumentError NCPLS.synthetic_multilinear_regression_data(predictive_components = Int[])
    @test_throws ArgumentError NCPLS.synthetic_multilinear_regression_data(ncomponents = 2, predictive_components = [3])
end

@testset "synthetic_multilinear_yadd_data returns aligned Yprim and Yadd" begin
    data = NCPLS.synthetic_multilinear_yadd_data(
        nsamples = 16,
        mode_dims = (10, 8),
        ncomponents = 2,
        nadditional = 2,
        rng = Random.MersenneTwister(26),
    )

    @test size(data.X) == (16, 10, 8)
    @test size(data.Yprim) == (16, 1)
    @test size(data.Yadd) == (16, 2)
    @test size(data.T) == (16, 2)
    @test data.predictive_component == 1
    @test length(data.mode_weights) == 2
    @test length(data.active_mz_channels) == 2
    @test all(data.X .>= 0)
end

@testset "compare_synthetic_yadd_effect returns with/without Yadd summaries" begin
    data = NCPLS.synthetic_multilinear_yadd_data(
        nsamples = 40,
        mode_dims = (12, 10),
        ncomponents = 2,
        nadditional = 2,
        rng = Random.MersenneTwister(27),
    )

    result = NCPLS.compare_synthetic_yadd_effect(
        data;
        multilinear = true,
        test_fraction = 0.25,
        rng = Random.MersenneTwister(28),
    )

    @test result.train_idx == result.without_yadd.train_idx
    @test result.train_idx == result.with_yadd.train_idx
    @test result.test_idx == result.without_yadd.test_idx
    @test result.test_idx == result.with_yadd.test_idx
    @test result.without_yadd.use_yadd === false
    @test result.with_yadd.use_yadd === true
    @test length(result.without_mode_abs_cor) == 2
    @test length(result.with_mode_abs_cor) == 2
    @test length(result.true_active_mz_channels) == length(result.with_recovered_top_mz)
    @test 0 <= result.without_mz_overlap <= length(result.true_active_mz_channels)
    @test 0 <= result.with_mz_overlap <= length(result.true_active_mz_channels)
    @test result.better_model_test_rmse in (:without_yadd, :with_yadd, :tie)
    @test result.better_model_test_r2 in (:without_yadd, :with_yadd, :tie)
end

@testset "synthetic_gcms_regression_data returns integer GC-MS tensors and aligned targets" begin
    data = NCPLS.synthetic_gcms_regression_data(
        nsamples = 8,
        n_rt = 12,
        n_mz = 10,
        n_compounds = 4,
        target_compounds = [1, 3],
        additional_compounds = [2],
        baseline = 0.5,
        rng = Random.MersenneTwister(1),
    )

    @test size(data.X) == (8, 12, 10)
    @test eltype(data.X) == Int
    @test all(data.X .>= 0)
    @test size(data.Yprim) == (8, 2)
    @test size(data.Yadd) == (8, 1)
    @test size(data.concentrations) == (8, 4)
    @test size(data.rt_profiles) == (12, 4)
    @test size(data.mz_patterns) == (10, 4)
    @test size(data.templates) == (12, 10, 4)
    @test size(data.lambda) == (8, 12, 10)

    @test data.Yprim ≈ data.concentrations[:, [1, 3]]
    @test data.Yadd ≈ data.concentrations[:, [2]]
    @test data.target_compounds == [1, 3]
    @test data.additional_compounds == [2]

    for k in 1:4
        @test data.templates[:, :, k] ≈ data.rt_profiles[:, k] * data.mz_patterns[:, k]'
        @test maximum(data.rt_profiles[:, k]) ≈ 1.0
        @test maximum(data.mz_patterns[:, k]) ≈ 1.0
    end
end

@testset "synthetic_gcms_regression_data validates compound selections" begin
    @test_throws ArgumentError NCPLS.synthetic_gcms_regression_data(target_compounds = Int[])
    @test_throws ArgumentError NCPLS.synthetic_gcms_regression_data(target_compounds = [1], additional_compounds = [1])
    @test_throws ArgumentError NCPLS.synthetic_gcms_regression_data(n_compounds = 2, target_compounds = [3])
    @test_throws ArgumentError NCPLS.synthetic_gcms_regression_data(n_compounds = 2, additional_compounds = [3])
end

@testset "analyze_synthetic_gcms_with_ncpls fits NCPLS and returns regression diagnostics" begin
    data = NCPLS.synthetic_gcms_regression_data(
        nsamples = 12,
        n_rt = 8,
        n_mz = 6,
        n_compounds = 4,
        target_compounds = [1, 3],
        additional_compounds = [2],
        baseline = 0.5,
        rng = Random.MersenneTwister(5),
    )

    result = NCPLS.analyze_synthetic_gcms_with_ncpls(
        data;
        ncomponents = 2,
        test_fraction = 0.25,
        rng = Random.MersenneTwister(6),
    )

    @test result.ncplsfit isa NCPLS.NCPLSFit
    @test result.ncpls_model.ncomponents == 2
    @test result.ncpls_model.multilinear === true
    @test sort(vcat(result.train_idx, result.test_idx)) == collect(1:size(data.X, 1))
    @test isempty(intersect(result.train_idx, result.test_idx))

    @test size(result.Xtrain) == (length(result.train_idx), size(data.X, 2), size(data.X, 3))
    @test size(result.Xtest) == (length(result.test_idx), size(data.X, 2), size(data.X, 3))
    @test result.Ytrain ≈ data.Yprim[result.train_idx, :]
    @test result.Ytest ≈ data.Yprim[result.test_idx, :]
    @test result.Yadd_train ≈ data.Yadd[result.train_idx, :]
    @test result.Yadd_test ≈ data.Yadd[result.test_idx, :]
    @test result.predictorlabels[1] == "rt1_mz1"
    @test result.predictorlabels[end] == "rt8_mz6"
    @test result.responselabels == ["compound_1", "compound_3"]

    @test size(result.scores_train) == (length(result.train_idx), 2)
    @test size(result.scores_test) == (length(result.test_idx), 2)
    @test size(result.Yhat_train) == (length(result.train_idx), size(data.Yprim, 2), 2)
    @test size(result.Yhat_test) == (length(result.test_idx), size(data.Yprim, 2), 2)
    @test size(result.rmse_train) == (size(data.Yprim, 2), 2)
    @test size(result.rmse_test) == (size(data.Yprim, 2), 2)
    @test size(result.r2_train) == (size(data.Yprim, 2), 2)
    @test size(result.r2_test) == (size(data.Yprim, 2), 2)
    @test length(result.rmse_train_overall) == 2
    @test length(result.rmse_test_overall) == 2
    @test length(result.r2_train_overall) == 2
    @test length(result.r2_test_overall) == 2
    @test all(isfinite, result.rmse_train)
    @test all(isfinite, result.rmse_test)
end

@testset "analyze_synthetic_gcms_with_ncpls supports multilinear fits" begin
    data = NCPLS.synthetic_gcms_regression_data(
        nsamples = 10,
        n_rt = 7,
        n_mz = 5,
        n_compounds = 4,
        target_compounds = [1],
        additional_compounds = [2],
        rng = Random.MersenneTwister(7),
    )

    result = NCPLS.analyze_synthetic_gcms_with_ncpls(
        data;
        ncomponents = 1,
        multilinear = true,
        test_fraction = 0.2,
        rng = Random.MersenneTwister(8),
    )

    @test result.ncplsfit.W_modes isa AbstractVector
    @test result.ncplsfit.model.multilinear === true
end

@testset "compare_synthetic_multilinear_models returns prediction and structure summaries" begin
    data = NCPLS.synthetic_multilinear_regression_data(
        nsamples = 24,
        mode_dims = (12, 10),
        ncomponents = 1,
        nresponses = 1,
        rng = Random.MersenneTwister(24),
    )

    result = NCPLS.compare_synthetic_multilinear_models(
        data;
        ncomponents = 1,
        test_fraction = 0.25,
        rng = Random.MersenneTwister(25),
    )

    @test result.common_ncomponents == 1
    @test result.train_idx == result.unfolded.train_idx
    @test result.train_idx == result.multilinear.train_idx
    @test result.test_idx == result.unfolded.test_idx
    @test result.test_idx == result.multilinear.test_idx
    @test length(result.prediction_rmse_test) == 1
    @test length(result.unfolded_true_score_abs_cor_test) == 1
    @test length(result.multilinear_true_score_abs_cor_test) == 1
    @test size(result.multilinear_mode_abs_cor) == (2, 1)
    @test length(result.true_active_mz_channels) == 1
    @test length(result.multilinear_recovered_top_mz) == 1
    @test length(result.multilinear_recovered_top_mz[1]) == length(result.true_active_mz_channels[1])
    @test 0 <= result.multilinear_mz_overlap[1] <= length(result.true_active_mz_channels[1])
    @test result.better_model_test_rmse[1] in (:unfolded, :multilinear, :tie)
    @test result.better_model_test_r2[1] in (:unfolded, :multilinear, :tie)
end

@testset "synthetic_mode_orthogonality_data distinguishes orthogonal and overlapping truth" begin
    orth = NCPLS.synthetic_mode_orthogonality_data(
        nsamples = 20,
        mode_dims = (12, 10),
        orthogonal_truth = true,
        rng = Random.MersenneTwister(29),
    )
    overlap = NCPLS.synthetic_mode_orthogonality_data(
        nsamples = 20,
        mode_dims = (12, 10),
        orthogonal_truth = false,
        rng = Random.MersenneTwister(30),
    )

    @test size(orth.X) == (20, 12, 10)
    @test size(orth.Yprim) == (20, 1)
    @test orth.orthogonal_truth === true
    @test overlap.orthogonal_truth === false
    @test length(orth.mode_weights) == 2
    @test length(overlap.mode_weights) == 2
    @test all(ip -> ip ≤ 1e-12, orth.true_mode_abs_inner_products)
    @test any(ip -> ip > 0.1, overlap.true_mode_abs_inner_products)
    @test length(orth.active_mz_channels) == 2
    @test length(overlap.active_mz_channels) == 2
end

@testset "compare_orthogonalize_mode_weights summarizes prediction and mode orthogonality" begin
    data = NCPLS.synthetic_mode_orthogonality_data(
        nsamples = 48,
        mode_dims = (16, 12),
        orthogonal_truth = false,
        rng = Random.MersenneTwister(31),
    )

    result = NCPLS.compare_orthogonalize_mode_weights(
        data;
        test_fraction = 0.25,
        rng = Random.MersenneTwister(32),
    )

    @test result.train_idx == result.without_orthogonalization.train_idx
    @test result.train_idx == result.with_orthogonalization.train_idx
    @test result.test_idx == result.without_orthogonalization.test_idx
    @test result.test_idx == result.with_orthogonalization.test_idx
    @test length(result.true_mode_abs_inner_products) == length(data.mode_weights)
    @test length(result.without_mode_abs_inner_products) == length(data.mode_weights)
    @test length(result.with_mode_abs_inner_products) == length(data.mode_weights)
    @test length(result.without_component_order) == 2
    @test length(result.with_component_order) == 2
    @test size(result.without_mode_abs_cor) == (length(data.mode_weights), 2)
    @test size(result.with_mode_abs_cor) == (length(data.mode_weights), 2)
    @test length(result.without_true_score_abs_cor_test) == 2
    @test length(result.with_true_score_abs_cor_test) == 2
    @test length(result.without_recovered_top_mz) == 2
    @test length(result.with_recovered_top_mz) == 2
    @test all(ip -> ip ≤ 1e-10, result.with_mode_abs_inner_products)
    @test maximum(result.with_mode_abs_inner_products) ≤ maximum(result.without_mode_abs_inner_products)
    @test all(w -> w in (:without_orthogonalization, :with_orthogonalization, :tie),
        result.better_model_test_rmse)
    @test all(w -> w in (:without_orthogonalization, :with_orthogonalization, :tie),
        result.better_model_test_r2)
end

@testset "compare_multilinear_init summarizes HOSVD versus random starts" begin
    data = NCPLS.synthetic_multilinear_regression_data(
        nsamples = 36,
        mode_dims = (10, 8, 4),
        ncomponents = 2,
        nresponses = 1,
        integer_counts = false,
        rng = Random.MersenneTwister(33),
    )

    result = NCPLS.compare_multilinear_init(
        data;
        ncomponents = 1,
        random_seeds = [1, 2, 3],
        test_fraction = 0.25,
        rng = Random.MersenneTwister(34),
    )

    @test length(result.random) == 3
    @test result.random_seeds == [1, 2, 3]
    @test length(result.hosvd.rmse_test_overall) == 1
    @test length(result.hosvd.true_score_abs_cor_test) == 1
    @test size(result.hosvd.mode_abs_cor) == (3, 1)
    @test length(result.hosvd.recovered_top_mz) == 1
    @test length(result.random_test_rmse_mean) == 1
    @test length(result.random_test_rmse_std) == 1
    @test length(result.random_test_r2_mean) == 1
    @test length(result.random_test_r2_std) == 1
    @test length(result.random_score_cor_test_mean) == 1
    @test length(result.random_score_cor_test_std) == 1
    @test size(dropdims(result.random_mode_cor_mean; dims = 3)) == (3, 1)
    @test size(dropdims(result.random_mode_cor_std; dims = 3)) == (3, 1)
    @test length(result.random_pred_rmse_vs_hosvd) == 3
    @test length(result.random_score_abs_cor_vs_hosvd) == 3
    @test all(length(v) == 1 for v in result.random_pred_rmse_vs_hosvd)
    @test all(length(v) == 1 for v in result.random_score_abs_cor_vs_hosvd)
    @test_throws ArgumentError NCPLS.compare_multilinear_init(
        NCPLS.synthetic_multilinear_regression_data(
            nsamples = 20,
            mode_dims = (12, 10),
            ncomponents = 1,
            nresponses = 1,
            rng = Random.MersenneTwister(35),
        );
        ncomponents = 1,
        random_seeds = [1, 2],
    )
end

@testset "compare_multilinear_convergence summarizes tolerance and iteration settings" begin
    data = NCPLS.synthetic_multilinear_regression_data(
        nsamples = 32,
        mode_dims = (10, 8, 4),
        ncomponents = 2,
        nresponses = 1,
        integer_counts = false,
        rng = Random.MersenneTwister(36),
    )

    result = NCPLS.compare_multilinear_convergence(
        data;
        ncomponents = 1,
        settings = [
            (maxiter = 2, tol = 1e-1),
            (maxiter = 20, tol = 1e-6),
        ],
        reference_setting = (maxiter = 50, tol = 1e-10),
        multilinear_init = :random,
        multilinear_seed = 7,
        test_fraction = 0.25,
        rng = Random.MersenneTwister(37),
    )

    @test length(result.reference.rmse_test_overall) == 1
    @test length(result.reference.true_score_abs_cor_test) == 1
    @test size(result.reference.mode_abs_cor) == (3, 1)
    @test length(result.reference.recovered_top_mz) == 1
    @test length(result.reference.parafac_relerr) == 1
    @test result.reference.multilinear_init == :random
    @test result.reference.multilinear_seed == 7
    @test length(result.settings) == 2
    @test all(hasproperty(s, :pred_rmse_vs_reference) for s in result.settings)
    @test all(hasproperty(s, :score_abs_cor_vs_reference) for s in result.settings)
    @test all(length(s.rmse_test_overall) == 1 for s in result.settings)
    @test all(length(s.true_score_abs_cor_test) == 1 for s in result.settings)
    @test all(size(s.mode_abs_cor) == (3, 1) for s in result.settings)
    @test all(length(s.parafac_relerr) == 1 for s in result.settings)
    @test length(result.rmse_test_mean) == 1
    @test length(result.rmse_test_std) == 1
    @test length(result.r2_test_mean) == 1
    @test length(result.r2_test_std) == 1
    @test length(result.score_cor_test_mean) == 1
    @test length(result.score_cor_test_std) == 1
    @test size(dropdims(result.mode_cor_mean; dims = 3)) == (3, 1)
    @test size(dropdims(result.mode_cor_std; dims = 3)) == (3, 1)
    @test_throws ArgumentError NCPLS.compare_multilinear_convergence(
        data;
        ncomponents = 1,
        settings = NamedTuple[],
    )
    @test_throws ArgumentError NCPLS.compare_multilinear_convergence(
        data;
        ncomponents = 1,
        settings = [(maxiter = 0, tol = 1e-2)],
    )
end

@testset "synthetic_obs_weighted_multilinear_data returns weighted heteroskedastic samples" begin
    data = NCPLS.synthetic_obs_weighted_multilinear_data(
        nsamples = 30,
        mode_dims = (12, 10),
        ncomponents = 2,
        nresponses = 1,
        noisy_fraction = 0.3,
        rng = Random.MersenneTwister(39),
    )

    @test size(data.X) == (30, 12, 10)
    @test size(data.Yprim) == (30, 1)
    @test length(data.obs_weights) == 30
    @test isapprox(mean(data.obs_weights), 1.0; atol = 1e-12)
    @test count(data.noisy_mask) == length(data.noisy_idx)
    @test count(.!data.noisy_mask) == length(data.clean_idx)
    @test sort(vcat(data.clean_idx, data.noisy_idx)) == collect(1:30)
    @test all(data.obs_weights[data.clean_idx] .> minimum(data.obs_weights[data.noisy_idx]))
end

@testset "compare_obs_weights_effect summarizes weighted versus unweighted NCPLS" begin
    data = NCPLS.synthetic_obs_weighted_multilinear_data(
        nsamples = 80,
        mode_dims = (14, 10),
        ncomponents = 2,
        nresponses = 1,
        noisy_fraction = 0.35,
        x_noise_scale_clean = 0.02,
        x_noise_scale_noisy = 0.50,
        y_noise_scale_clean = 0.02,
        y_noise_scale_noisy = 0.40,
        rng = Random.MersenneTwister(40),
    )

    result = NCPLS.compare_obs_weights_effect(
        data;
        ncomponents = 1,
        multilinear = true,
        test_fraction = 0.25,
        rng = Random.MersenneTwister(41),
    )

    @test result.train_idx == result.without_weights.train_idx
    @test result.train_idx == result.with_weights.train_idx
    @test result.test_idx == result.without_weights.test_idx
    @test result.test_idx == result.with_weights.test_idx
    @test result.common_ncomponents == 1
    @test length(result.without_true_score_abs_cor_test) == 1
    @test length(result.with_true_score_abs_cor_test) == 1
    @test size(result.without_mode_abs_cor) == (2, 1)
    @test size(result.with_mode_abs_cor) == (2, 1)
    @test length(result.without_recovered_top_mz) == 1
    @test length(result.with_recovered_top_mz) == 1
    @test length(result.without_weights.weighted_rmse_test_overall) == 1
    @test length(result.with_weights.weighted_rmse_test_overall) == 1
    @test length(result.without_weights.clean_rmse_test_overall) == 1
    @test length(result.with_weights.noisy_rmse_test_overall) == 1
    @test result.clean_test_count + result.noisy_test_count == length(result.test_idx)
    @test result.better_model_test_rmse[1] in (:without_weights, :with_weights, :tie)
    @test result.better_model_weighted_test_rmse[1] in (:without_weights, :with_weights, :tie)
    @test result.better_model_test_r2[1] in (:without_weights, :with_weights, :tie)
    @test result.better_model_weighted_test_r2[1] in (:without_weights, :with_weights, :tie)
    @test result.with_weights.weighted_rmse_test_overall[1] ≤
        result.without_weights.weighted_rmse_test_overall[1] + 1e-10
end

@testset "assess_multilinear_stress summarizes init and convergence under harder regimes" begin
    result = NCPLS.assess_multilinear_stress(
        nsamples = 28,
        mode_dims = (10, 8, 4),
        ncomponents = 2,
        nresponses = 1,
        stress_settings = [
            (label = :mild, x_noise_scale = 0.10, y_noise_scale = 0.10, component_strength_scale = 1.0),
            (label = :hard, x_noise_scale = 0.35, y_noise_scale = 0.35, component_strength_scale = 0.60),
        ],
        random_seeds = [1, 2],
        convergence_settings = [
            (maxiter = 2, tol = 1e-1),
            (maxiter = 20, tol = 1e-6),
        ],
        reference_setting = (maxiter = 50, tol = 1e-10),
        integer_counts = false,
        test_fraction = 0.25,
        rng = Random.MersenneTwister(38),
    )

    @test result.mode_dims == (10, 8, 4)
    @test result.nsamples == 28
    @test length(result.cases) == 2
    @test result.cases[1].label == :mild
    @test result.cases[2].label == :hard
    @test length(result.cases[1].init.hosvd_rmse_test) == 1
    @test length(result.cases[1].init.random_rmse_test_mean) == 1
    @test size(result.cases[1].init.hosvd_mode_cor) == (3, 1)
    @test size(result.cases[1].init.random_mode_cor_mean) == (3, 1)
    @test length(result.cases[1].init.random_pred_rmse_vs_hosvd_max) == 1
    @test length(result.cases[1].init.random_score_abs_cor_vs_hosvd_min) == 1
    @test length(result.cases[1].convergence.reference_rmse_test) == 1
    @test length(result.cases[1].convergence.reference_parafac_relerr) == 1
    @test length(result.cases[1].convergence.settings) == 2
    @test all(hasproperty(s, :pred_rmse_vs_reference) for s in result.cases[1].convergence.settings)
    @test all(hasproperty(s, :score_abs_cor_vs_reference) for s in result.cases[1].convergence.settings)
    @test_throws ArgumentError NCPLS.assess_multilinear_stress(
        mode_dims = (12, 10),
    )
    @test_throws ArgumentError NCPLS.assess_multilinear_stress(
        stress_settings = [(label = :bad, x_noise_scale = -0.1, y_noise_scale = 0.1, component_strength_scale = 1.0)],
    )
end
