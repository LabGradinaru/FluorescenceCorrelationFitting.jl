@testset "fcs_fit" begin

    @testset "helpers" begin
        # log_lags
        l = log_lags(10, 0, 1000)
        @test all(0 .≤ l .≤ 1000)
        @test issorted(l) && length(unique(l)) == length(l)

        # Small range → fewer than requested points (no duplicates)
        l2 = log_lags(100, 10, 15)
        @test all(10 .≤ l2 .≤ 15)
        @test issorted(l2)
        @test length(l2) ≤ 6 # at most the distinct integers in [10,15]

        # Degenerate single-point range
        @test log_lags(5, 7, 7) == [7]

        # infer_noscale_indices
        p0 = [1.0, 0.0, 1e-3, 1e-5, 1e-6, 0.1, 0.2]
        @test FCSFitting.infer_noscale_indices(:fcs_2d, p0; offset=nothing) == [6,7]
        p0 = [1.0, 0.0, 1e-4, 1e-3, 1e-2, 0.2, 0.3, 1e-5, 0.1] # m=1, K at last index
        @test FCSFitting.infer_noscale_indices(:fcs_2d_mdiff, p0; n_diff=3, offset=nothing) == [6,7,9]
        p0 = [1.0, 0.0, 1e-4, 2e-3, 0.9, 1.0, 0.25, 1e-5, 0.1] # m=1 → K at 9
        @test FCSFitting.infer_noscale_indices(:fcs_2d_anom_mdiff, p0; n_diff=2, offset=nothing) == [7,9]
        p0 = [1.0, 0.0, 8.0, 1e-3, 1e-5, 0.1]
        @test FCSFitting.infer_noscale_indices(:fcs_3d, p0; offset=nothing) == [6]
        p0 = [1.0, 8.0, 1e-3, 1e-5, 0.1] # fcs_3d, fixed offset
        @test FCSFitting.infer_noscale_indices(:fcs_3d, p0; offset=0.0) == [5]
        p0 = [1.0, 0.0, 8.0, 1e-4, 2e-3, 0.25, 1e-5, 0.1] # n=2 → w at index 6, K at 8
        @test FCSFitting.infer_noscale_indices(:fcs_3d_mdiff, p0; n_diff=2, offset=nothing) == [6,8]

        # build_scales_from_p0
        p0 = [1.0, 0.0, 1e-3, 0.5]
        noscale = [4]
        θ0, s = FCSFitting.build_scales_from_p0(p0; noscale_idx=noscale, zero_sub=2.0)
        @test s ≈ [1.0, 2.0, 1e-3, 1.0]
        @test θ0 ≈ [1.0, 0.0, 1.0, 0.5]
    end


    @testset "fcs_fit" begin
        τ = collect(range(1e-6, 1e-2; length=400))
        g0_true = rand()
        offset_true = 1e-3 * randn()
        τD_true = 1e-3 * rand()
        y = fcs_2d(τ, [g0_true, offset_true, τD_true])

        # ensure correct fitting without bounds
        p0 = [0.5, 0.0, 5e-4] # mean initial guesses
        fit, sc = fcs_fit(fcs_2d, τ, y, p0)
        p̂ = fit.param .* sc

        @test p̂[1] ≈ g0_true rtol=1e-5
        @test p̂[2] ≈ offset_true rtol=1e-5
        @test p̂[3] ≈ τD_true rtol=1e-5

        # testing that bounds are respected but fitting results are still "reasonable"
        p02 = [1.5, 0.0, 5e-4]
        lower = [1.0, -1e-1, 0.0]
        upper = [2.0, 1e-1, 1e-2]  # g0 lower > true → should clamp near 1.0
        fit, sc = fcs_fit(fcs_2d, τ, y, p02; lower=lower, upper=upper)
        p̂ = fit.param .* sc

        @test p̂[1] ≈ lower[1] rtol=1e-3
        @test p̂[2] ≈ offset_true atol=0.1
        @test p̂[3] ≈ τD_true atol=2τD_true

        y_true = fcs_2d(τ, [g0_true, offset_true, τD_true])

        # testing that the weighting is correctly implemented for heteroscedastic noise
        σ = @. 0.002 + 0.003 * (τ / maximum(τ))
        y = y_true .+ σ .* randn(length(τ))

        p0 = [1.0, 0.0, 1e-3]
        fitA, scA = fcs_fit(fcs_2d, τ, y, p0; σ=σ)
        pA = fitA.param .* scA
        wt = @. 1 / σ^2
        fitB, scB = fcs_fit(fcs_2d, τ, y, p0; wt=wt)
        pB = fitB.param .* scB
        @test pA ≈ pB rtol=1e-4

        # test that we recover a reasonable τD when fitting with fixed diffusivity
        # Initial guesses far undershoot on average
        D = 5e-11
        w0_true = 5e-7 * rand()
        τD_true = τD(D, w0_true)
        g0_true = rand()
        offset_fixed = 0.0
        y = fcs_2d(τ, [g0_true, w0_true]; diffusivity=D, offset=offset_fixed)

        p0 = [0.001, 2e-9]
        lower = [0.0, 0.0]
        upper = [1.0, 500e-9]
        fit, sc = fcs_fit(fcs_2d, τ, y, p0; diffusivity=D, offset=offset_fixed, lower=lower, upper=upper)
        p̂_free = fit.param .* sc
        g0o, w0o = p̂_free

        @test g0o ≈ g0_true rtol=1e-4
        @test w0o≈ w0_true rtol=1e-4
        @test τD(D, w0o) ≈ τD_true rtol=1e-4
    end
end