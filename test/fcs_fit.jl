@testset "fcs_fit" begin

    @testset "FCSFitResult constructor + _to_lfr + StatsAPI shims" begin
        exp_model(x, θ) = @. θ[1] * exp(-x/θ[2]) + θ[3]

        a_true, b_true, c_true = 1.25, 3.2, 0.08
        x = collect(range(0.01, 8.0; length=300))
        σ = @. 0.01 + 0.01 * (x / maximum(x))
        y = exp_model(x, [a_true, b_true, c_true]) .+ σ .* randn(length(x))

        wt = @. 1 / σ^2
        θ0 = [1.0, 1.0, 0.0]
        lsf = curve_fit(exp_model, x, y, wt, θ0)

        spec_free = FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1, offset=nothing)

        scales = [2.0, 10.0, 0.5]
        ffr = FCSFitResult(lsf, spec_free, scales)

        lsf2 = FCSFitting._to_lfr(ffr)
        @test lsf2.param == lsf.param
        @test lsf2.converged == lsf.converged
        @test lsf2.wt == lsf.wt

        # StatsAPI methods
        @test StatsAPI.nobs(ffr) == length(y)
        @test StatsAPI.residuals(ffr) === ffr.resid
        @test StatsAPI.weights(ffr) === wt
        @test StatsAPI.rss(ffr) ≈ sum(abs2, ffr.resid)
        @test StatsAPI.dof(ffr) == (length(y) - length(lsf.param))

        # coefficients in physical space
        @test StatsAPI.coef(ffr) ≈ (lsf.param .* scales)

        # standard errors in physical space
        @test all(isfinite, StatsAPI.stderror(ffr))
        @test StatsAPI.stderror(ffr) ≈ (LsqFit.stderror(lsf) .* scales)

        # mse and convergence helpers
        @test FCSFitting.mse(ffr) ≈ StatsAPI.rss(ffr) / StatsAPI.dof(ffr)
        @test FCSFitting.isconverged(ffr) == lsf.converged

        # loglikelihood sanity: less residuals → higher LL
        # Build a "better" fit by re-fitting with the true parameters as θ0
        lsf_better = curve_fit(exp_model, x, y, wt, [a_true, b_true, c_true])
        ffr_better = FCSFitResult(lsf_better, spec_free, scales)
        @test StatsAPI.loglikelihood(ffr_better) > StatsAPI.loglikelihood(ffr)

        fixed_off = -0.0123
        spec_fixed = FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1, offset=fixed_off)
        ffr_fixed = FCSFitResult(lsf, spec_fixed, scales)
        @test StatsAPI.offset(ffr_fixed) == fixed_off
    end

    @testset "build_scales_from_p0 + build_scales" begin
        # p0 with zeros and a "noscale" index
        p0 = [1.0, 0.0, 1e-3, 0.5, 0.0]
        # e.g. last entry is a K_dyn that must not be scaled
        noscale = [5]
        θ0, s = FCSFitting.build_scales_from_p0(p0; noscale_idx=noscale, zero_sub=2.0)

        @test s ≈ [1.0, 2.0, 1e-3, 0.5, 1.0]
        @test θ0 ≈ [1.0, 0.0, 1.0, 1.0, 0.0]

        # wrapper does no protection → identical to noscale_idx=[]
        θ1, s1 = FCSFitting.build_scales(p0; zero_sub=2.0)
        @test s1 ≈ [1.0, 2.0, 1e-3, 0.5, 2.0]
        @test θ1 ≈ [1.0, 0.0, 1.0, 1.0, 0.0]
    end

    @testset "infer_noscale_indices (simple 2D case)" begin
        # 2D, normal diffusion, ONE diffuser, FREE offset (offset=nothing)
        # Parameter layout (n=1, m=1): [g0, offset, τD, τ_dyn, K_dyn]
        spec = FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1, offset=nothing)
        p0 = [1.0, 0.0, 1e-3, 1e-6, 0.2]
        idxs = FCSFitting.infer_noscale_indices(spec, p0)
        # No mixture weights (n=1); K_dyn is the last position → index 5
        @test idxs == [5]
    end

    @testset "infer_noscale_indices (3D, two diffusers, weights + dynamics)" begin
        # 3D, Brownian (anom=none), TWO diffusers, FIXED offset
        # Parameter layout (n=2, m=1, fixed offset):
        #   [g0, κ, τD1, τD2, w1, τ_dyn, K_dyn]
        spec = FCSModelSpec(; dim=FCSFitting.d3, anom=FCSFitting.none, n_diff=2, offset=0.0)
        p0 = [1.0, 8.0, 1e-3, 2e-4, 0.3, 5e-6, 0.1]
        idxs = FCSFitting.infer_noscale_indices(spec, p0)
        # Expect: weight w1 (index 5) and K_dyn (index 7) to be "noscale"
        @test idxs == [5, 7]
    end

    @testset "fitting (2D Brownian, free offset)" begin
        # Synthetic data
        τ = 10 .^ range(-6, -1; length=300)
        g0_true = rand()
        offset_true = 1e-3 * randn()
        τD_true = 1e-3 * rand()

        # Model spec: 2D, Brownian, single diffuser, free offset
        spec = FCSFitting.FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1)

        # Generate noiseless data with the generalized model
        model = FCSFitting.FCSModel(; spec)
        y = model(τ, [g0_true, offset_true, τD_true])

        # --- Fit without bounds
        p0 = [0.5, 0.0, 5e-4]   # rough initial guesses
        fit = FCSFitting.fcs_fit(spec, τ, y, p0)
        p̂ = coef(fit)

        @test p̂[1] ≈ g0_true rtol=1e-5
        @test p̂[2] ≈ offset_true rtol=1e-5
        @test p̂[3] ≈ τD_true rtol=1e-5

        # --- Fit with bounds (make g0 lower bound larger than truth)
        p02 = [1.5, 0.0, 5e-4]
        lower = [1.0, -1e-1, 0.0]
        upper = [2.0, 1e-1, 1e-2]
        fit = FCSFitting.fcs_fit(spec, τ, y, p02; lower=lower, upper=upper)
        p̂ = coef(fit)

        @test p̂[1] ≥ lower[1] && p̂[1] ≤ upper[1]
        @test p̂[2] ≈ offset_true atol=0.1
        @test p̂[3] ≈ τD_true atol=2τD_true
    end

    @testset "fitting (3D Brownian, 2 diffusive components, with dynamics)" begin
        τ = 10 .^ range(-9, -1; length=500)
        g0_true = rand()
        κ_true = 10 * rand()
        τD1_true = 1e-3 * rand()
        τD2_true = 1e-4 * rand()
        wt1_true = rand()
        τdyn_true = 1e-6 * rand()
        Kdyn_true = 0.5 * rand()

        # Model spec: 3D, Brownian, two diffusers, fixed offset
        spec = FCSFitting.FCSModelSpec(; dim=FCSFitting.d3, anom=FCSFitting.none, n_diff=2, offset=0.0)

        model = FCSFitting.FCSModel(; spec)
        y = model(τ, [g0_true, κ_true, τD1_true, τD2_true, wt1_true, τdyn_true, Kdyn_true])

        p0 = [0.5, 5, 5e-4, 5e-5, 0.5, 5e-7, 0.25]
        fit = FCSFitting.fcs_fit(spec, τ, y, p0)
        p̂ = coef(fit)

        @test p̂[1] ≈ g0_true atol=1e-2*g0_true
        @test p̂[2] ≈ κ_true atol=1e-2*κ_true
        @test p̂[3] ≈ τD1_true atol=1e-2*τD1_true
        @test p̂[4] ≈ τD2_true atol=1e-2*τD2_true
        @test p̂[5] ≈ wt1_true atol=1e-2*wt1_true
        @test p̂[6] ≈ τdyn_true atol=1e-2*τdyn_true
        @test p̂[7] ≈ Kdyn_true atol=1e-1*Kdyn_true
    end

    @testset "weights vs σ equivalence (heteroscedastic)" begin
        τ = 10 .^ range(-6, -1; length=300)
        g0_true = 0.9
        offset_true = 2e-3
        τD_true = 8e-4

        spec = FCSFitting.FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1)
        model = FCSFitting.FCSModel(; spec)
        y_true = model(τ, [g0_true, offset_true, τD_true])

        # Heteroscedastic noise
        σ = @. 0.002 + 0.003 * (τ / maximum(τ))
        y = y_true .+ σ .* randn(length(τ))

        p0 = [1.0, 0.0, 1e-3]

        # Using σ (internally converted to 1/σ²)
        fitA = FCSFitting.fcs_fit(spec, τ, y, p0; σ=σ)
        pA = coef(fitA)

        # Using explicit weights
        wt = @. 1 / σ^2
        fitB = FCSFitting.fcs_fit(spec, τ, y, p0; wt=wt)
        pB = pA = coef(fitB)

        @test pA ≈ pB rtol=1e-4
    end


    @testset "fixed diffusivity (w₀ fitted)" begin
        τ = 10 .^ range(-6, -1; length=300)

        # Truth
        D = 5e-11
        w0_true = 5e-7 * rand()
        τD_true = FCSFitting.τD(D, w0_true)
        g0_true = rand()
        off_fixed = 0.0

        # Spec: 2D, Brownian, single diffuser, fixed offset & fixed D (so p = [g0, w0])
        spec = FCSFitting.FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1, offset=off_fixed, diffusivity=D)
        model = FCSFitting.FCSModel(; spec)
        y = model(τ, [g0_true, w0_true])

        # Initial guesses & bounds
        p0 = [0.001, 2e-9]
        lower = [0.0, 0.0]
        upper = [1.0, 500e-9]

        fit = FCSFitting.fcs_fit(spec, τ, y, p0; lower=lower, upper=upper)
        p̂ = coef(fit)
        g0o, w0o = p̂

        @test g0o ≈ g0_true rtol=1e-4
        @test w0o ≈ w0_true rtol=1e-4
        @test FCSFitting.τD(D, w0o) ≈ τD_true rtol=1e-4
    end
end