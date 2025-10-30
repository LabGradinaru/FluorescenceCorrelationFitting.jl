@testset "fcs_fit" begin

    @testset "helpers" begin
        # ----------------------------
        # log_lags
        # ----------------------------
        l = log_lags(10, 0, 1000)
        @test all(0 .≤ l .≤ 1000)
        @test issorted(l) && length(unique(l)) == length(l)

        # Small range → fewer than requested points (no duplicates)
        l2 = log_lags(100, 10, 15)
        @test all(10 .≤ l2 .≤ 15)
        @test issorted(l2)
        @test length(l2) ≤ (15 - 10 + 1)

        # Degenerate single-point range
        @test log_lags(5, 7, 7) == [7]

        # ----------------------------
        # build_scales (new helper)
        # ----------------------------
        p0 = [1.0, 0.0, 1e-3, 0.5]
        θ0, s = FCSFitting.build_scales(p0; zero_sub=2.0)
        @test s ≈ [1.0, 2.0, 1e-3, 0.5]         # zero → zero_sub
        @test θ0 ≈ [1.0, 0.0, 1.0, 1.0]         # p ./ s
    end


    @testset "fitting (2D Brownian, free offset)" begin
        # Synthetic data
        τ = collect(range(1e-6, 1e-2; length=400))
        g0_true     = rand()
        offset_true = 1e-3 * randn()
        τD_true     = 1e-3 * rand()

        # Model spec: 2D, Brownian, single diffuser, free offset
        spec = FCSFitting.FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1, offset=nothing)

        # Generate noiseless data with the generalized model
        model = FCSFitting.FCSModel(; spec)
        y = model(τ, [g0_true, offset_true, τD_true])

        # --- Fit without bounds
        p0 = [0.5, 0.0, 5e-4]   # rough initial guesses
        fit, sc = FCSFitting.fcs_fit(spec, τ, y, p0)
        p̂ = fit.param .* sc

        @test p̂[1] ≈ g0_true     rtol=1e-5
        @test p̂[2] ≈ offset_true rtol=1e-5
        @test p̂[3] ≈ τD_true     rtol=1e-5

        # --- Fit with bounds (make g0 lower bound larger than truth)
        p02   = [1.5, 0.0, 5e-4]
        lower = [1.0, -1e-1, 0.0]
        upper = [2.0,  1e-1, 1e-2]
        fit, sc = FCSFitting.fcs_fit(spec, τ, y, p02; lower=lower, upper=upper)
        p̂ = fit.param .* sc

        @test p̂[1] ≥ lower[1] && p̂[1] ≤ upper[1]
        @test p̂[2] ≈ offset_true atol=0.1
        @test p̂[3] ≈ τD_true    atol=2τD_true
    end


    @testset "weights vs σ equivalence (heteroscedastic)" begin
        τ = collect(range(1e-6, 1e-2; length=400))
        g0_true     = 0.9
        offset_true = 2e-3
        τD_true     = 8e-4

        spec = FCSFitting.FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1, offset=nothing)
        model = FCSFitting.FCSModel(; spec)
        y_true = model(τ, [g0_true, offset_true, τD_true])

        # Heteroscedastic noise
        σ = @. 0.002 + 0.003 * (τ / maximum(τ))
        y = y_true .+ σ .* randn(length(τ))

        p0 = [1.0, 0.0, 1e-3]

        # Using σ (internally converted to 1/σ²)
        fitA, scA = FCSFitting.fcs_fit(spec, τ, y, p0; σ=σ)
        pA = fitA.param .* scA

        # Using explicit weights
        wt = @. 1 / σ^2
        fitB, scB = FCSFitting.fcs_fit(spec, τ, y, p0; wt=wt)
        pB = fitB.param .* scB

        @test pA ≈ pB rtol=1e-4
    end


    @testset "fixed diffusivity (w₀ fitted)" begin
        τ = collect(range(1e-6, 1e-2; length=400))

        # Truth
        D         = 5e-11
        w0_true   = 5e-7 * rand()
        τD_true   = FCSFitting.τD(D, w0_true)
        g0_true   = rand()
        off_fixed = 0.0

        # Spec: 2D, Brownian, single diffuser, fixed offset & fixed D (so p = [g0, w0])
        spec = FCSFitting.FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1, offset=off_fixed, diffusivity=D)
        model = FCSFitting.FCSModel(; spec)
        y = model(τ, [g0_true, w0_true])

        # Initial guesses & bounds
        p0    = [0.001, 2e-9]
        lower = [0.0, 0.0]
        upper = [1.0, 500e-9]

        fit, sc = FCSFitting.fcs_fit(spec, τ, y, p0; lower=lower, upper=upper)
        p̂ = fit.param .* sc
        g0o, w0o = p̂

        @test g0o ≈ g0_true rtol=1e-4
        @test w0o ≈ w0_true rtol=1e-4
        @test FCSFitting.τD(D, w0o) ≈ τD_true rtol=1e-4
    end
end