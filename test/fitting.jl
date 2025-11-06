@testset "fitting" begin
    @testset "FCSFitResult constructor + StatsAPI shims" begin
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

        # Build a "better" fit by re-fitting with the true parameters as θ0
        lsf_better = curve_fit(exp_model, x, y, wt, [a_true, b_true, c_true])
        ffr_better = FCSFitResult(lsf_better, spec_free, scales)
        @test StatsAPI.loglikelihood(ffr_better) >= StatsAPI.loglikelihood(ffr)
        @test StatsAPI.aic(ffr_better) <= StatsAPI.aic(ffr)
        @test StatsAPI.aicc(ffr_better) <= StatsAPI.aicc(ffr)
        @test StatsAPI.bic(ffr_better) <= StatsAPI.bic(ffr)
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

    @testset "infer_noscale_indices" begin
        # 2D, normal diffusion, ONE diffuser, FREE offset (offset=nothing)
        # Parameter layout (n=1, m=1): [g0, offset, τD, τ_dyn, K_dyn]
        spec = FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1, offset=nothing)
        p0 = [1.0, 0.0, 1e-3, 1e-6, 0.2]
        idxs = FCSFitting.infer_noscale_indices(spec, p0)
        @test idxs == [5]

        # 3D, Brownian (anom=none), TWO diffusers, FIXED offset
        # Parameter layout (n=2, m=1, fixed offset): [g0, κ, τD1, τD2, w1, τ_dyn, K_dyn]
        spec = FCSModelSpec(; dim=FCSFitting.d3, anom=FCSFitting.none, n_diff=2, offset=0.0)
        p0 = [1.0, 8.0, 1e-3, 2e-4, 0.3, 5e-6, 0.1]
        idxs = FCSFitting.infer_noscale_indices(spec, p0)
        @test idxs == [5, 7]

        # Parameter layout (n=2, m=1, fixed offset): [g0, κ, τD1, τD2, α1, α2, w1, τ_dyn, K_dyn]
        spec = FCSModelSpec(; dim=FCSFitting.d3, anom=FCSFitting.perpop, n_diff=2, offset=0.0)
        p0 = [1.0, 8.0, 1e-3, 2e-4, 1.1, 0.9, 0.3, 5e-6, 0.1]
        idxs = FCSFitting.infer_noscale_indices(spec, p0)
        @test idxs == [7, 9]
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

        # Fit without bounds
        p0 = [0.5, 0.0, 5e-4]   # rough initial guesses
        fit = FCSFitting.fcs_fit(spec, τ, y, p0)
        p̂ = coef(fit)

        @test p̂[1] ≈ g0_true rtol=1e-5
        @test p̂[2] ≈ offset_true rtol=1e-5
        @test p̂[3] ≈ τD_true rtol=1e-5

        # Fit with bounds (make g0 lower bound larger than truth)
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

        # Fit with only lower bound
        p0 = [0.5, 5, 5e-4, 5e-5, 0.5, 5e-7, 0.25]
        lower = zeros(7)
        fit = FCSFitting.fcs_fit(spec, τ, y, p0; lower)
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
        g0_true = rand()
        offset_true = 2e-3
        τD_true = 8e-4

        spec = FCSFitting.FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1)
        model = FCSFitting.FCSModel(; spec)
        y_true = model(τ, [g0_true, offset_true, τD_true])

        # Heteroscedastic noise
        σ = @. 0.002 + 0.003 * (τ / maximum(τ))
        y = y_true .+ σ .* randn(length(τ))

        ch = FCSChannel("G1", τ, y, σ)

        p0 = [0.5, 0.0, 1e-3]
        upper = [1, 1e-2, 1e-2]

        # Using σ (internally converted to 1/σ²)
        fitA = FCSFitting.fcs_fit(spec, ch, p0; upper)
        pA = coef(fitA)

        # Using explicit weights
        wt = @. 1 / σ^2
        fitB = FCSFitting.fcs_fit(model, ch, p0; wt=wt, upper)
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

        fit = FCSFitting.fcs_fit(model, τ, y, p0; lower=lower, upper=upper)
        p̂ = coef(fit)
        g0o, w0o = p̂

        @test g0o ≈ g0_true rtol=1e-4
        @test w0o ≈ w0_true rtol=1e-4
        @test FCSFitting.τD(D, w0o) ≈ τD_true rtol=1e-4
    end


    NA = FCSFitting.AVAGADROS
    kB = FCSFitting.BOLTZMANN

    @testset "Calculators (base)" begin
        D = 5e-11
        w0 = 250e-9
        κ = 10 * rand()
        g0 = rand()
        Ks = [0.1, 0.2]

        # τD and diffusivity
        τ = FCSFitting.τD(D, w0)
        @test τ ≈ (w0^2) / (4D)
        τ_scale = FCSFitting.τD(D, w0; scale="μ")
        @test τ_scale ≈ 1e6 * (w0^2) / (4D)
        D_back = FCSFitting.diffusivity(τ, w0)
        @test D_back ≈ D

        @test_throws ArgumentError FCSFitting.τD(-1.0, w0)
        @test_throws ArgumentError FCSFitting.τD(D, -1.0)
        @test_throws ArgumentError FCSFitting.diffusivity(-1.0, w0)
        @test_throws ArgumentError FCSFitting.diffusivity(τ, -1.0)

        # confocal volume/area
        vol = FCSFitting.Veff(w0, κ)
        @test vol ≈ π^(3/2) * w0^3 * κ rtol=1e-12
        vol_scale = FCSFitting.Veff(w0, κ; scale="n")
        @test vol_scale ≈ 1e27 * π^(3/2) * w0^3 * κ
        ar = FCSFitting.Aeff(w0)
        @test ar ≈ π * w0^2 rtol=1e-12
        ar_scale = FCSFitting.Aeff(w0; scale="n")
        @test ar_scale ≈ 1e18 * π * w0^2

        @test_throws ArgumentError FCSFitting.Veff(-1.0, κ)
        @test_throws ArgumentError FCSFitting.Veff(w0, -1.0)
        @test_throws ArgumentError FCSFitting.Aeff(-1.0)

        # concentration (blinkless)
        c = FCSFitting.concentration(g0, κ, w0)
        @test c ≈ (1/g0) / (NA * vol * 1000.0)
        c_scale = FCSFitting.concentration(g0, κ, w0; scale="m")
        @test c_scale ≈ (1/g0) / (NA * vol)

        @test_throws ArgumentError FCSFitting.concentration(-1.0, κ, w0)
        @test_throws ArgumentError FCSFitting.concentration(g0, -1.0, w0)
        @test_throws ArgumentError FCSFitting.concentration(g0, κ, -1.0)
        @test_throws ArgumentError FCSFitting.concentration(g0, κ, w0; Ks=[-0.1])
        @test_throws ArgumentError FCSFitting.concentration(g0, κ, w0; Ks=[0.1,0.2], ics=[1])
        @test_throws ArgumentError FCSFitting.concentration(g0, κ, w0; Ks=[], ics=[1])

        B0 = 1 + (0.1/0.9 + 0.2/0.8)
        Neff1 = B0 / g0
        c_expected = Neff1 / (NA * vol * 1000.0)
        @test FCSFitting.concentration(g0, κ, w0; Ks=Ks, ics=[2]) ≈ c_expected

        B0 = (1 + 0.1/0.9) * (1 + 0.2/0.8)
        Neff2 = B0 / g0
        c_expected = Neff2 / (NA * vol * 1000.0)
        @test FCSFitting.concentration(g0, κ, w0; Ks=Ks, ics=[1,1]) ≈ c_expected

        # surface density (2D analogue)
        sa = FCSFitting.surface_density(g0, w0)
        sa_expected = (1/g0) / (NA * ar)
        @test sa ≈ sa_expected rtol=1e-12

        @test_throws ArgumentError FCSFitting.surface_density(-1.0, w0)
        @test_throws ArgumentError FCSFitting.surface_density(g0, -1.0)
        @test_throws ArgumentError FCSFitting.surface_density(g0, w0; Ks=[-0.1])
        @test_throws ArgumentError FCSFitting.surface_density(g0, w0; Ks=[0.1,0.2], ics=[1])
        @test_throws ArgumentError FCSFitting.surface_density(g0, w0; Ks=[], ics=[1])

        sa_expected = Neff1 / (NA * ar)
        @test FCSFitting.surface_density(g0, w0; Ks=Ks, ics=[2]) ≈ sa_expected
        sa_expected = Neff2 / (NA * ar)
        @test FCSFitting.surface_density(g0, w0; Ks=Ks, ics=[1,1]) ≈ sa_expected

        # hydrodynamic radius
        T = 293.0;  η = 1.0016e-3
        Rh = FCSFitting.hydrodynamic(D; T=T, η=η)
        @test Rh ≈ kB * T / (6π * η * D)
        Rh_scale = FCSFitting.hydrodynamic(D; T=T, η=η, scale="A")
        @test Rh_scale ≈ 1e10 * kB * T / (6π * η * D)

        @test_throws ArgumentError FCSFitting.hydrodynamic(-1.0; T=T, η=η)
        @test_throws ArgumentError FCSFitting.hydrodynamic(D; T=-1.0, η=η)
        @test_throws ArgumentError FCSFitting.hydrodynamic(D; T=T, η=-1.0)
    end


    @testset "Calculators (from spec+fit)" begin
        D = 1e-10 * rand()
        w0 = 500e-9 * rand()
        κ = 10*rand()
        g0 = rand()
        
        τ = 10 .^ range(-6, 0; length=300)

        # ------------------------------------------------------------
        # 3D, fixed diffusivity, one diffuser → params = [g0, κ, w0]
        # ------------------------------------------------------------
        spec3 = FCSFitting.FCSModelSpec(;
            dim = FCSFitting.d3,
            anom = FCSFitting.none,
            offset = 0.0,
            diffusivity = D,
            n_diff = 1,
        )
        model3 = FCSFitting.FCSModel(; spec = spec3)
        y3_true = model3(τ, [g0, κ, w0])

        p0_3 = [0.5, 5, 250e-9]
        fit3 = FCSFitting.fcs_fit(spec3, τ, y3_true, p0_3)

        τD_expected = FCSFitting.τD(D, w0)
        @test FCSFitting.τD(spec3, fit3) ≈ τD_expected rtol=1e-6
        @test FCSFitting.τD(spec3, fit3; scale="μ") ≈ 1e6 * τD_expected rtol=1e-6

        @test FCSFitting.diffusivity(spec3) ≈ D rtol=1e-12
        @test FCSFitting.diffusivity(spec3; scale="μ") ≈ 1e12 * D rtol=1e-12

        V_expected = FCSFitting.Veff(w0, κ)
        @test FCSFitting.Veff(spec3, fit3) ≈ V_expected rtol=1e-6
        @test FCSFitting.Veff(spec3, fit3; scale="n") ≈ 1e27 * V_expected rtol=1e-6

        c_expected = FCSFitting.concentration(g0, κ, w0)
        @test FCSFitting.concentration(spec3, fit3; scale="L") ≈ c_expected rtol=1e-6
        @test FCSFitting.concentration(spec3, fit3; scale="n") ≈ 1e9 * c_expected rtol=1e-6
        @test_throws ArgumentError FCSFitting.concentration(spec3, fit3; nd=2) # more diffusers than allowed
        @test_throws ArgumentError FCSFitting.concentration(spec3, fit3; nd=0) # fewer diffusers than allowed

        # ------------------------------------------------------------
        # 3D, free diffusivity, one diffuser → params = [g0, κ, τD]
        # ------------------------------------------------------------
        spec3_free = FCSFitting.FCSModelSpec(;
            dim = FCSFitting.d3,
            anom = FCSFitting.none,
            offset = 0.0,
            n_diff = 1,
        )
        model3_free = FCSFitting.FCSModel(; spec = spec3_free)
        y3_free_true = model3_free(τ, [g0, κ, τD_expected])

        p0_3free = [0.5, 5, 0.0003125]
        lower = [0.0, 1.0, 1e-5]
        upper = [1.5, 10.0, 1e-3]
        fit3_free = FCSFitting.fcs_fit(spec3_free, τ, y3_free_true, p0_3free; lower, upper)

        @test_throws ArgumentError FCSFitting.Veff(spec3_free, fit3_free)
        @test_throws ArgumentError FCSFitting.concentration(spec3_free, fit3_free)
        @test_throws ArgumentError FCSFitting.concentration(spec3_free, fit3_free; w0=w0, nd=0)

        @test FCSFitting.Veff(spec3_free, fit3_free; w0 = w0) ≈ V_expected rtol=1e-6
        @test FCSFitting.concentration(spec3_free, fit3_free; w0=w0) ≈ c_expected rtol=1e-6

        # ------------------------------------------------------------
        # 2D, fixed diffusivity, one diffuser → params = [g0, w0]
        # ------------------------------------------------------------
        spec2_fixed = FCSFitting.FCSModelSpec(;
            dim = FCSFitting.d2,
            anom = FCSFitting.none,
            offset = 0.0,
            diffusivity = D,
            n_diff = 1,
        )
        model2_fixed = FCSFitting.FCSModel(; spec = spec2_fixed)
        y2_fixed_true = model2_fixed(τ, [g0, w0])

        p0_2f = [0.5, 250e-9]
        fit2_fixed = FCSFitting.fcs_fit(spec2_fixed, τ, y2_fixed_true, p0_2f)

        ar = FCSFitting.Aeff(w0)
        @test FCSFitting.Aeff(spec2_fixed, fit2_fixed) ≈ ar rtol=1e-6

        sa_expected = (1/g0) / (NA * ar)
        sa_from_fit = FCSFitting.surface_density(spec2_fixed, fit2_fixed)
        @test sa_from_fit ≈ sa_expected rtol=1e-6

        # ------------------------------------------------------------
        # 2D, *free* diffusivity, one diffuser → params = [g0, τD]
        # ------------------------------------------------------------
        τD_true = 1e-3*rand()
        spec2_free = FCSFitting.FCSModelSpec(;
            dim = FCSFitting.d2,
            anom = FCSFitting.none,
            offset = 0.0,
            n_diff = 1,
        )
        model2_free = FCSFitting.FCSModel(; spec = spec2_free)
        y2_free_true = model2_free(τ, [g0, τD_true])

        p0_2free = [0.3, τD_true*0.8]
        fit2_free = FCSFitting.fcs_fit(spec2_free, τ, y2_free_true, p0_2free)

        # calling without w0 should error
        @test_throws ArgumentError FCSFitting.surface_density(spec2_free, fit2_free)
        @test_throws ArgumentError FCSFitting.diffusivity(spec2_free, fit2_free)

        sa_from_free = FCSFitting.surface_density(spec2_free, fit2_free; w0 = w0)
        @test sa_from_free ≈ sa_expected rtol=1e-6
        
        diff_from_free = FCSFitting.diffusivity(spec2_free, fit2_free; w0 = w0)
        @test diff_from_free ≈ w0^2 / (4τD_true)

        τD_from_2free = FCSFitting.τD(spec2_free, fit2_free)
        @test τD_from_2free ≈ τD_true rtol=1e-10

        # ------------------------------------------------------------
        # 3D, fixed diffusivity, 2 diffusers, GLOBAL anomalous exponent
        # params = [g0, κ, w01, w02, α, w1]
        # ------------------------------------------------------------
        w01 = 300e-9
        w02 = 700e-9
        α = rand()
        w1 = 0.5*rand()
        spec3_glob = FCSFitting.FCSModelSpec(;
            dim = FCSFitting.d3,
            anom = FCSFitting.globe,
            offset = 0.0,
            diffusivity = D,
            n_diff = 2,
        )
        model3_glob = FCSFitting.FCSModel(; spec = spec3_glob)
        y3_glob_true = model3_glob(τ, [g0, κ, w01, w02, α, w1])
        p0_glob = [0.5, 5, w01, w02, 0.5, 0.25]
        lower_glob = [0, 0, 250e-9, 650e-9, 0, 0]
        upper_glob = [1, 10, 350e-9, 750e-9, 1, 0.5]
        fit3_glob = FCSFitting.fcs_fit(spec3_glob, τ, y3_glob_true, p0_glob; lower=lower_glob, upper=upper_glob)

        # τD for first and second diffuser should match their w0 slots
        τD1_exp = FCSFitting.τD(D, w01)
        τD2_exp = FCSFitting.τD(D, w02)
        @test FCSFitting.τD(spec3_glob, fit3_glob; nd=1) ≈ τD1_exp rtol=1e-6
        @test FCSFitting.τD(spec3_glob, fit3_glob; nd=2) ≈ τD2_exp rtol=1e-6
        @test_throws ArgumentError FCSFitting.τD(spec3_glob, fit3_glob; nd=3)

        # concentration for nd=2 should also work
        c2_exp = FCSFitting.concentration(g0, κ, w02)
        @test FCSFitting.concentration(spec3_glob, fit3_glob; nd=2) ≈ c2_exp rtol=1e-6

        # ------------------------------------------------------------
        # 3D, fixed diffusivity, 2 diffusers, PER-POP anomalous exponents
        # params = [g0, κ, w01, w02, α1, α2, w1]
        # ------------------------------------------------------------
        α1 = rand()
        α2 = rand()+1
        spec3_perpop = FCSFitting.FCSModelSpec(;
            dim = FCSFitting.d3,
            anom = FCSFitting.perpop,
            offset = 0.0,
            diffusivity = D,
            n_diff = 2,
        )
        τ = 10 .^ range(-7, 0; length=400)
        model3_perpop = FCSFitting.FCSModel(; spec = spec3_perpop)
        y3_perpop_true = model3_perpop(τ, [g0, κ, w01, w02, α1, α2, w1])
        p0_perpop = [0.55, 5, w01, w02, 0.5, 1.5, 0.25]
        lower_perpop = [0, 0, 250e-9, 650e-9, 0, 1, 0]
        upper_perpop = [1, 10, 350e-9, 750e-9, 1, 2, 1]
        fit3_perpop = FCSFitting.fcs_fit(spec3_perpop, τ, y3_perpop_true, p0_perpop; lower=lower_perpop, upper=upper_perpop)
        
        @test FCSFitting.τD(spec3_perpop, fit3_perpop; nd=1) ≈ τD1_exp rtol=1e-6
        @test FCSFitting.τD(spec3_perpop, fit3_perpop; nd=2) ≈ τD2_exp rtol=1e-6

        # concentration still well-defined
        @test FCSFitting.concentration(spec3_perpop, fit3_perpop; nd=1) ≈ FCSFitting.concentration(g0, κ, w01) rtol=1e-6

        # ------------------------------------------------------------
        # 2D, fixed diffusivity, anomalous (global)
        # just make sure surface_density still works
        # ------------------------------------------------------------
        spec2_anom = FCSFitting.FCSModelSpec(;
            dim = FCSFitting.d2,
            anom = FCSFitting.globe,
            offset = 0.0,
            diffusivity = D,
            n_diff = 1,
        )
        model2_anom = FCSFitting.FCSModel(; spec = spec2_anom)
        y2_anom_true = model2_anom(τ, [g0, w0, 0.85])  # [g0, w0, α]
        p0_2anom = [0.4, w0*1.1, 0.9]
        fit2_anom = FCSFitting.fcs_fit(spec2_anom, τ, y2_anom_true, p0_2anom)

        @test FCSFitting.surface_density(spec2_anom, fit2_anom) ≈ sa_expected rtol=1e-6
    end
end