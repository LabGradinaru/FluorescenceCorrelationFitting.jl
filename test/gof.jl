@testset "gof" begin
    # Build a small deterministic least-squares fit so we can test AIC/BIC/etc.
    # Model: y = a*x + b; data = aₜ*x + bₜ + ε (constant ε so rss is deterministic)
    function make_linear_fit(N::Int=20; a_true=2.0, b_true=1.0, eps=0.1)
        x = collect(1.0:N)
        y = a_true .* x .+ b_true .+ eps
        model(x, p) = @. p[1]*x + p[2]
        p0 = [1.5, 0.5]
        fit = curve_fit(model, x, y, p0)  # returns LsqFit.LsqFitResult
        return (x=x, y=y, fit=fit, model=model)
    end

    # Manual information-criteria eval for an LsqFitResult
    function manual_ic(fit::LsqFit.LsqFitResult)
        k, N = length(coef(fit)), nobs(fit)
        σ2 = rss(fit) / N
        AIC  = 2k + N*log(σ2)
        AICc = AIC + (2k*(k+1)) / (N - k - 1)
        BIC  = k*log(N) + N*log(σ2)
        BICc = N*log(σ2) + N*k*log(N) / (N - k - 2)
        return (AIC, AICc, BIC, BICc)
    end

    # --- aic / aicc / bic / bicc ----------------------------------------------

    @testset "Information criteria" begin
        data = make_linear_fit(25; a_true=2.0, b_true=1.0, eps=0.2)
        fit = data.fit

        AICm, AICcm, BICm, BICcm = manual_ic(fit)

        @test isapprox(FCSFitting.aic(fit),  AICm;  rtol=1e-12)
        @test isapprox(FCSFitting.aicc(fit), AICcm; rtol=1e-12)
        @test isapprox(FCSFitting.bic(fit),  BICm;  rtol=1e-12)
        @test isapprox(FCSFitting.bicc(fit), BICcm; rtol=1e-12)

        # Sanity: as N grows with same noise level, AICc ≈ AIC
        data_big = make_linear_fit(400; a_true=2.0, b_true=1.0, eps=0.2)
        δ = abs(FCSFitting.aicc(data_big.fit) - FCSFitting.aic(data_big.fit))
        @test δ ≈ 12/397
    end

    # --- chi_squared (weighted & unweighted, reduced, dof override) ------------

    @testset "Chi-squared" begin
        data = make_linear_fit(30; eps=0.3)
        fit = data.fit
        N = length(fit.resid)
        k = length(coef(fit))
        dof = N - k

        # Unweighted: χ² = sum rᵢ²
        χ2_unw = sum(abs2, fit.resid)
        @test isapprox(FCSFitting.chi_squared(fit), χ2_unw; rtol=0, atol=0)

        # Reduced form divides by dof
        @test isapprox(FCSFitting.chi_squared(fit; reduced=true), χ2_unw/dof; rtol=0, atol=0)

        # Weighted: σᵢ = 2 → χ² becomes quarter of unweighted (since (r/2)²)
        σ = fill(2.0, N)
        @test isapprox(FCSFitting.chi_squared(fit, σ), χ2_unw/4; rtol=0, atol=0)

        # dof_override respected in reduced path
        dof2 = 10
        @test isapprox(FCSFitting.chi_squared(fit; reduced=true, dof_override=dof2), χ2_unw/dof2; atol=0, rtol=0)
    end

    # --- Wald–Wolfowitz runs test & _count_runs --------------------------------

    @testset "Runs test" begin
        # Explicit runs count
        @test FCSFitting._count_runs([1, -1, 1, -1]) == 4
        @test FCSFitting._count_runs([1, 1, 1, -1, -1, 1]) == 3

        # z-score sign sanity:
        # Many runs (alternating signs) ⇒ R ≫ μ ⇒ positive z
        z_alt = FCSFitting.ww_test([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        @test z_alt > 0

        # Few runs (grouped signs) ⇒ R ≪ μ ⇒ negative z
        z_grp = FCSFitting.ww_test([1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
        @test z_grp < 0

        # Dropping zeros changes nothing here (no zeros), but ensure API works
        @test FCSFitting.ww_test([1.0, -1.0, 0.0, 1.0, -1.0]; drop_zeros=true) > 0

        # Error on all-same-sign after dropping zeros
        @test_throws AssertionError FCSFitting.ww_test([0.0, 0.0, 1.0]; drop_zeros=true)
    end

    # --- acf -------------------------------------------------------------------

    @testset "ACF properties" begin
        # 1) Bad maxlag throws
        x = randn(10)  # small throwaway
        @test_throws AssertionError FCSFitting.acf(x; maxlag=10)

        # 2) Geometric deterministic sequence, demean=false:
        #    y_t = ϕ^t  ⇒  ρ(k) ≈ ϕ^k for large N
        ϕ = 0.8
        N = 200
        y = [ϕ^(t-1) for t in 1:N]
        ρ = FCSFitting.acf(y; maxlag=10, demean=false, unbiased=false)
        @test ρ[1] ≈ 1.0
        for k in 1:10
            @test isapprox(ρ[k+1], ϕ^k; atol=5e-3)  # finite-size tolerance
        end

        # 3) With demean=true the match is not exact; just basic invariants
        ρd = FCSFitting.acf(y; maxlag=5, demean=true, unbiased=true)
        @test ρd[1] ≈ 1.0
        @test all(abs.(ρd[2:end]) .<= 1.0 .+ 1e-12)
    end

    # --- Ljung–Box Q -----------------------------------------------------------

    @testset "Ljung-Box Q" begin
        # Compare a negatively autocorrelated alternating sequence vs white noise
        N = 200
        white = randn(N)

        alt = [(-1.0)^(i) for i in 1:N]  # strong lag-1 correlation (≈ -1)

        Q_white = FCSFitting.ljung_box(white; h=10, demean=true)
        Q_alt   = FCSFitting.ljung_box(alt;   h=10, demean=true)

        @test Q_alt > Q_white   # autocorrelation should inflate Q

        # h=:auto is within [1, N-2] and returns a scalar
        Q_auto = FCSFitting.ljung_box(white; h=:auto, demean=true)
        @test Q_auto isa Real
    end

end