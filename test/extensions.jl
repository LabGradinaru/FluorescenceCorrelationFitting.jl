@testset "extensions" begin
    # Common toy data
    τp = collect(1e-6:1e-6:1e-4)
    Gp = 1.0 ./ (1 .+ τp ./ 1e-4)
    chp = FCSChannel("G[1]", τp, Gp, nothing)
    spec_for_plot = FCSModelSpec(; dim=FluorescenceCorrelationFitting.d2, anom=FluorescenceCorrelationFitting.none, n_diff=1)

    @testset "shims (no optional deps loaded)" begin
        function _errstr(f)
            try
                f()
                return nothing
            catch e
                return sprint(showerror, e)
            end
        end

        s = _errstr(() -> FluorescenceCorrelationFitting.fcs_plot(spec_for_plot, chp, [1.0, 0.0, 1e-3]))
        @test s !== nothing
        @test occursin("requires CairoMakie", s)

        s = _errstr(() -> FluorescenceCorrelationFitting._fcs_plot(spec_for_plot, chp, [1.0, 0.0, 1e-3]))
        @test s !== nothing
        @test occursin("requires CairoMakie", s)

        s = _errstr(() -> FluorescenceCorrelationFitting._resid_acf_plot(randn(10), 10))
        @test s !== nothing
        @test occursin("requires CairoMakie", s)

        s = _errstr(() -> FluorescenceCorrelationFitting.fcs_table(spec_for_plot, nothing, nothing))
        @test s !== nothing
        @test occursin("requires PrettyTables", s)
    end

    @testset "helpers (core logic)" begin
        c1, c2, c3 = :blue, :red, :green

        # Preferred: tuple/NamedTuple
        colors = FluorescenceCorrelationFitting._fcs_colors_nt((c1, c2, c3))
        @test colors[:data] == c1
        @test colors[:fit]  == c2
        @test colors[:resid]== c3

        colors = FluorescenceCorrelationFitting._fcs_colors_nt((c1, c2))
        @test colors[:data] == c1
        @test colors[:fit]  == c2
        @test colors[:resid]== FluorescenceCorrelationFitting.DEFAULT_FCS_PLOT_COLORS[:resid]


        x = [1.0, 2.0, 4.0, 7.0]
        ρ = FluorescenceCorrelationFitting.acf(x; maxlag=2, demean=true, unbiased=true)

        # compute expected by hand (same normalization as implementation)
        N = length(x)
        μ = sum(x) / N
        y = x .- μ
        γ0 = sum(abs2, y) / (N - 1)  # unbiased=true
        num1 = sum(@view(y[1:N-1]) .* @view(y[2:N]))
        num2 = sum(@view(y[1:N-2]) .* @view(y[3:N]))
        ρ1 = (num1 / (N - 1)) / γ0
        ρ2 = (num2 / (N - 2)) / γ0

        @test ρ[1] == 1
        @test isapprox(ρ[2], ρ1; rtol=0, atol=1e-12)
        @test isapprox(ρ[3], ρ2; rtol=0, atol=1e-12)

        # bounds / assertions
        @test_throws AssertionError FluorescenceCorrelationFitting.acf(randn(5); maxlag=0)
        @test_throws AssertionError FluorescenceCorrelationFitting.acf(randn(5); maxlag=5)

        Nbig = 5000
        xb = randn(Nbig)
        ρb = FluorescenceCorrelationFitting.acf(xb; maxlag=50)
        @test ρb[1] == 1
        @test maximum(abs, ρb[2:end]) < 0.08
    end
end
