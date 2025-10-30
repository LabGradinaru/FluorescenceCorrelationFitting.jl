using CairoMakie, LaTeXStrings

CairoMakie.activate!()  # headless-friendly

@testset "FCSFittingCairoMakieExt" begin
    Random.seed!(42)

    # Synthetic 3D Brownian (no anomalous), one diffuser, fixed offset
    τ = 10 .^ range(-6, -1; length=300)
    g0_true = 0.9
    κ_true = 4.2
    off_true = 0.0
    τD_true = 7.5e-4

    spec = FCSFitting.FCSModelSpec(; dim=FCSFitting.d3, anom=FCSFitting.none,
                                    n_diff=1, offset=off_true)
    model = FCSFitting.FCSModel(; spec)
    y_clean = model(τ, [g0_true, κ_true, τD_true])

    # Slight noise for realistic plotting
    σ = 0.002
    y = @. y_clean + σ * randn()

    # Fit quickly (unweighted is fine for this test)
    p0 = [0.5, 3.0, 1e-4]
    fit = FCSFitting.fcs_fit(spec, τ, y, p0)

    @test fit isa FCSFitting.FCSFitResult

    @testset "_fcs_plot (with residuals; 3-color variant)" begin
        fig, fit_out = FCSFitting._fcs_plot(fit, τ, y, :deepskyblue3, :orangered2, :steelblue4)
        @test fit_out === fit
        @test fig isa Makie.Figure
        # two axes should exist (top + bottom)
        axes = [obj for obj in fig.content if obj isa Makie.Axis]
        @test length(axes) ≥ 2
    end

    @testset "_fcs_plot (no residuals; 2-color variant)" begin
        fig, fit_out = FCSFitting._fcs_plot(fit, τ, y, :deepskyblue3, :orangered2)
        @test fit_out === fit
        @test fig isa Makie.Figure
        axes = [obj for obj in fig.content if obj isa Makie.Axis]
        @test length(axes) ≥ 1
    end

    @testset "resid_acf_plot" begin
        fig, ρ = FCSFitting.resid_acf_plot(fit; fontsize=16, maxlag=20)
        @test fig isa Makie.Figure
        @test length(ρ) == 21  # 0..20
        @test ρ[1] ≈ 1.0 atol=1e-12
    end
end