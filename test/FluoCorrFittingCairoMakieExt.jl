using CairoMakie, LaTeXStrings

CairoMakie.activate!()  # headless-friendly

@testset "FluoCorrFittingCairoMakieExt" begin
    # Synthetic 3D Brownian (no anomalous), one diffuser, fixed offset
    τ = 10 .^ range(-6, 0; length=250)
    g0_true = 0.9
    κ_true = 4.2
    off_true = 0.0
    τD_true = 7.5e-4

    spec = FluoCorrFitting.FCSModelSpec(; dim=FluoCorrFitting.d3, anom=FluoCorrFitting.none,
                                    n_diff=1, offset=off_true)
    p0 = [0.5, 3.0, 1e-4]
    model = FluoCorrFitting.FCSModel(spec, τ, p0)
    y_clean = model(τ, [g0_true, κ_true, τD_true])

    # Slight noise for realistic plotting
    σ = 0.002
    y = @. y_clean + σ * randn()

    
    fit = FluoCorrFitting.fcs_fit(spec, τ, y, p0)

    @test fit isa FluoCorrFitting.FCSFitResult

    @testset "_fcs_plot" begin
        fig, fit_out = FluoCorrFitting._fcs_plot(fit, τ, y; fit_kw = (color=:red,), resid_kw = (color=:blue,))
        @test fit_out === fit
        @test fig isa Makie.Figure
        # two axes should exist (top + bottom)
        axes = [obj for obj in fig.content if obj isa Makie.Axis]
        @test length(axes) == 2


        fig, fit_out = FluoCorrFitting._fcs_plot(fit, τ, y; residuals=false, data_kw = (color=:red,))
        @test fit_out === fit
        @test fig isa Makie.Figure
        axes = [obj for obj in fig.content if obj isa Makie.Axis]
        @test length(axes) == 1

        fig2, fit_out2 = FluoCorrFitting._fcs_plot(fit, τ, y; residuals=false, fig = fig)
        @test fit_out2 === fit
        @test fig2 isa Makie.Figure
        axes = [obj for obj in fig2.content if obj isa Makie.Axis]
        @test length(axes) == 1
    end

    @testset "_resid_acf_plot" begin
        fig = FluoCorrFitting._resid_acf_plot(randn(100), 200; figure_kw=(fontsize=16,))
        @test fig isa Makie.Figure
    end
end