using PrettyTables

@testset "FCSFittingPrettyTablesExt" begin
    # Build a simple synthetic problem (2D Brownian, free offset)
    τ = collect(range(1e-6, 1e-3; length=128))
    g0_true = 0.85
    off_true = 1e-3
    τD_true = 7.5e-4

    spec = FCSFitting.FCSModelSpec(; dim=FCSFitting.d2, anom=FCSFitting.none, n_diff=1, offset=nothing)
    model = FCSFitting.FCSModel(; spec)

    y = model(τ, [g0_true, off_true, τD_true])
    p0 = [0.5, 0.0, 1e-4]

    fit = FCSFitting.fcs_fit(spec, τ, y, p0)

    # Render a table and capture the output
    io = IOBuffer()
    redirect_stdout(() -> begin
        FCSFitting.fcs_table(fit; backend=:unicode, units=["", "", "μ"])
    end, io)
    txt = String(take!(io))

    # Basic sanity checks on the rendered output
    @test occursin("Parameters", txt)
    @test occursin("Std. Dev.", txt)
    @test occursin("bic", txt)
    @test occursin("μs", txt)
    @test length(txt) > 0

    # BIC should be finite for a well-posed fit
    @test isfinite(StatsAPI.bic(fit))
end