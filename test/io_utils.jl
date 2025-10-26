@testset "io_utils" begin
    # SI_PREFIXES
    @test FCSFitting.SI_PREFIXES[""]  == 1.0
    @test FCSFitting.SI_PREFIXES["d"] == 1e1
    @test FCSFitting.SI_PREFIXES["c"] == 1e2
    @test FCSFitting.SI_PREFIXES["m"] == 1e3
    @test FCSFitting.SI_PREFIXES["μ"] == 1e6
    @test FCSFitting.SI_PREFIXES["u"] == 1e6
    @test FCSFitting.SI_PREFIXES["n"] == 1e9
    @test FCSFitting.SI_PREFIXES["p"] == 1e12


    # FCSChannel / FCSData
    τ = collect(range(1e-6, 1e-3; length=10))
    G = 0.02 .+ 1.0 ./ (1 .+ τ ./ 1e-4)
    σ = fill(1e-3, length(G))

    ch = FCSChannel("G[1]", τ, G, σ)
    @test ch.name == "G[1]"
    @test ch.τ === τ
    @test ch.G === G
    @test ch.σ === σ

    data = FCSData([ch], Dict("sample" => "test", "note" => 123), "in-memory")
    @test length(data.channels) == 1
    @test data.metadata["sample"] == "test"
    @test data.source == "in-memory"


    # Parameter name inference helpers
    # Convenience aliases to constants used in names
    G0 = FCSFitting.G0_NAME
    OFF = FCSFitting.OFF_NAME
    AN = FCSFitting.ANOM_NAME
    ST = FCSFitting.STRUCT_NAME
    RT = FCSFitting.DIFFTIME_NAME
    WF = FCSFitting.DIFFFRAC_NAME
    BW = FCSFitting.BEAM_NAME
    DT = FCSFitting.DYNTIME_NAME
    DF = FCSFitting.DYNFRAC_NAME

    # 2D, Brownian, single diffuser, free offset
    spec_2d = FCSModelSpec(; dim=:d2)
    @test FCSFitting.expected_parameter_names(spec_2d) ==
        [G0, OFF, "$(RT) [s]", "$(DT) [1:m] [s]", "$(DF) [1:m]"]

    # 2D, Brownian, single diffuser, fixed offset (removed from p), still τD slots
    spec_2d_fixoff = FCSModelSpec(; dim=:d2, offset=0.0)
    @test FCSFitting.expected_parameter_names(spec_2d_fixoff)[1:1] == [G0]  # no OFF in front matter

    # 2D, diffusion given by w0 (fixed D in spec) → label should be Beam width
    spec_2d_w0 = FCSModelSpec(; dim=:d2, diffusivity=5e-11)
    # Only checking the base portion that changes wording
    base2d = FCSFitting.infer_parameter_names(spec_2d_w0, [0.0, 0.0, 0.0])  # g0, off fixed? no → expect G0, OFF, w0, (no dynamics)
    @test base2d[1:3] == [G0, OFF, "$(BW) [m]"] # TODO: issue here :/

    # 2D, anomalous (global α), n_diff=1
    spec_2d_ag = FCSModelSpec(; dim=:d2, anom=:global)
    nd_ag = FCSFitting._no_dynamics_params(spec_2d_ag)
    @test nd_ag == [G0, OFF, "$(RT) [s]", AN]
    # With one dynamics block (τ_dyn1, K_dyn1)
    names_ag = FCSFitting.infer_parameter_names(spec_2d_ag, zeros(1 + 1 + 1 + 1 + 2))  # g0, off, τD, α, (τ,K)
    @test names_ag[end-1:end] == ["$(DT) 1 [s]", "$(DF) 1"]

    # 2D, anomalous per-pop, n_diff=2 (α1, α2) + 1 weight
    spec_2d_ap = FCSModelSpec(; dim=:d2, anom=:perpop, n_diff=2)
    nd_ap = FCSFitting._no_dynamics_params(spec_2d_ap)
    @test nd_ap == [G0, OFF, "$(RT) 1 [s]", "$(RT) 2 [s]", "$(AN) 1", "$(AN) 2", "$(WF) 1"]

    # 3D, Brownian, single diffuser
    spec_3d = FCSModelSpec(; dim=:d3)
    nd_3d = FCSFitting._no_dynamics_params(spec_3d)
    @test nd_3d == [G0, OFF, ST, "$(RT) [s]"]

    # 3D, anomalous (global), with dynamics count inferred from params length
    spec_3d_ag = FCSModelSpec(; dim=:d3, anom=:global)
    # g0, off, κ, τD, α, (τ1,K1), (τ2,K2)  -> total 5 + 4 = 9
    names_3d_ag = FCSFitting.infer_parameter_names(spec_3d_ag, zeros(9))
    @test names_3d_ag[1:5] == [G0, OFF, ST, "$(RT) [s]", AN]
    @test names_3d_ag[6:9] == ["$(DT) 1 [s]", "$(DT) 2 [s]", "$(DF) 1", "$(DF) 2"]


    # sigstr
    @test FCSFitting.sigstr(0.0) == "0"
    @test FCSFitting.sigstr(Inf) == "Inf"
    @test FCSFitting.sigstr(-Inf) == "-Inf"
    @test FCSFitting.sigstr(NaN) == "NaN"
    @test FCSFitting.sigstr(12.3456, 3) == "12.3"
    @test FCSFitting.sigstr(999.999, 4) == "1000"
    @test occursin("e-5", FCSFitting.sigstr(9.99e-5, 3))
    @test occursin("e+6", FCSFitting.sigstr(1.23456e6, 4))
    @test occursin("e-3", FCSFitting.sigstr(9.9999e-4, 4))
    @test FCSFitting.sigstr(1.23000, 5) == "1.23"


    # Error shims when optional extensions are not loaded
    τp = 1e-6:1e-6:1e-4
    Gp = 0.0 .+ 1.0 ./ (1 .+ τp ./ 1e-4)
    chp = FCSChannel("G[1]", collect(τp), Gp, nothing)
    spec_for_plot = FCSModelSpec(; dim=:d2, anom=:none, n_diff=1)

    @test_throws ErrorException FCSFitting.fcs_plot(spec_for_plot, chp, [1.0, 0.0, 1e-3])
    @test_throws ErrorException FCSFitting._fcs_plot(spec_for_plot, chp, [1.0, 0.0, 1e-3])
    @test_throws ErrorException FCSFitting.resid_acf_plot([0.1, -0.1, 0.0])
    @test_throws ErrorException FCSFitting.fcs_table(spec_for_plot, nothing, nothing)
    @test_throws ErrorException FCSFitting.read_fcs("somefile.txt")


    # parameters / errors utilities
    model(x, θ) = @. θ[1] * exp(-x/θ[2]) + θ[3]
    a_true, b_true, c_true = 1.25, 4.0, 0.05
    x = range(0, 10; length=200) |> collect
    y = model(x, [a_true, b_true, c_true])

    θ0 = [1.0, 1.0, 0.0]
    fit = curve_fit(model, x, y, θ0)

    sc = [2.0, 10.0, 0.5]
    p_phys = parameters(fit, sc)
    e_phys = errors(fit, sc)

    @test p_phys ≈ (fit.param .* sc)
    @test e_phys ≈ (LsqFit.stderror(fit) .* sc)
end
