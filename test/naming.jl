@testset "naming" begin
    # SI_PREFIXES
    @test FluoCorrFitting.SI_PREFIXES[""]  == 1.0
    @test FluoCorrFitting.SI_PREFIXES["d"] == 1e1
    @test FluoCorrFitting.SI_PREFIXES["c"] == 1e2
    @test FluoCorrFitting.SI_PREFIXES["m"] == 1e3
    @test FluoCorrFitting.SI_PREFIXES["μ"] == 1e6
    @test FluoCorrFitting.SI_PREFIXES["u"] == 1e6
    @test FluoCorrFitting.SI_PREFIXES["n"] == 1e9
    @test FluoCorrFitting.SI_PREFIXES["p"] == 1e12


    # Parameter name inference helpers
    # Convenience aliases to constants used in names
    G0 = FluoCorrFitting.G0_NAME
    OFF = FluoCorrFitting.OFF_NAME
    AN = FluoCorrFitting.ANOM_NAME
    ST = FluoCorrFitting.STRUCT_NAME
    RT = FluoCorrFitting.DIFFTIME_NAME
    WF = FluoCorrFitting.DIFFFRAC_NAME
    BW = FluoCorrFitting.BEAM_NAME
    DT = FluoCorrFitting.DYNTIME_NAME
    DF = FluoCorrFitting.DYNFRAC_NAME

    # 2D, Brownian, single diffuser, free offset
    spec_2d = FCSModelSpec(; dim=FluoCorrFitting.d2)
    @test FluoCorrFitting.expected_parameter_names(spec_2d) ==
        [G0, OFF, "$(RT) [s]", "$(DT) [1:m] [s]", "$(DF) [1:m]"]

    # 2D, Brownian, single diffuser, fixed offset (removed from p), still τD slots
    spec_2d_fixoff = FCSModelSpec(; dim=FluoCorrFitting.d2, offset=0.0)
    @test FluoCorrFitting.expected_parameter_names(spec_2d_fixoff)[1:1] == [G0]  # no OFF in front matter

    # 2D, diffusion given by w0 (fixed D in spec) → label should be Beam width
    spec_2d_w0 = FCSModelSpec(; dim=FluoCorrFitting.d2, diffusivity=5e-11)
    # Only checking the base portion that changes wording
    base2d = FluoCorrFitting.infer_parameter_names(spec_2d_w0, [0.0, 0.0, 0.0])  # g0, off fixed? no → expect G0, OFF, w0, (no dynamics)
    @test base2d[1:3] == [G0, OFF, "$(BW) [m]"]

    # 2D, anomalous (global α), n_diff=1
    spec_2d_ag = FCSModelSpec(; dim=FluoCorrFitting.d2, anom=FluoCorrFitting.globe)
    nd_ag = FluoCorrFitting._no_dynamics_params(spec_2d_ag)
    @test nd_ag == [G0, OFF, "$(RT) [s]", AN]
    # With one dynamics block (τ_dyn1, K_dyn1)
    names_ag = FluoCorrFitting.infer_parameter_names(spec_2d_ag, zeros(1 + 1 + 1 + 1 + 2))  # g0, off, τD, α, (τ,K)
    @test names_ag[end-1:end] == ["$(DT) 1 [s]", "$(DF) 1"]

    # 2D, anomalous per-pop, n_diff=2 (α1, α2) + 1 weight
    spec_2d_ap = FCSModelSpec(; dim=FluoCorrFitting.d2, anom=FluoCorrFitting.perpop, n_diff=2)
    nd_ap = FluoCorrFitting._no_dynamics_params(spec_2d_ap)
    @test nd_ap == [G0, OFF, "$(RT) 1 [s]", "$(RT) 2 [s]", "$(AN) 1", "$(AN) 2", "$(WF) 1"]

    # 3D, Brownian, single diffuser
    spec_3d = FCSModelSpec(; dim=FluoCorrFitting.d3)
    nd_3d = FluoCorrFitting._no_dynamics_params(spec_3d)
    @test nd_3d == [G0, OFF, ST, "$(RT) [s]"]

    # 3D, anomalous (global), with dynamics count inferred from params length
    spec_3d_ag = FCSModelSpec(; dim=FluoCorrFitting.d3, anom=FluoCorrFitting.globe)
    # g0, off, κ, τD, α, (τ1,K1), (τ2,K2)  -> total 5 + 4 = 9
    names_3d_ag = FluoCorrFitting.infer_parameter_names(spec_3d_ag, zeros(9))
    @test names_3d_ag[1:5] == [G0, OFF, ST, "$(RT) [s]", AN]
    @test names_3d_ag[6:9] == ["$(DT) 1 [s]", "$(DT) 2 [s]", "$(DF) 1", "$(DF) 2"]


    # sigstr
    @test FluoCorrFitting.sigstr(0.0) == "0"
    @test FluoCorrFitting.sigstr(Inf) == "Inf"
    @test FluoCorrFitting.sigstr(-Inf) == "-Inf"
    @test FluoCorrFitting.sigstr(NaN) == "NaN"
    @test FluoCorrFitting.sigstr(12.3456, 3) == "12.3"
    @test FluoCorrFitting.sigstr(999.999, 4) == "1000"
    @test occursin("e-5", FluoCorrFitting.sigstr(9.99e-5, 3))
    @test occursin("e+6", FluoCorrFitting.sigstr(1.23456e6, 4))
    @test occursin("e-3", FluoCorrFitting.sigstr(9.9999e-4, 4))
    @test FluoCorrFitting.sigstr(1.23000, 5) == "1.23"
end
