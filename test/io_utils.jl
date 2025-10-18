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


    # FCSChannel & FCSData construction
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


    # infer_parameter_list
    names = FCSFitting.infer_parameter_list(:fcs_2d, zeros(3); diffusivity=nothing, offset=nothing)
    @test names == ["Current amplitude G(0)", "Offset G(∞)", "Diffusion time τ_D [s]"]
    names = FCSFitting.infer_parameter_list(:fcs_2d, zeros(4); diffusivity=1.0, offset=0.0)
    @test names == ["Current amplitude G(0)", "Beam width w₀ [m]", "Dynamic time 1 (τ_dyn) [s]", "Dynamic fraction 1 (K_dyn)"]
    
    @test_throws ArgumentError FCSFitting.infer_parameter_list(:fcs_2d_mdiff, zeros(3); offset=nothing) # no n_diff input
    names = FCSFitting.infer_parameter_list(:fcs_2d_mdiff, zeros(3); n_diff=1, offset=nothing)
    @test names == ["Current amplitude G(0)", "Offset G(∞)", "Diffusion time τ_D[1] [s]"]
    names = FCSFitting.infer_parameter_list(:fcs_2d_mdiff, zeros(5); n_diff=2, offset=nothing)
    @test names == ["Current amplitude G(0)", "Offset G(∞)", "Diffusion time τ_D[1] [s]", 
                    "Diffusion time τ_D[2] [s]", "Population fraction w[1]"]
    names = FCSFitting.infer_parameter_list(:fcs_2d_mdiff, zeros(6); n_diff=2, offset=0.0)
    @test names == ["Current amplitude G(0)", "Diffusion time τ_D[1] [s]", 
                    "Diffusion time τ_D[2] [s]", "Population fraction w[1]",
                    "Dynamic time 1 (τ_dyn) [s]", "Dynamic fraction 1 (K_dyn)"]

    names = FCSFitting.infer_parameter_list(:fcs_2d_anom, zeros(4); diffusivity=nothing, offset=nothing)
    @test names == ["Current amplitude G(0)", "Offset G(∞)", "Diffusion time τ_D [s]", "Anomolous exponent α"]
    names = FCSFitting.infer_parameter_list(:fcs_2d_anom, zeros(5); diffusivity=1.0, offset=0.0)
    @test names == ["Current amplitude G(0)", "Beam width w₀ [m]", "Anomolous exponent α", 
                    "Dynamic time 1 (τ_dyn) [s]", "Dynamic fraction 1 (K_dyn)"]

    @test_throws ArgumentError FCSFitting.infer_parameter_list(:fcs_2d_anom_mdiff, zeros(3); offset=nothing) # no n_diff input
    names = FCSFitting.infer_parameter_list(:fcs_2d_anom_mdiff, zeros(4); n_diff=1, offset=nothing)
    @test names == ["Current amplitude G(0)", "Offset G(∞)", "Diffusion time τ_D[1] [s]", "Anomalous exponent α[1]"]
    names = FCSFitting.infer_parameter_list(:fcs_2d_anom_mdiff, zeros(10); n_diff=2, offset=0.0)
    @test names == ["Current amplitude G(0)", "Diffusion time τ_D[1] [s]", "Diffusion time τ_D[2] [s]", 
                    "Anomalous exponent α[1]", "Anomalous exponent α[2]", "Population fraction w[1]",
                    "Dynamic time 1 (τ_dyn) [s]", "Dynamic time 2 (τ_dyn) [s]", 
                    "Dynamic fraction 1 (K_dyn)", "Dynamic fraction 2 (K_dyn)"]

    names = FCSFitting.infer_parameter_list(:fcs_3d, zeros(4); diffusivity=nothing, offset=nothing)
    @test names == ["Current amplitude G(0)", "Offset G(∞)", "Structure factor κ", "Diffusion time τ_D [s]"]
    names = FCSFitting.infer_parameter_list(:fcs_3d, zeros(5); diffusivity=1.0, offset=0.0)
    @test names == ["Current amplitude G(0)", "Structure factor κ", "Beam width w₀ [m]",
                    "Dynamic time 1 (τ_dyn) [s]", "Dynamic fraction 1 (K_dyn)"]

    @test_throws ArgumentError FCSFitting.infer_parameter_list(:fcs_3d_mdiff, zeros(3); offset=nothing) # no n_diff input
    names = FCSFitting.infer_parameter_list(:fcs_3d_mdiff, zeros(3); n_diff=1, offset=0.0)
    @test names == ["Current amplitude G(0)", "Structure factor κ", "Diffusion time τ_D[1] [s]"]
    names = FCSFitting.infer_parameter_list(:fcs_3d_mdiff, zeros(8); n_diff=2, offset=nothing)
    @test names == ["Current amplitude G(0)", "Offset G(∞)", "Structure factor κ", 
                    "Diffusion time τ_D[1] [s]", "Diffusion time τ_D[2] [s]", "Population fraction w[1]",
                    "Dynamic time 1 (τ_dyn) [s]", "Dynamic fraction 1 (K_dyn)"]

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


    # error message shims when extensions are not loaded
    τ = 1e-6:1e-6:1e-4
    G = 0.0 .+ 1.0 ./ (1 .+ τ ./ 1e-4)
    ch = FCSChannel("G[1]", collect(τ), G, nothing)
    dummy_model = (x, θ; kwargs...) -> (x isa AbstractVector ? similar(x, Float64) .= 1.0 : 1.0)

    @test_throws ErrorException FCSFitting.fcs_plot(dummy_model, ch, [1.0, 0.0, 1e-3])
    @test_throws ErrorException FCSFitting._fcs_plot(dummy_model, ch, [1.0, 0.0, 1e-3])
    @test_throws ErrorException FCSFitting.resid_acf_plot([0.1, -0.1, 0.0])
    @test_throws ErrorException FCSFitting.fcs_table(dummy_model, nothing, nothing)
    @test_throws ErrorException FCSFitting.read_fcs("somefile.txt")


    # parameters and errors
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