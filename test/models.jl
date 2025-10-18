@testset "models" begin
    NA = FCSFitting.AVAGADROS
    kB = FCSFitting.BOLTZMANN
 
    @testset "Calculators" begin
        D = 5e-11
        w0 = 250e-9
        κ = 10 * rand()
        g0 = rand()
        Ks  = [0.1, 0.2]


        # τD and diffusivity
        τ = FCSFitting.τD(D, w0)
        @test τ ≈ (w0^2) / (4D)
        D_back = FCSFitting.diffusivity(τ, w0)
        @test D_back ≈ D
        
        @test_throws ArgumentError FCSFitting.τD(-1.0, w0)
        @test_throws ArgumentError FCSFitting.τD(D, -1.0)
        @test_throws ArgumentError FCSFitting.diffusivity(-1.0, w0)
        @test_throws ArgumentError FCSFitting.diffusivity(τ, -1.0)


        # confocal volume/ area
        vol = FCSFitting.volume(w0, κ)
        @test vol ≈ π^(3/2) * w0^3 * κ rtol=1e-12
        ar = FCSFitting.area(w0)
        @test ar ≈ π * w0^2 rtol=1e-12
        
        @test_throws ArgumentError FCSFitting.volume(-1.0, κ)
        @test_throws ArgumentError FCSFitting.volume(w0, -1.0)
        @test_throws ArgumentError FCSFitting.area(-1.0)


        # concentration
        c = FCSFitting.concentration(w0, κ, g0)
        c_expected = (1/g0) / (NA * FCSFitting.volume(w0, κ) * 1000.0)
        @test c ≈ c_expected rtol=1e-12
        
        @test_throws ArgumentError FCSFitting.concentration(-1.0, κ, g0)
        @test_throws ArgumentError FCSFitting.concentration(w0, -1.0, g0)
        @test_throws ArgumentError FCSFitting.concentration(w0, κ, -1.0)
        @test_throws ArgumentError FCSFitting.concentration(w0, κ, g0; Ks=[-0.1])
        @test_throws ArgumentError FCSFitting.concentration(w0, κ, g0; Ks=[0.1, 0.2], ics=[1])  # sum(ics) ≠ length(Ks)
        @test_throws ArgumentError FCSFitting.concentration(w0, κ, g0; Ks=[], ics=[1]) # with Ks=[], ics must be [0]
        
        B0 = 1 + (0.1/0.9 + 0.2/0.8)
        Neff1 = B0 / g0
        c_expected = Neff1 / (NA * FCSFitting.volume(w0, κ) * 1000.0)
        c = FCSFitting.concentration(w0, κ, g0; Ks=Ks, ics=[2])
        @test c ≈ c_expected

        B0 = (1 + 0.1/0.9) * (1 + 0.2/0.8)
        Neff2 = B0 / g0
        c_expected = Neff2 / (NA * FCSFitting.volume(w0, κ) * 1000.0)
        c = FCSFitting.concentration(w0, κ, g0; Ks=Ks, ics=[1,1])
        @test c ≈ c_expected


        # surface density
        sa = FCSFitting.surface_density(w0, g0)
        sa_expected = (1/g0) / (NA * FCSFitting.area(w0))
        @test sa ≈ sa_expected rtol=1e-12
        
        @test_throws ArgumentError FCSFitting.surface_density(-1.0, g0)
        @test_throws ArgumentError FCSFitting.surface_density(w0, -1.0)
        @test_throws ArgumentError FCSFitting.surface_density(w0, g0; Ks=[-0.1])
        @test_throws ArgumentError FCSFitting.surface_density(w0, g0; Ks=[0.1, 0.2], ics=[1])  # sum(ics) ≠ length(Ks)
        @test_throws ArgumentError FCSFitting.surface_density(w0, g0; Ks=[], ics=[1]) # with Ks=[], ics must be [0]
        
        sa_expected = Neff1 / (NA * FCSFitting.area(w0))
        sa = FCSFitting.surface_density(w0, g0; Ks=Ks, ics=[2])
        @test sa ≈ sa_expected

        sa_expected = Neff2 / (NA * FCSFitting.area(w0))
        sa = FCSFitting.surface_density(w0, g0; Ks=Ks, ics=[1,1])
        @test sa ≈ sa_expected


        # hydrodynamic radius
        T = 293.0
        η = 1.0016e-3
        Rh_expected = kB * T / (6π * η * D)
        Rh = FCSFitting.hydrodynamic(D; T=T, η=η)
        @test Rh ≈ Rh_expected rtol=1e-12

        @test_throws ArgumentError FCSFitting.hydrodynamic(-1.0; T=T, η=η)
        @test_throws ArgumentError FCSFitting.hydrodynamic(D; T=-1.0, η=η)
        @test_throws ArgumentError FCSFitting.hydrodynamic(D; T=T, η=-1.0)

        Rh2, Rh2_err = FCSFitting.hydrodynamic(D; T=T, η=η, D_err=1e-12)
        @test Rh2 ≈ Rh_expected rtol=1e-12
        @test Rh2_err ≈ (kB*T/(6π*η)) * (1e-12) / D^2 rtol=1e-12
    end

    @testset "Low Level Helpers" begin
        # _ndyn_from_len
        @test FCSFitting._ndyn_from_len(0) == 0
        @test FCSFitting._ndyn_from_len(2) == 1
        @test FCSFitting._ndyn_from_len(10) == 5
        
        @test_throws ArgumentError FCSFitting._ndyn_from_len(-2)
        @test_throws ArgumentError FCSFitting._ndyn_from_len(1)
        @test_throws ArgumentError FCSFitting._ndyn_from_len(3)
        

        # _dynamics_factor
        N = 50
        τ = 1e-3; K = rand()
        τs = [1e-4, 1e-5]; Ks = rand(2)
        t = 10.0 .^ (range(-6, -3, length=N))
        
        @test FCSFitting._dynamics_factor(t, Float64[], Float64[], Int[0]) == ones(N)
        @test FCSFitting._dynamics_factor(t[1], Float64[], Float64[], Int[0]) == 1.0
        
        expected_vec = @. 1 + K * (exp(-t/τ) - 1)
        expected_sca = 1 + K * (exp(-t[1]/τ) - 1)
        @test FCSFitting._dynamics_factor(t, [τ], [K], [1]) ≈ expected_vec rtol=1e-12
        @test FCSFitting._dynamics_factor(t[1], [τ], [K], [1]) ≈ expected_sca rtol=1e-12
        
        prod_vec = ( @. 1 + Ks[1]*(exp(-t/τs[1]) - 1) ) .* ( @. 1 + Ks[2]*(exp(-t/τs[2]) - 1) )
        prod_sca = (1 + Ks[1]*(exp(-t[1]/τs[1]) - 1))*(1 + Ks[2]*(exp(-t[1]/τs[2]) - 1))
        @test FCSFitting._dynamics_factor(t, τs, Ks, [1,1]) ≈ prod_vec rtol=1e-12
        @test FCSFitting._dynamics_factor(t[1], τs, Ks, [1,1]) ≈ prod_sca rtol=1e-12
        
        blk_vec = @. 1 + Ks[1]*(exp(-t/τs[1]) - 1) + Ks[2]*(exp(-t/τs[2]) - 1)
        blk_sca = 1 + Ks[1]*(exp(-t[1]/τs[1]) - 1) + Ks[2]*(exp(-t[1]/τs[2]) - 1)
        @test FCSFitting._dynamics_factor(t, τs, Ks, [2]) ≈ blk_vec rtol=1e-12
        @test FCSFitting._dynamics_factor(t[1], τs, Ks, [2]) ≈ blk_sca rtol=1e-12

        t32 = Float32[1f-5, 2f-5]
        out32 = FCSFitting._dynamics_factor(t32, [1e-4], [0.1], [1])
        @test eltype(out32) == Float64

        @test_throws ArgumentError FCSFitting._dynamics_factor(t, [1e-4], [0.1, 0.2], [2]) # length mismatch
        @test_throws ArgumentError FCSFitting._dynamics_factor(t, [1e-4, 1e-5], [0.1], [2]) # length mismatch
        @test_throws ArgumentError FCSFitting._dynamics_factor(t, [1e-4], [1.2], [1]) # K out of range
        @test_throws ArgumentError FCSFitting._dynamics_factor(t, [1e-4], [0.2], [2]) # sum(ics) mismatch


        # _mdiff
        τDs = 1e-3; wts = Float64[]
        τDs2 = [1e-4, 1e-3]; wts2 = [0.3]

        @test FCSFitting._mdiff(t, [τDs], wts, FCSFitting.udc_2d) == FCSFitting.udc_2d(t, τDs)
        @test FCSFitting._mdiff(t[1], [τDs], wts, FCSFitting.udc_2d) == FCSFitting.udc_2d(t[1], τDs)

        mix_ud = @. 0.3*FCSFitting.udc_2d(t, τDs2[1]) + 0.7*FCSFitting.udc_2d(t, τDs2[2])
        @test FCSFitting._mdiff(t, τDs2, wts2, FCSFitting.udc_2d) ≈ mix_ud rtol=1e-12
        @test FCSFitting._mdiff(t[1], τDs2, wts2, FCSFitting.udc_2d) ≈ mix_ud[1] rtol=1e-12

        @test_throws ArgumentError FCSFitting._mdiff(t, Float64[], wts2, FCSFitting.udc_2d) # n<1
        @test_throws ArgumentError FCSFitting._mdiff(t, τDs2, Float64[], FCSFitting.udc_2d) # wrong weights length
        @test_throws ArgumentError FCSFitting._mdiff(t, τDs2, [1.1], FCSFitting.udc_2d)


        # _mdiff_anom
        αs = 2 * rand()
        τDs3 = [1e-5, 1e-4, 1e-3]; αs3  = [0.7, 0.9, 1.0]; wts3 = [0.2, 0.3]
        @test FCSFitting._mdiff_anom(t, [τDs], [αs], wts, FCSFitting.udc_2d_anom) ≈ FCSFitting.udc_2d_anom(t, τDs, αs)

        mix = 0.2 .* FCSFitting.udc_2d_anom(t, τDs3[1], αs3[1]) .+
              0.3 .* FCSFitting.udc_2d_anom(t, τDs3[2], αs3[2]) .+
              0.5 .* FCSFitting.udc_2d_anom(t, τDs3[3], αs3[3])
        @test FCSFitting._mdiff_anom(t, τDs3, αs3, wts3, FCSFitting.udc_2d_anom) ≈ mix rtol=1e-12
        @test FCSFitting._mdiff_anom(t[1], τDs3, αs3, wts3, FCSFitting.udc_2d_anom) ≈ mix[1] rtol=1e-12

        @test_throws ArgumentError FCSFitting._mdiff_anom(t, τDs3, [0.8], wts3, FCSFitting.udc_2d_anom) # α length mismatch
        @test_throws ArgumentError FCSFitting._mdiff_anom(t, τDs3, αs3, [0.2], FCSFitting.udc_2d_anom) # weights length
        @test_throws ArgumentError FCSFitting._mdiff_anom(t, τDs3, αs3, [0.8, 0.3], FCSFitting.udc_2d_anom) # sum(wts) > 1
    end

    @testset "Diffusion Kernels" begin
        τD = 1e-3
        α = 2 * rand()
        κ = 10 * rand()
        t = rand()
        ts = rand(100)


        # 2D with Brownian motion
        udc = FCSFitting.udc_2d(t, τD)
        udc_expected = inv(1 + t/τD)
        @test udc ≈ udc_expected
        @test isreal(udc)
        udcs = FCSFitting.udc_2d(ts, τD)
        udcs_expected = @. inv(1 + ts/τD)
        @test udcs ≈ udcs_expected
        @test all(isreal, udcs)


        # 2D with anomalous diffusion
        udc = FCSFitting.udc_2d_anom(t, τD, α)
        udc_expected = inv(1 + (t/τD)^α)
        @test udc ≈ udc_expected
        @test isreal(udc)
        udcs = FCSFitting.udc_2d_anom(ts, τD, α)
        udcs_expected = @. inv(1 + (ts/τD)^α)
        @test udcs ≈ udcs_expected
        @test all(isreal, udcs)


        # 3D with Brownian motion
        udc = FCSFitting.udc_3d(t, τD, κ)
        udc_expected = inv((1 + t/τD) * sqrt(1 + t/(κ^2 * τD)))
        @test udc ≈ udc_expected
        @test isreal(udc)
        udcs = FCSFitting.udc_3d(ts, τD, κ)
        udcs_expected = @. inv((1 + ts/τD) * sqrt(1 + ts/(κ^2 * τD)))
        @test udcs ≈ udcs_expected
        @test all(isreal, udcs)


        # 3D with anomalous diffusion
        udc = FCSFitting.udc_3d_anom(t, τD, κ, α)
        udc_expected = inv((1 + (t/τD)^α) * sqrt(1 + (t/τD)^α / κ^2))
        @test udc ≈ udc_expected
        @test isreal(udc)
        udcs = FCSFitting.udc_3d_anom(ts, τD, κ, α)
        udcs_expected = @. inv((1 + (ts/τD)^α) * sqrt(1 + (ts/τD)^α / κ^2))
        @test udcs ≈ udcs_expected
        @test all(isreal, udcs)
    end

    @testset "fcs_2d" begin
        t = 10 .^ range(-6, 0, length=200)
        g0 = rand()
        off = 1e-2 * (randn() - 0.5)
        τD = 1e-3

        p = [g0, off, τD]
        G = fcs_2d(t, p) # no dynamics
        @test G ≈ off .+ g0 .* FCSFitting.udc_2d(t, τD)
        

        Ks = [0.1, 0.2]; τs = [1e-5, 1e-4]; ics = [1,1]
        p = vcat(g0, off, τD, τs, Ks)
        G = fcs_2d(t, p; ics=ics) # two independent dynamic terms
        @test G ≈ off .+ g0 .* FCSFitting.udc_2d(t, τD) .* FCSFitting._dynamics_factor(t, τs, Ks, ics)


        p = vcat(g0, τD, τs, Ks)
        G = fcs_2d(t, p; ics=ics, offset=off) # fixed offset
        @test G ≈ off .+ g0 .* FCSFitting.udc_2d(t, τD) .* FCSFitting._dynamics_factor(t, τs, Ks, ics)


        D  = 5e-11; w0 = 250e-9
        τD_fix = FCSFitting.τD(D, w0)
        p = [g0, off, w0]
        G = fcs_2d(t, p; diffusivity=D) # fixed diffusivity
        @test G ≈ off .+ g0 .* FCSFitting.udc_2d(t, τD_fix)
    end

    @testset "fcs_2d_mdiff" begin
        t = 10 .^ range(-6, 0, length=200)
        g0 = rand()
        off = 1e-2 * (randn() - 0.5)

        τDs1 = [1e-3]
        p = vcat(g0, off, τDs1)
        G = fcs_2d_mdiff(t, p; n_diff=1)
        @test G ≈ off .+ g0 .* FCSFitting.udc_2d(t, τDs1[1])


        τDs2 = [1e-4, 2e-3]; wts = [0.3]
        p = vcat(g0, off, τDs2, wts)
        mix = @. 0.3 * FCSFitting.udc_2d(t, τDs2[1]) + 0.7 * FCSFitting.udc_2d(t, τDs2[2])
        G = FCSFitting.fcs_2d_mdiff(t, p; n_diff=2)
        @test G ≈ off .+ g0 .* mix


        Ks = [0.15]; τs = [5e-5]; ics = [1]
        p = vcat(g0, off, τDs2, wts, τs, Ks)
        G = fcs_2d_mdiff(t, p; n_diff=2, ics=ics)
        @test G ≈ off .+ g0 .* mix .* FCSFitting._dynamics_factor(t, τs, Ks, ics)
    end

    @testset "fcs_2d_anom" begin
        t = 10 .^ range(-6, 0, length=200)
        g0 = rand()
        off = 1e-2 * (randn() - 0.5)
        τD_true = 1e-3 * rand()
        α = 2 * rand()

        p = [g0, off, τD_true, α]
        G = fcs_2d_anom(t, p)
        @test G ≈ off .+ g0 .* FCSFitting.udc_2d_anom(t, τD_true, α)


        Ks = [0.2]; τs=[2e-5]; ics=[1]
        p = vcat(g0, off, τD_true, α, τs, Ks)
        G = fcs_2d_anom(t, p; ics=ics)
        @test G ≈ off .+ g0 .* FCSFitting.udc_2d_anom(t, τD_true, α) .* FCSFitting._dynamics_factor(t, τs, Ks, ics)


        D = 5e-11; w0 = 250e-9; τD_fix = τD(D, w0)
        p = [g0, off, w0, α]
        G = fcs_2d_anom(t, p; diffusivity=D)
        @test G ≈ off .+ g0 .* FCSFitting.udc_2d_anom(t, τD_fix, α)
    end

    @testset "fcs_2d_mdiff_anom" begin
        t = 10 .^ range(-6, 0, length=200)
        g0 = rand()
        off = 1e-2 * (randn() - 0.5)
        τDs = [5e-4, 2e-3, 8e-3]
        αs  = [0.8, 0.95, 1.0]
        wts = [0.2, 0.3]
        p = vcat(g0, off, τDs, αs, wts)

        mix = 0.2 .* FCSFitting.udc_2d_anom(t, τDs[1], αs[1]) .+
              0.3 .* FCSFitting.udc_2d_anom(t, τDs[2], αs[2]) .+
              0.5 .* FCSFitting.udc_2d_anom(t, τDs[3], αs[3])
        G = fcs_2d_anom_mdiff(t, p; n_diff=3)
        @test G ≈ off .+ g0 .* mix
    end
    
    @testset "fcs_3d" begin
        t = 10 .^ range(-6, 0, length=200)
        g0 = rand()
        κ = 10 * rand()
        off = 1e-2 * (randn() - 0.5)
        τD_true = 1e-3

        p = [g0, off, κ, τD_true]
        G = fcs_3d(t, p)
        @test G ≈ off .+ g0 .* FCSFitting.udc_3d(t, τD_true, κ)


        τs = [1e-5]; Ks=[0.1]; ics=[1]
        p = vcat(g0, off, κ, τD_true, τs, Ks)
        G = fcs_3d(t, p; ics=ics)
        @test G ≈ off .+ g0 .* FCSFitting.udc_3d(t, τD_true, κ) .* FCSFitting._dynamics_factor(t, τs, Ks, ics)

        D = 5e-11; w0 = 250e-9; τD_fix = τD(D, w0)
        p = [g0, off, κ, w0]
        G = fcs_3d(t, p; diffusivity=D)
        @test G ≈ off .+ g0 .* FCSFitting.udc_3d(t, τD_fix, κ)
    end

    @testset "fcs_3d_mdiff" begin
        t = 10 .^ range(-6, 0, length=200)
        g0 = rand()
        κ = 10 * rand()
        off = 1e-2 * (randn() - 0.5)


        τDs = [1e-4, 2e-3]
        wts = [0.25]
        p = vcat(g0, off, κ, τDs, wts)
        mix = @. 0.25 * FCSFitting.udc_3d(t, τDs[1], κ) + 0.75 * FCSFitting.udc_3d(t, τDs[2], κ)
        G = fcs_3d_mdiff(t, p; n_diff=2)
        @test G ≈ off .+ g0 .* mix


        τs = [1e-5, 5e-5]; Ks=[0.1, 0.2]; ics=[1,1]
        p = vcat(g0, off, κ, τDs, wts, τs, Ks)
        G = fcs_3d_mdiff(t, p; n_diff=2, ics=ics)
        @test G ≈ off .+ g0 .* mix .* FCSFitting._dynamics_factor(t, τs, Ks, ics)
    end
end