@testset "models" begin
    NA = FCSFitting.AVAGADROS
    kB = FCSFitting.BOLTZMANN

    @testset "Calculators" begin
        D  = 5e-11
        w0 = 250e-9
        κ  = 10 * rand()
        g0 = rand()
        Ks = [0.1, 0.2]

        # τD and diffusivity
        τ = FCSFitting.τD(D, w0)
        @test τ ≈ (w0^2) / (4D)
        D_back = FCSFitting.diffusivity(τ, w0)
        @test D_back ≈ D

        @test_throws ArgumentError FCSFitting.τD(-1.0, w0)
        @test_throws ArgumentError FCSFitting.τD(D, -1.0)
        @test_throws ArgumentError FCSFitting.diffusivity(-1.0, w0)
        @test_throws ArgumentError FCSFitting.diffusivity(τ, -1.0)

        # confocal volume/area
        vol = FCSFitting.volume(w0, κ)
        @test vol ≈ π^(3/2) * w0^3 * κ rtol=1e-12
        ar = FCSFitting.area(w0)
        @test ar ≈ π * w0^2 rtol=1e-12

        @test_throws ArgumentError FCSFitting.volume(-1.0, κ)
        @test_throws ArgumentError FCSFitting.volume(w0, -1.0)
        @test_throws ArgumentError FCSFitting.area(-1.0)

        # concentration (blinkless)
        c = FCSFitting.concentration(w0, κ, g0)
        c_expected = (1/g0) / (NA * vol * 1000.0)
        @test c ≈ c_expected rtol=1e-12

        @test_throws ArgumentError FCSFitting.concentration(-1.0, κ, g0)
        @test_throws ArgumentError FCSFitting.concentration(w0, -1.0, g0)
        @test_throws ArgumentError FCSFitting.concentration(w0, κ, -1.0)
        @test_throws ArgumentError FCSFitting.concentration(w0, κ, g0; Ks=[-0.1])
        @test_throws ArgumentError FCSFitting.concentration(w0, κ, g0; Ks=[0.1,0.2], ics=[1])
        @test_throws ArgumentError FCSFitting.concentration(w0, κ, g0; Ks=[], ics=[1])

        B0 = 1 + (0.1/0.9 + 0.2/0.8)
        Neff1 = B0 / g0
        c_expected = Neff1 / (NA * vol * 1000.0)
        @test FCSFitting.concentration(w0, κ, g0; Ks=Ks, ics=[2]) ≈ c_expected

        B0 = (1 + 0.1/0.9) * (1 + 0.2/0.8)
        Neff2 = B0 / g0
        c_expected = Neff2 / (NA * vol * 1000.0)
        @test FCSFitting.concentration(w0, κ, g0; Ks=Ks, ics=[1,1]) ≈ c_expected

        # surface density (2D analogue)
        sa = FCSFitting.surface_density(w0, g0)
        sa_expected = (1/g0) / (NA * ar)
        @test sa ≈ sa_expected rtol=1e-12

        @test_throws ArgumentError FCSFitting.surface_density(-1.0, g0)
        @test_throws ArgumentError FCSFitting.surface_density(w0, -1.0)
        @test_throws ArgumentError FCSFitting.surface_density(w0, g0; Ks=[-0.1])
        @test_throws ArgumentError FCSFitting.surface_density(w0, g0; Ks=[0.1,0.2], ics=[1])
        @test_throws ArgumentError FCSFitting.surface_density(w0, g0; Ks=[], ics=[1])

        sa_expected = Neff1 / (NA * ar)
        @test FCSFitting.surface_density(w0, g0; Ks=Ks, ics=[2]) ≈ sa_expected
        sa_expected = Neff2 / (NA * ar)
        @test FCSFitting.surface_density(w0, g0; Ks=Ks, ics=[1,1]) ≈ sa_expected

        # hydrodynamic radius
        T  = 293.0
        η  = 1.0016e-3
        Rh_expected = kB * T / (6π * η * D)
        Rh = FCSFitting.hydrodynamic(D; T=T, η=η)
        @test Rh ≈ Rh_expected rtol=1e-12

        @test_throws ArgumentError FCSFitting.hydrodynamic(-1.0; T=T, η=η)
        @test_throws ArgumentError FCSFitting.hydrodynamic(D; T=-1.0, η=η)
        @test_throws ArgumentError FCSFitting.hydrodynamic(D; T=T, η=-1.0)

        Rh2, Rh2err = FCSFitting.hydrodynamic(D; T=T, η=η, D_err=1e-12)
        @test Rh2 ≈ Rh_expected rtol=1e-12
        @test Rh2err ≈ (kB*T/(6π*η)) * (1e-12) / D^2 rtol=1e-12
    end


    @testset "Low-level helpers" begin
        # _ndyn_from_len
        @test FCSFitting._ndyn_from_len(0)  == 0
        @test FCSFitting._ndyn_from_len(2)  == 1
        @test FCSFitting._ndyn_from_len(10) == 5
        @test_throws ArgumentError FCSFitting._ndyn_from_len(-2)
        @test_throws ArgumentError FCSFitting._ndyn_from_len(1)
        @test_throws ArgumentError FCSFitting._ndyn_from_len(3)

        # dynamics_factor
        N   = 50
        τ   = 1e-3
        K   = rand()
        τs  = [1e-4, 1e-5]
        Ks  = rand(2)
        t   = 10.0 .^ (range(-6, -3, length=N))

        @test FCSFitting.dynamics_factor(t, Float64[], Float64[], Int[0]) == ones(N)
        @test FCSFitting.dynamics_factor(first(t), Float64[], Float64[], Int[0]) == 1.0

        expected_vec = @. 1 + K * (exp(-t/τ) - 1)
        expected_sca = 1 + K * (exp(-first(t)/τ) - 1)
        @test FCSFitting.dynamics_factor(t, [τ], [K], [1]) ≈ expected_vec rtol=1e-12
        @test FCSFitting.dynamics_factor(first(t), [τ], [K], [1]) ≈ expected_sca rtol=1e-12

        prod_vec = (@. 1 + Ks[1]*(exp(-t/τs[1]) - 1)) .* (@. 1 + Ks[2]*(exp(-t/τs[2]) - 1))
        prod_sca = (1 + Ks[1]*(exp(-t[1]/τs[1]) - 1))*(1 + Ks[2]*(exp(-t[1]/τs[2]) - 1))
        @test FCSFitting.dynamics_factor(t, τs, Ks, [1,1]) ≈ prod_vec rtol=1e-12
        @test FCSFitting.dynamics_factor(t[1], τs, Ks, [1,1]) ≈ prod_sca rtol=1e-12

        blk_vec = @. 1 + Ks[1]*(exp(-t/τs[1]) - 1) + Ks[2]*(exp(-t/τs[2]) - 1)
        blk_sca = 1 + Ks[1]*(exp(-t[1]/τs[1]) - 1) + Ks[2]*(exp(-t[1]/τs[2]) - 1)
        @test FCSFitting.dynamics_factor(t, τs, Ks, [2]) ≈ blk_vec rtol=1e-12
        @test FCSFitting.dynamics_factor(t[1], τs, Ks, [2]) ≈ blk_sca rtol=1e-12

        t32 = Float32[1f-5, 2f-5]
        out32 = FCSFitting.dynamics_factor(t32, [1e-4], [0.1], [1])
        @test eltype(out32) == Float64  # promotion behavior in current impl

        @test_throws ArgumentError FCSFitting.dynamics_factor(t, [1e-4], [0.1,0.2], [2])
        @test_throws ArgumentError FCSFitting.dynamics_factor(t, [1e-4,1e-5], [0.1], [2])
        @test_throws ArgumentError FCSFitting.dynamics_factor(t, [1e-4], [1.2], [1])
        @test_throws ArgumentError FCSFitting.dynamics_factor(t, [1e-4], [0.2], [2])

        # diff_factor (2D/3D, Brownian/anomalous)
        τDs1 = [1e-3]; wts1 = Float64[]
        @test FCSFitting.diff_factor(t, nothing, τDs1, nothing, wts1) == FCSFitting.udc_2d(t, τDs1[1])
        @test FCSFitting.diff_factor(t[1], nothing, τDs1, nothing, wts1) == FCSFitting.udc_2d(t[1], τDs1[1])

        τDs2 = [1e-4, 1e-3]; wts2 = [0.3]
        mix2d = @. 0.3*FCSFitting.udc_2d(t, τDs2[1]) + 0.7*FCSFitting.udc_2d(t, τDs2[2])
        @test FCSFitting.diff_factor(t, nothing, τDs2, nothing, wts2) ≈ mix2d rtol=1e-12
        @test FCSFitting.diff_factor(t[1], nothing, τDs2, nothing, wts2) ≈ mix2d[1] rtol=1e-12

        αs3  = [0.7, 0.9, 1.0]
        τDs3 = [1e-5, 1e-4, 1e-3]; wts3 = [0.2, 0.3]
        mix2d_anom = 0.2 .* FCSFitting.udc_2d(t, τDs3[1], αs3[1]) .+
                     0.3 .* FCSFitting.udc_2d(t, τDs3[2], αs3[2]) .+
                     0.5 .* FCSFitting.udc_2d(t, τDs3[3], αs3[3])
        @test FCSFitting.diff_factor(t, nothing, τDs3, αs3, wts3) ≈ mix2d_anom rtol=1e-12

        κ3d = 10*rand()
        mix3d = @. 0.3*FCSFitting.udc_3d(t, τDs2[1], κ3d) + 0.7*FCSFitting.udc_3d(t, τDs2[2], κ3d)
        @test FCSFitting.diff_factor(t, κ3d, τDs2, nothing, wts2) ≈ mix3d rtol=1e-12

        mix3d_anom = 0.2 .* FCSFitting.udc_3d(t, τDs3[1], κ3d, αs3[1]) .+
                      0.3 .* FCSFitting.udc_3d(t, τDs3[2], κ3d, αs3[2]) .+
                      0.5 .* FCSFitting.udc_3d(t, τDs3[3], κ3d, αs3[3])
        @test FCSFitting.diff_factor(t, κ3d, τDs3, αs3, wts3) ≈ mix3d_anom rtol=1e-12

        # error paths
        @test_throws ArgumentError FCSFitting.diff_factor(t, nothing, Float64[], nothing, wts2)
        @test_throws ArgumentError FCSFitting.diff_factor(t, nothing, τDs2, [0.8], [0.2,0.3]) # α length mismatch
        @test_throws ArgumentError FCSFitting.diff_factor(t, nothing, τDs2, nothing, Float64[]) # weights length
        @test_throws ArgumentError FCSFitting.diff_factor(t, nothing, τDs2, nothing, [1.1]) # sum(wts)>1
    end

    
    @testset "Diffusion kernels" begin
        τD = 1e-3
        α  = 0.5 + 1.5*rand()
        κ  = 10 * rand()
        t  = rand()
        ts = rand(100)

        # 2D Brownian
        @test FCSFitting.udc_2d(t, τD) ≈ inv(1 + t/τD)
        @test all(isreal, FCSFitting.udc_2d(ts, τD))

        # 2D anomalous
        @test FCSFitting.udc_2d(t, τD, α) ≈ inv(1 + (t/τD)^α)
        @test all(isreal, FCSFitting.udc_2d(ts, τD, α))

        # 3D Brownian
        @test FCSFitting.udc_3d(t, τD, κ) ≈ inv((1 + t/τD) * sqrt(1 + t/(κ^2 * τD)))
        @test all(isreal, FCSFitting.udc_3d(ts, τD, κ))

        # 3D anomalous
        @test FCSFitting.udc_3d(t, τD, κ, α) ≈ inv((1 + (t/τD)^α) * sqrt(1 + (t/τD)^α / κ^2))
        @test all(isreal, FCSFitting.udc_3d(ts, τD, κ, α))
    end


    @testset "FCSModel end-to-end" begin
        t = 10 .^ range(-6, 0, length=200)
        g0  = 0.8 + 0.4rand()
        off = 1e-2 * (randn() - 0.5)

        # 2D Brownian, free offset
        τD = 1e-3
        spec = FCSFitting.FCSModel(; spec=FCSFitting.FCSModelSpec(; dim=:d2, anom=:none, n_diff=1))
        p = [g0, off, τD]
        @test spec(t, p) ≈ @. off + g0 * FCSFitting.udc_2d(t, τD)

        # 2D Brownian, fixed offset in spec (omit offset from p)
        spec_fixoff = FCSFitting.FCSModel(; spec=FCSFitting.FCSModelSpec(; dim=:d2, anom=:none, n_diff=1, offset=off))
        p_fixoff = [g0, τD]
        @test spec_fixoff(t, p_fixoff) ≈ @. off + g0 * FCSFitting.udc_2d(t, τD)

        # 2D with dynamics (two independent components)
        τs = [1e-5, 1e-4]; Ks = [0.1, 0.2]; ics = [1,1]
        spec_dyn = FCSFitting.FCSModel(; spec=FCSFitting.FCSModelSpec(; dim=:d2, anom=:none, n_diff=1, ics=ics))
        p_dyn = vcat(g0, off, τD, τs, Ks)
        dynfac = FCSFitting.dynamics_factor(t, τs, Ks, ics)
        @test spec_dyn(t, p_dyn) ≈ @. off + g0 * FCSFitting.udc_2d(t, τD) * dynfac

        # 2D anomalous, global α
        α = 0.8
        spec_ag = FCSFitting.FCSModel(; spec=FCSFitting.FCSModelSpec(; dim=:d2, anom=:global, n_diff=1))
        p_ag = [g0, off, τD, α]
        @test spec_ag(t, p_ag) ≈ @. off + g0 * FCSFitting.udc_2d(t, τD, α)

        # 2D anomalous, per-population α, mixture of 2 diffusers
        τDs2 = [1e-4, 2e-3]; αs2 = [0.7, 1.0]; wts = [0.3]
        spec_per = FCSFitting.FCSModel(; spec=FCSFitting.FCSModelSpec(; dim=:d2, anom=:perpop, n_diff=2))
        p_per = vcat(g0, off, τDs2, αs2, wts)
        mix = @. 0.3*FCSFitting.udc_2d(t, τDs2[1], αs2[1]) + 0.7*FCSFitting.udc_2d(t, τDs2[2], αs2[2])
        @test spec_per(t, p_per) ≈ @. off + g0 * mix

        # 3D Brownian, free offset
        κ = 10 * rand()
        τD3 = 1e-3
        spec3d = FCSFitting.FCSModel(; spec=FCSFitting.FCSModelSpec(; dim=:d3, anom=:none, n_diff=1))
        p3d = [g0, off, κ, τD3]
        @test spec3d(t, p3d) ≈ @. off + g0 * FCSFitting.udc_3d(t, τD3, κ)

        # 3D anomalous, global α
        αg = 0.9
        spec3d_ag = FCSFitting.FCSModel(; spec=FCSFitting.FCSModelSpec(; dim=:d3, anom=:global, n_diff=1))
        p3d_ag = [g0, off, κ, τD3, αg]
        @test spec3d_ag(t, p3d_ag) ≈ @. off + g0 * FCSFitting.udc_3d(t, τD3, κ, αg)

        # 3D anomalous, per-population α, 2 diffusers + dynamics
        τDs3 = [5e-4, 2e-3]; αs3 = [0.8, 1.0]; wts3 = [0.25]
        τdyn = [2e-5]; Kdyn = [0.15]; ics = [1]
        spec3d_per = FCSFitting.FCSModel(; spec=FCSFitting.FCSModelSpec(; dim=:d3, anom=:perpop, n_diff=2, ics=ics))
        p3d_per = vcat(g0, off, κ, τDs3, αs3, wts3, τdyn, Kdyn)
        dynfac3 = FCSFitting.dynamics_factor(t, τdyn, Kdyn, ics)
        mix3 = @. 0.25*FCSFitting.udc_3d(t, τDs3[1], κ, αs3[1]) + 0.75*FCSFitting.udc_3d(t, τDs3[2], κ, αs3[2])
        @test spec3d_per(t, p3d_per) ≈ @. off + g0 * mix3 * dynfac3
    end
end
