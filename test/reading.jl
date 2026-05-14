@testset "reading" begin
    txtpath = "../examples/sample1.txt"
    pqrespath = "../examples/sample2.pqres"

    @testset "TXT read" begin
        data1 = read_fcs(txtpath)
        @test length(data1) === 4

        @testset "channel structure" begin
            for (k, ch) in enumerate(data1.channels)
                @test ch.name == "G[$k]"
                @test ch.τ isa Vector{Float64}
                @test ch.G isa Vector{Float64}
                @test ch.σ isa Vector{Float64}
                @test length(ch.τ) == length(ch.G) == length(ch.σ) == 300
            end
            @test data1.channels[1].τ[1] ≈ 1.0e-9
            @test data1.source == txtpath
        end

        @testset "row index filtering" begin
            sliced = read_fcs(txtpath; start_idx=10, end_idx=20)
            @test length(sliced.channels[1]) == 11
            @test sliced.channels[1].τ[1] ≈ data1.channels[1].τ[10]
        end

        @testset "time filtering" begin
            τ1 = data1.channels[1].τ[1]
            τ_end = data1.channels[1].τ[end]
            τ_mid = data1.channels[1].τ[150]
            filtered = read_fcs(txtpath; start_time=τ_mid)
            @test filtered.channels[1].τ[1] ≈ τ_mid
            @test filtered.channels[1].τ[end] ≈ τ_end
            filtered2 = read_fcs(txtpath; end_time=τ_mid)
            @test filtered2.channels[1].τ[1] ≈ τ1
            @test filtered2.channels[1].τ[end] ≈ τ_mid
        end

        @testset "colspec" begin
            cs = (τ=1, G=[2, 3], σ=[6, 7], names=["G_A", "G_B"])
            custom = read_fcs(txtpath; colspec=cs)
            @test length(custom) == 2
            @test custom.channels[1].name == "G_A"
            @test custom.channels[2].name == "G_B"
            @test custom.channels[1].σ isa Vector{Float64}
        end

        @testset "metadata passthrough" begin
            meta = Dict{String,Any}("sample" => "test123")
            data_m = read_fcs(txtpath; metadata=meta)
            @test data_m.metadata["sample"] == "test123"
        end
    end

    @testset "PQRES read" begin
        data2 = read_fcs(pqrespath)
        @test length(data2) === 1

        @testset "channel structure" begin
            ch = data2.channels[1]
            @test startswith(ch.name, "G[")
            @test ch.τ isa Vector{Float64}
            @test ch.G isa Vector{Float64}
            @test !isempty(ch.τ)
            @test length(ch.τ) == length(ch.G)
            @test ch.σ isa Vector{Float64}
            @test length(ch.σ) == length(ch.G)
        end

        @testset "metadata and source" begin
            @test !isempty(data2.metadata)
            @test data2.source == pqrespath
        end
    end

    @testset "unsupported extension" begin
        result = read_fcs("fake_file.csv")
        @test result isa ArgumentError
    end
end
