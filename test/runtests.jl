using FCSFitting, Test, Random, LsqFit, StatsAPI

Random.seed!(42)

include("models.jl")
include("io_utils.jl")
include("fcs_fit.jl")
include("FCSFittingCairoMakieExt.jl")