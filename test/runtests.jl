using FluorescenceCorrelationFitting, Test, Random, LsqFit, StatsAPI

Random.seed!(42)

include("reading.jl")
include("data_structures.jl")
include("modelling.jl")
include("fitting.jl")
include("calculators.jl")
include("naming.jl")
include("extensions.jl")
include("FluorescenceCorrelationFittingCairoMakieExt.jl")