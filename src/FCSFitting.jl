module FCSFitting

export fcs_fit, log_lags,
       fcs_2d, fcs_3d,
       fcs_2d_mdiff, fcs_3d_mdiff

using LsqFit
using Makie

include("models.jl")
include("fcs_fit.jl")
include("plot_utils.jl")

end #module