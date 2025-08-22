module FCSFitting

export fcs_plot, fcs_fit, log_lags,
       fcs_2d, fcs_3d, fcs_2d_mdiff, fcs_3d_mdiff

using LsqFit
using LaTeXStrings
using CairoMakie


include("models.jl")
include("fcs_fit.jl")

end #module