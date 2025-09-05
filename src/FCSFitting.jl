module FCSFitting

export fcs_plot, fcs_table, 
       fcs_fit, log_lags,
       fcs_2d, fcs_3d, 
       fcs_2d_mdiff, fcs_3d_mdiff,
       fcs_2d_anom

using LsqFit
using LaTeXStrings
using CairoMakie
using PrettyTables
using Printf

include("models.jl")
include("fcs_fit.jl")
include("fcs_io.jl")

end #module