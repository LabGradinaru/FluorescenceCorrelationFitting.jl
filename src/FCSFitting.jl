module FCSFitting

export FCSChannel, FCSData, read_fcs,
       fcs_plot, fcs_table, resid_acf_plot,
       parameters, errors,
       fcs_fit, log_lags,
       fcs_2d, fcs_2d_mdiff, 
       fcs_2d_anom, fcs_2d_anom_mdiff,
       fcs_3d, fcs_3d_mdiff,
       τD, diffusivity, volume, area, 
       concentration, surface_density, hydrodynamic

using LsqFit

include("models.jl")
include("fcs_fit.jl")
include("gof.jl")
include("io_utils.jl")

end #module