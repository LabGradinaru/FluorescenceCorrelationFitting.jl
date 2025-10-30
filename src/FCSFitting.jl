module FCSFitting

export FCSChannel, FCSData, 
       expected_parameter_names, 
       infer_parameter_names,
       read_fcs,
       fcs_plot, fcs_table, 
       resid_acf_plot,
       fcs_fit, FCSFitResult,
       Dim, Scope, FCSModelSpec,
       τD, diffusivity, 
       volume, area, 
       concentration, 
       surface_density, 
       hydrodynamic

using LsqFit
using StatsAPI

include("FCSData.jl")
include("models.jl")
include("fcs_fit.jl")
include("io_utils.jl")

end #module