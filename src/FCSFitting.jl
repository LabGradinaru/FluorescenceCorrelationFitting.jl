module FCSFitting

export FCSChannel, 
       FCSData,

       Dim, 
       Scope,
       FCSModelSpec,

       FCSFitResult,
       GlobalFCSFitResult,
       coef,
       dof,
       nobs,
       residuals,
       rss,
       weights,
       stderror,
       loglikelihood,
       aic,
       aicc,
       bic,
       shared_coef,
       channel_coef,
       channel_result,

       τD, 
       diffusivity, 
       Veff, 
       Aeff, 
       concentration, 
       surface_density, 
       hydrodynamic,

       fcs_fit,

       expected_parameter_names,
       infer_parameter_names,
       fcs_plot,
       resid_acf_plot,
       fcs_table,
       read_fcs
       
using LsqFit
using StatsAPI

include("data_structures.jl")
include("modelling.jl")
include("fitting.jl")
include("calculators.jl")
include("naming.jl")
include("extensions.jl")

end #module