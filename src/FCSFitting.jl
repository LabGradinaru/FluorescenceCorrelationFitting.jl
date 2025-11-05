module FCSFitting

export FCSChannel, 
       FCSData,

       Dim, 
       Scope,
       FCSModelSpec,
       FCSModel,

       τD, 
       diffusivity, 
       Veff, 
       Aeff, 
       concentration, 
       surface_density, 
       hydrodynamic,
       FCSFitResult,
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

       fcs_fit,

       expected_parameter_names,
       infer_parameter_names,
       fcs_plot,
       resid_acf_plot,
       fcs_table,
       read_fcs
       
using LsqFit
using StatsAPI

include("FCSData.jl")
include("models.jl")
include("fcs_fit.jl")
include("io_utils.jl")

end #module