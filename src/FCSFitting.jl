module FCSFitting

export fcs_plot, fcs_fit, log_lags,
       fcs_2d, fcs_3d, fcs_2d_mdiff, fcs_3d_mdiff

using LsqFit
using LaTeXStrings
using CairoMakie

MT = Makie.MathTeXEngine
mt_fonts_dir = joinpath(dirname(pathof(MT)), "..", "assets", "fonts", "NewComputerModern")

set_theme!(fonts = (
    regular = joinpath(mt_fonts_dir, "NewCM10-Regular.otf"),
    bold = joinpath(mt_fonts_dir, "NewCM10-Bold.otf")
))

include("models.jl")
include("fcs_fit.jl")

end #module