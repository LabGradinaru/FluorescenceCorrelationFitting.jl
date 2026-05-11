"""
    fcs_plot(fit, τ, g) -> fig, fit    
    fcs_plot(fit, ch) -> fig, fit
    fcs_plot(spec, τ, data, p0) -> fig, fit
    fcs_plot(spec, ch, p0) -> fig, fit
    
Plot autocorrelation data and a fitted curve. Optionally include a residuals panel.

## Keywords (main)
- `residuals=true`: show residuals panel
- `colors=(; data=:deepskyblue3, fit=:orangered2, resid=:steelblue4)`: NamedTuple of colors
- `fig=nothing`: pass an existing figure to draw onto (extension decides how it reuses axes)

# Example
```julia
using CairoMakie, LaTeXStrings, FCSFitting

# Synthetic example parameters and data: [g0, n_exp_terms, τD, τ_dyn, K_dyn]
initial_parameters = [1.0, 5.0, 2e-7, 1e-7, 0.1]
t = range(1e-7, 1e-2; length=256)
g = model(spec, initial_parameters, t) .+ 0.02 .* randn(length(t))

# Organize data into a channel for easier handling
channel = FCSChannel("sample", t, g, nothing)

fig, fit = fcs_plot(spec, channel, initial_parameters)
save("corr1.png", fig)
```

# Notes
- Delegates fitting to `fcs_fit` and plotting to the internal `_fcs_plot` method.
- Uses log10-scaled τ axis.
"""
function fcs_plot end

function fcs_plot(fit::FCSFitResult, τ::AbstractVector, data::AbstractVector;
                  residuals::Bool = true, colors = DEFAULT_FCS_PLOT_COLORS,
                  color1 = nothing, color2 = nothing, color3 = nothing,
                  fig = nothing, plot_kwargs::NamedTuple = NamedTuple())
    c = _fcs_colors(colors; color1, color2, color3)
    return _fcs_plot(fit, τ, data; residuals, colors = c, fig, plot_kwargs...)
end

fcs_plot(fit::FCSFitResult, ch::FCSChannel; kwargs...) =
    fcs_plot(fit, ch.τ, ch.G; kwargs...)

function fcs_plot(spec::FCSModelSpec, τ::AbstractVector,
                  data::AbstractVector, p0::AbstractVector;
                  fit_kwargs::NamedTuple = NamedTuple(),
                  plot_kwargs::NamedTuple = NamedTuple(),
                  residuals::Bool = true,
                  colors = DEFAULT_FCS_PLOT_COLORS,
                  color1 = nothing, color2 = nothing,
                  color3 = nothing, fig = nothing)
    fit = fcs_fit(spec, τ, data, p0; fit_kwargs...)
    return fcs_plot(fit, τ, data; residuals, colors, color1, color2, color3, fig, plot_kwargs)
end

function fcs_plot(spec::FCSModelSpec, ch::FCSChannel, p0::AbstractVector;
                  fit_kwargs::NamedTuple = NamedTuple(),
                  plot_kwargs::NamedTuple = NamedTuple(),
                  kwargs...)
    kw = merge(isnothing(ch.σ) ? NamedTuple() : (σ = ch.σ,), fit_kwargs)
    return fcs_plot(spec, ch.τ, ch.G, p0; fit_kwargs = kw, plot_kwargs, kwargs...)
end

_fcs_plot(args...; kwargs...) =
    error("`fcs_plot` requires CairoMakie (and LaTeXStrings). Load them: `using CairoMakie, LaTeXStrings`.")



const DEFAULT_FCS_PLOT_COLORS = (data = :deepskyblue3, fit = :orangered2, resid = :steelblue4)

_fcs_colors_nt(colors) =
    colors isa NamedTuple ? colors :
    (length(colors) == 2 ? (data = colors[1], fit = colors[2], resid = DEFAULT_FCS_PLOT_COLORS.resid) :
     length(colors) == 3 ? (data = colors[1], fit = colors[2], resid = colors[3]) :
     throw(ArgumentError("`colors` must be a NamedTuple or an AbstractArray of length 2 or 3")))

function _fcs_colors(colors; color1=nothing, color2=nothing, color3=nothing)
    c = _fcs_colors_nt(colors)
    color1 === nothing || (c = merge(c, (data = color1,)))
    color2 === nothing || (c = merge(c, (fit  = color2,)))
    color3 === nothing || (c = merge(c, (resid= color3,)))
    return c
end



"""
    resid_acf_plot(fit; kwargs...)

Plot the autocorrelation of residuals of a fit as a qualitative test of goodness of fit.

# Arguments
- `fit::FCSFitResult` — Nonlinear least-squares fit result.

# Keyword arguments
- `acf_kwargs=NamedTuple()`: forwarded to `acf`
- `plot_kwargs=NamedTuple()`: forwarded to `_resid_acf_plot`
- `fig=nothing`: optionally draw onto an existing figure
"""
function resid_acf_plot(
    fit::FCSFitResult;
    acf_kwargs::NamedTuple = NamedTuple(),
    plot_kwargs::NamedTuple = NamedTuple(),
    fig = nothing,
)
    ρ = acf(fit.resid; acf_kwargs...)
    return _resid_acf_plot(ρ, nobs(fit); fig=fig, plot_kwargs...)
end

"""
    acf(x; maxlag=clamp(length(x) - 1, 1, 1000), demean=true, unbiased=true)

Autocorrelation function up to `maxlag` (inclusive), returning ρ₀..ρ_maxlag.
"""
function acf(x::AbstractVector; maxlag::Int=clamp(length(x)-1,1,1000),
             demean::Bool=true, unbiased::Bool=true)
    N = length(x)
    @assert maxlag ≥ 1 && maxlag < N "maxlag must be in [1, N-1]"
    μ = demean ? (sum(x)/N) : zero(eltype(x))
    y = x .- μ
    γ0 = sum(abs2, y) / (unbiased ? (N - 1) : N)

    ρ = similar(y, maxlag + 1)
    ρ[1] = 1
    @inbounds for k in 1:maxlag
        num = sum(@view(y[1:N-k]) .* @view(y[1+k:N]))
        denom = unbiased ? (N - k) : N
        ρ[k+1] = (num / denom) / γ0
    end
    return ρ
end

_resid_acf_plot(args...; kwargs...) =
    error("`resid_acf_plot` requires CairoMakie. Load it: `using CairoMakie`.")

fcs_table(args...; kwargs...) = 
    error("`fcs_table` requires PrettyTables. Load it: `using PrettyTables`.")