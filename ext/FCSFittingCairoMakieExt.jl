module FCSFittingCairoMakieExt

using CairoMakie
using LsqFit
using LaTeXStrings
import FCSFitting: FCSModelSpec, FCSModel, FCSFitResult, _fcs_plot, resid_acf_plot, fcs_fit


const CORR_NAME = L"\mathrm{Correlation}"
const LAG_NAME = L"\mathrm{Logarithmic\ lag\ time}"


latexify_axis(x) = [L"%$(round(x[i], sigdigits=2))" for i in eachindex(x)]


"""
    _fcs_plot(spec, ch, θ0, color1, color2, color3) -> (fig, fit, scales)
    _fcs_plot(spec, ch, θ0, color1, color2) -> (fig, fit, scales)

Internal method for rendering previously fitted FCS data into a figure, with optional residuals.

# Keywords
- `fig=nothing`: `Makie.Figure` object to render the figure onto. This allows for multiple
                 fitting results/ curves to be plotted on a single set of axes.
- `kwargs...`: Passed to `Makie.Figure()`

# Notes
- In the three-color (residuals included) variant, two stacked axes are created if no axes exist in `fig`: top = correlations, bottom = residuals.
"""
function _fcs_plot end

function _fcs_plot(fit::FCSFitResult, τ::AbstractVector, data::AbstractVector, 
                   color1::Symbol, color2::Symbol, color3::Symbol; 
                   fig::Union{Nothing,Makie.Figure}=nothing, kwargs...)
    # Create or reuse a figure
    fig = isnothing(fig) ? Figure(size=(700, 600); kwargs...) : fig

    # Find existing axes in the provided figure (in creation order)
    axes_in_fig = [obj for obj in fig.content if obj isa Makie.Axis]
    if length(axes_in_fig) >= 2
        # Reuse the first two axes (assumed top then bottom)
        top_ax, bot_ax = axes_in_fig[1], axes_in_fig[2]
    else
        # Create missing axes (top: correlation; bottom: residuals)
        top_ax = Axis(fig[1, 1];
                      xticklabelsvisible = false, ylabel = CORR_NAME,
                      ytickformat = ys -> latexify_axis(ys),
                      xscale = log10, height = 400, width = 600)

        bot_ax = Axis(fig[2, 1];
                      xlabel = LAG_NAME, ylabel = L"\mathrm{Residuals}",
                      xscale = log10, height = 100, width = 600,
                      xtickformat = xs -> [L"%$(log10(xs[i]))" for i in eachindex(xs)],
                      ytickformat = ys -> latexify_axis(ys))
    end

    # Plot data and fit on the top axis
    scatter!(top_ax, τ, data; markersize=10, color=color1,
             strokewidth=1, strokecolor=:black, alpha=0.7)

    # Generate model that fit the data and plot it
    model = FCSModel(; fit.spec, fit.scales)
    lines!(top_ax, τ, model(τ, fit.param); linewidth=3, color=color2, alpha=0.9)

    # Plot residuals on the bottom axis
    scatterlines!(bot_ax, τ, fit.resid; color=color3, markersize=5, strokewidth=1, alpha=0.7)

    return fig, fit
end

function _fcs_plot(fit::FCSFitResult, τ::AbstractVector, data::AbstractVector, 
                   color1::Symbol, color2::Symbol; 
                   fig::Union{Nothing,Makie.Figure}=nothing, kwargs...)
    fig = isnothing(fig) ? Figure(size=(700, 600); kwargs...) : fig

    axes_in_fig = [obj for obj in fig.content if obj isa Makie.Axis]
    ax = length(axes_in_fig) >= 1 ? axes_in_fig[1] :
        Axis(fig[1,1]; xlabel = LAG_NAME, ylabel = CORR_NAME,
             ytickformat = ys -> latexify_axis(ys), xscale = log10)

    scatter!(ax, τ, data; markersize=10, color=color1, strokewidth=1, strokecolor=:black, alpha=0.7)

    model = FCSModel(; fit.spec, fit.scales)    
    lines!(ax, τ, model(τ, fit.param); linewidth=3, color=color2, alpha=0.9)

    return fig, fit
end


"""
    resid_acf_plot(fit; fontsize=20, fig=nothing, kwargs...)

Plot the autocorrelation of residuals of a fit as a qualitative test of goodness of fit.

# Arguments
- `fit::LsqFit.LsqFitResult` — Nonlinear least-squares fit result.

# Keywords
- `fontsize::Int=20` — Base font size for the figure.
- `fig::Union{Nothing,Makie.Figure}=nothing` — Reuse an existing figure if provided (first axis is reused).
- `color::Symbol=:orangered2` — Color of stems.
- `kwargs...` — Passed to `acf`.
"""
function resid_acf_plot(fit::FCSFitResult; fontsize::Int=20, 
                        fig::Union{Nothing,Makie.Figure}=nothing, 
                        color::Symbol=:orangered2, kwargs...)
    ρ = acf(fit.resid; kwargs...)

    fig = isnothing(fig) ? Figure(size=(700, 600); fontsize) : fig

    axes_in_fig = [obj for obj in fig.content if obj isa Makie.Axis]
    if length(axes_in_fig) >= 1
        ax = axes_in_fig[1]
    else
        ax = Axis(fig[1,1];
                  xlabel = L"\mathrm{Lag\ time}", ylabel = CORR_NAME,
                  xtickformat = xs -> latexify_axis(xs),
                  ytickformat = ys -> latexify_axis(ys))
    end

    stem!(ax, 0:(length(ρ)-1), ρ; color)
    return fig, ρ
end

"""
    acf(x; maxlag=clamp(length(x) - 1, 1, 1000), demean=true, unbiased=true)

Autocorrelation function up to `maxlag` (inclusive), returning ρ₀..ρ_maxlag.
"""
function acf(x::AbstractVector; maxlag::Int=clamp(length(x) - 1, 1, 1000),
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

end #module