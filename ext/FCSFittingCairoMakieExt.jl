module FCSFittingCairoMakieExt

using CairoMakie
using LsqFit
using LaTeXStrings
import FCSFitting: FCSChannel, _fcs_plot, resid_acf_plot, fcs_fit, acf


"""
    _fcs_plot(model, ch, θ0, color1, color2, color3; fig=nothing, fontsize=20, kwargs...)

Internal: fit `ch` with `model` via `fcs_fit` and render **data + fit + residuals**.

# Arguments
- `model::Function`, `ch::FCSChannel`, `θ0::AbstractVector` — See `fcs_plot`.

# Keywords
- `color1`, `color2`, `color3` — Colors for scatter, fit, and residuals.
- `fig::Union{Nothing,Makie.Figure}=nothing` — Reuse an existing figure if provided (first two axes are reused).
- `fontsize::Int=20` — Base font size for the figure.
- `kwargs...` — Forwarded to `fcs_fit` (e.g., `σ`, `diffusivity`, bounds, weights).

# Returns
- `(fig::Makie.Figure, fit::LsqFit.LsqFitResult, scales::AbstractVector)`

# Notes
Creates two stacked axes if no axes exist in `fig`: top = `G(τ)`, bottom = residuals.
"""
function _fcs_plot(model::Function, ch::FCSChannel, θ0::AbstractVector, 
                   color1::Symbol, color2::Symbol, color3::Symbol; 
                   fig::Union{Nothing,Makie.Figure}=nothing, 
                   fontsize::Int = 20, kwargs...)
    fit, scales = fcs_fit(model, ch.τ, ch.G, θ0; σ = ch.σ, kwargs...)

    # Create or reuse a figure
    fig = isnothing(fig) ? Figure(size=(700, 600); fontsize) : fig

    # Find existing axes in the provided figure (in creation order)
    axes_in_fig = [obj for obj in fig.content if obj isa Makie.Axis]
    if length(axes_in_fig) >= 2
        # Reuse the first two axes (assumed top then bottom)
        top_ax, bot_ax = axes_in_fig[1], axes_in_fig[2]
    else
        # Create missing axes (top: correlation; bottom: residuals)
        top_ax = Axis(fig[1, 1];
                      xticklabelsvisible = false,
                      ylabel = L"\mathrm{Correlation}\;G(\tau)",
                      ytickformat = ys -> [L"%$(round(ys[i], sigdigits=2))" for i in eachindex(ys)],
                      xscale = log10, height = 400, width = 600)

        bot_ax = Axis(fig[2, 1];
                      xlabel = L"\mathrm{Logarithmic\ lag\ time}\; \log_{10}{\tau}",
                      ylabel = L"\mathrm{Residuals}",
                      xscale = log10, height = 100, width = 600,
                      xtickformat = xs -> [L"%$(log10(xs[i]))" for i in eachindex(xs)],
                      ytickformat = ys -> [L"%$(round(ys[i], sigdigits=2))" for i in eachindex(ys)])
    end

    # Plot data and fit on the top axis
    scatter!(top_ax, ch.τ, ch.G; markersize=10, color=color1,
             strokewidth=1, strokecolor=:black, alpha=0.7)

    n_diff = get(kwargs, :n_diff, nothing)
    isnothing(n_diff) ?       
        lines!(top_ax, ch.τ, 
               model(ch.τ, fit.param .* scales;
                     diffusivity = get(kwargs, :diffusivity, nothing),
                     offset = get(kwargs, :offset, nothing));
               linewidth=3, color=color2, alpha=0.9) :
        lines!(top_ax, ch.τ, 
               model(ch.τ, fit.param .* scales; n_diff,
                     offset = get(kwargs, :offset, nothing));
               linewidth=3, color=color2, alpha=0.9)

    # Plot residuals on the bottom axis
    scatterlines!(bot_ax, ch.τ, fit.resid; color=color3,
                  markersize=5, strokewidth=1, alpha=0.7)

    return fig, fit, scales
end

"""
    _fcs_plot(model, ch, θ0, color1, color2; fig=nothing, fontsize=20, kwargs...)

Internal: fit `ch` with `model` via `fcs_fit` and render **data + fit** (no residuals).

# Arguments
- `model::Function`, `ch::FCSChannel`, `θ0::AbstractVector` — See `fcs_plot`.

# Keywords
- `color1`, `color2` — Colors for scatter and fit.
- `fig::Union{Nothing,Makie.Figure}=nothing` — Reuse an existing figure if provided (first axis is reused).
- `fontsize::Int=20` — Base font size for the figure.
- `kwargs...` — Forwarded to `fcs_fit` (e.g., `σ`, `diffusivity`, bounds, weights).

# Returns
- `(fig::Makie.Figure, fit::LsqFit.LsqFitResult, scales::AbstractVector)`

# Notes
Creates a single log-τ axis if none exists in `fig`.
"""
function _fcs_plot(model::Function, ch::FCSChannel, θ0::AbstractVector, 
                   color1::Symbol, color2::Symbol; 
                   fig::Union{Nothing,Makie.Figure}=nothing, 
                   fontsize::Int = 20, kwargs...)
    fit, scales = fcs_fit(model, ch.τ, ch.G, θ0; σ = ch.σ, kwargs...)

    fig = isnothing(fig) ? Figure(size=(700, 600); fontsize) : fig

    axes_in_fig = [obj for obj in fig.content if obj isa Makie.Axis]
    if length(axes_in_fig) >= 1
        ax = axes_in_fig[1]
    else
        ax = Axis(fig[1,1];
                  xlabel = L"\mathrm{Logarithmic\ lag\ time}\; \log_{10}{\tau}",
                  ylabel = L"\mathrm{Correlation} \; G(\tau)",
                  ytickformat = ys -> [L"%$(round(ys[i],sigdigits=2))" for i in eachindex(ys)],
                  xscale = log10)
    end

    scatter!(ax, ch.τ, ch.G; markersize=10, color=color1, strokewidth=1, strokecolor=:black, alpha=0.7)

    n_diff = get(kwargs, :n_diff, nothing)
    isnothing(n_diff) ?       
        lines!(ax, ch.τ, 
               model(ch.τ, fit.param .* scales;
                     diffusivity = get(kwargs, :diffusivity, nothing),
                     offset = get(kwargs, :offset, nothing));
               linewidth=3, color=color2, alpha=0.9) :
        lines!(ax, ch.τ, 
               model(ch.τ, fit.param .* scales; n_diff,
                     offset = get(kwargs, :offset, nothing));
               linewidth=3, color=color2, alpha=0.9)

    return fig, fit, scales
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
function resid_acf_plot(fit::LsqFit.LsqFitResult; fontsize::Int=20, 
                        fig::Union{Nothing,Makie.Figure}=nothing, 
                        color::Symbol=:orangered2, kwargs...)
    ρ = acf(fit.resid; kwargs...)

    fig = isnothing(fig) ? Figure(size=(700, 600); fontsize) : fig

    axes_in_fig = [obj for obj in fig.content if obj isa Makie.Axis]
    if length(axes_in_fig) >= 1
        ax = axes_in_fig[1]
    else
        ax = Axis(fig[1,1];
                  xlabel = L"\mathrm{Lag\ time}\; k",
                  ylabel = L"\mathrm{Correlation} \; \hat{\rho}_k",
                  ytickformat = ys -> [L"%$(round(ys[i],sigdigits=2))" for i in eachindex(ys)])
    end

    stem!(ax, 0:(length(ρ)-1), ρ; color)
    return fig, ρ
end

end #module