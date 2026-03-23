module FluoCorrFittingCairoMakieExt

using CairoMakie
using LaTeXStrings

import FluoCorrFitting: FCSModel, FCSFitResult, _fcs_plot, _resid_acf_plot


const CORR_NAME = L"\mathrm{Correlation}"
const LAG_NAME = L"\mathrm{Logarithmic\ lag\ time}"


latexify_axis(x; n=1) = [L"%$(round(x[i], sigdigits=n))" for i in eachindex(x)]
latexify_log10(vals; sigdigits::Int=3) = [L"%$(round(log10(vals[i]), sigdigits=sigdigits))" for i in eachindex(vals)]


"""
    _fcs_plot(fit, τ, data; residuals=true, colors=(;data,fit,resid), fig=nothing,
              figure_kwargs=NamedTuple(), data_kw=NamedTuple(), fit_kw=NamedTuple(), resid_kw=NamedTuple()) -> fig, fit

Internal method for rendering previously fitted FCS data into a figure, with optional residuals.
"""
function _fcs_plot(
    fit::FCSFitResult,
    τ::AbstractVector,
    data::AbstractVector;
    residuals::Bool = true,
    colors::NamedTuple = (data = :deepskyblue3, fit = :orangered2, resid = :steelblue4),
    fig::Union{Nothing,Makie.Figure} = nothing,
    figure_kw::NamedTuple = (fontsize=20,),
    data_kw::NamedTuple = NamedTuple(),
    fit_kw::NamedTuple = NamedTuple(),
    resid_kw::NamedTuple = NamedTuple(),
)
    fig = isnothing(fig) ? Figure(; figure_kw...) : fig
    ax_top, ax_bot = _get_axes!(fig; residuals)

    # sensible defaults, user can override via *_kw
    data_defaults = (markersize=10, color=colors.data, strokewidth=1, strokecolor=:black, alpha=0.7)
    fit_defaults = (linewidth=3, color=colors.fit, alpha=0.9)
    resid_defaults = (markersize=5, color=colors.resid, strokewidth=1, alpha=0.7)

    scatter!(ax_top, τ, data; merge(data_defaults, data_kw)...)

    # Generate model curve from fit
    model = FCSModel(fit.spec, τ, fit.param; fit.scales)
    lines!(ax_top, τ, model(τ, fit.param); merge(fit_defaults, fit_kw)...)

    if residuals
        scatterlines!(ax_bot, τ, fit.resid; merge(resid_defaults, resid_kw)...)
    end

    return fig, fit
end

# ensure axes exist (or reuse the first ones found in fig)
function _get_axes!(fig::Makie.Figure; residuals::Bool)
    axes_in_fig = [obj for obj in fig.content if obj isa Makie.Axis]

    dims = fig.scene.viewport[].widths

    if residuals
        if length(axes_in_fig) >= 2
            return axes_in_fig[1], axes_in_fig[2]
        end
        top_ax = Axis(fig[1, 1];
            xticklabelsvisible = false,
            ylabel = CORR_NAME,
            xscale = log10,
            ytickformat = ys -> latexify_axis(ys),
            height = 4*dims[2]/7, width = 5*dims[1]/6,
        )
        bot_ax = Axis(fig[2, 1];
            xlabel = LAG_NAME,
            ylabel = L"\mathrm{Residuals}",
            xscale = log10,
            xtickformat = xs -> latexify_log10(xs),
            ytickformat = ys -> latexify_axis(ys),
            height = dims[2]/7, width = 5*dims[1]/6,
        )
        return top_ax, bot_ax
    else
        if !isempty(axes_in_fig)
            return axes_in_fig[1], nothing
        end
        ax = Axis(fig[1, 1];
            xlabel = LAG_NAME,
            ylabel = CORR_NAME,
            xscale = log10,
            ytickformat = ys -> latexify_axis(ys),
        )
        return ax, nothing
    end
end


"""
    _resid_acf_plot(data, N; fig=nothing, figure_kwargs=NamedTuple(), 
                    stem_kw=NamedTuple(), hline_kw=NamedTuple())

Internal method for rendering the autocorrelation of the residuals of a fit result.

# Keyword arguments
- `fig=nothing`: `Makie.Figure` object to render the figure onto. This allows for multiple
                 fitting results/ curves to be plotted on a single set of axes.
- `kwargs...`: Passed to `Makie.Figure()`
"""
function _resid_acf_plot(
    ρ::AbstractVector, N::Int;
    fig::Union{Nothing,Makie.Figure}=nothing,
    figure_kw::NamedTuple = (fontsize=20,),
    stem_kw::NamedTuple = NamedTuple(),
    hline_kw::NamedTuple = NamedTuple(),
)
    fig = isnothing(fig) ? Figure(; figure_kw...) : fig

    axes_in_fig = [obj for obj in fig.content if obj isa Makie.Axis]
    ax = !isempty(axes_in_fig) ? axes_in_fig[1] :
        Axis(fig[1,1];
            xlabel = L"\mathrm{Lag\ time}",
            ylabel = CORR_NAME,
            xtickformat = xs -> latexify_axis(xs),
            ytickformat = ys -> latexify_axis(ys),
        )

    stem!(ax, 0:(length(ρ)-1), ρ; stem_kw...)
    conf = 2 / sqrt(N)
    hlines!(ax, [conf, -conf]; merge(hline_kw, (xmax=length(ρ)-1,))...)
    return fig
end

end #module