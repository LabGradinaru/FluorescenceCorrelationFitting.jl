"""
    infer_parameter_list(model_name, params; n_diff=nothing)

Infer the names of parameters used in the fitting based on the model name and
parameter vector length. The returned names follow the same ordering as the
model parameter vectors:
- base parameters
- all dynamic times (τ_dyn)
- all dynamic fractions (K_dyn)
"""
function infer_parameter_list(model_name::Symbol, params::AbstractVector; n_diff::Union{Nothing,Int}=nothing)
    L = length(params)
    column_names = String[]

    if model_name === :fcs_2d
        # base = 3 → τD, g0, offset
        m = _ndyn_from_len(L - 3)
        append!(column_names, [
            "Diffusion time τ_D",
            "Current amplitude G(0)",
            "Offset G(∞)",
        ])
        append!(column_names, ["Dynamic time $(i) (τ_dyn)"      for i in 1:m])
        append!(column_names, ["Dynamic fraction $(i) (K_dyn)"  for i in 1:m])
    elseif model_name === :fcs_2d_mdiff
        isnothing(n_diff) && throw(ArgumentError("n_diff required for fcs_2d_mdiff"))
        n = n_diff
        base = 2n + 2  # τDs[1:n], wts[1:n], g0, offset
        m = _ndyn_from_len(L - base)

        append!(column_names, ["Diffusion time $(i) τ_D[$i]"      for i in 1:n])
        append!(column_names, ["Population fraction $(i) w[$i]"   for i in 1:n])
        append!(column_names, [
            "Current amplitude G(0)",
            "Offset G(∞)",
        ])
        append!(column_names, ["Dynamic time $(i) (τ_dyn)"      for i in 1:m])
        append!(column_names, ["Dynamic fraction $(i) (K_dyn)"  for i in 1:m])
    elseif model_name === :fcs_3d
        # base = 4 → τD, g0, offset, κ
        m = _ndyn_from_len(L - 4)
        append!(column_names, [
            "Diffusion time τ_D",
            "Current amplitude G(0)",
            "Offset G(∞)",
            "Structure factor κ",
        ])
        append!(column_names, ["Dynamic time $(i) (τ_dyn)"      for i in 1:m])
        append!(column_names, ["Dynamic fraction $(i) (K_dyn)"  for i in 1:m])
    elseif model_name === :fcs_3d_mdiff
        isnothing(n_diff) && throw(ArgumentError("n_diff required for fcs_3d_mdiff"))
        n = n_diff
        base = 2n + 3  # τDs[1:n], wts[1:n], g0, offset, κ
        m = _ndyn_from_len(L - base)

        append!(column_names, ["Diffusion time $(i) τ_D[$i]"      for i in 1:n])
        append!(column_names, ["Population fraction $(i) w[$i]"   for i in 1:n])
        append!(column_names, [
            "Current amplitude G(0)",
            "Offset G(∞)",
            "Structure factor κ",
        ])
        append!(column_names, ["Dynamic time $(i) (τ_dyn)"      for i in 1:m])
        append!(column_names, ["Dynamic fraction $(i) (K_dyn)"  for i in 1:m])
    else
        return String[]
    end

    # Sanity check: inferred names should match the parameter vector length.
    if length(column_names) != L
        throw(ArgumentError("Inferred $(length(column_names)) names for $model_name, but params length is $L. Check n_diff or dynamic count."))
    end

    return column_names
end

"""
    fcs_plot(model::Function, lag_times, data, θ0; fontsize=20, 
             color1=:deepskyblue3, color2=:orangered2, color3=:steelblue4, kwargs...)

Fit FCS data with `model` using `fcs_fit` and generate a plot of the fit and the residuals. 
"""
function fcs_plot(model::Function, lag_times::AbstractVector, data::AbstractVector, θ0::AbstractVector; 
                  fontsize::Int = 20, color1=:deepskyblue3, color2=:orangered2, color3=:steelblue4, kwargs...)
    fit, scales = fcs_fit(model, lag_times, data, θ0; kwargs...)

    fig = Figure(size=(700, 600), fontsize=fontsize)

    # Top panel
    Axis(fig[1,1];
         xticklabelsvisible = false,
         ylabel = L"\mathrm{Correlation} \; G(\tau)",
         ytickformat = ys -> [L"%$(round(ys[i],sigdigits=2))" for i in eachindex(ys)],
         xscale = log10, height = 400, width = 600)

    scatter!(lag_times, data; markersize=10, color=color1, strokewidth=1, strokecolor=:black, alpha=0.7)
    lines!(lag_times, model(lag_times, fit.param .* scales; diffusivity = get(kwargs, :diffusivity, nothing)); 
           linewidth=3, color=color2, alpha=0.9)

    # Bottom panel
    Axis(fig[2,1];
         xlabel = L"\mathrm{Logarithmic\ lag\ time}\; \log_{10}{\tau}",
         ylabel = L"\mathrm{Residuals}",
         xscale = log10, height = 100, width = 600,
         xtickformat = xs -> [L"%$(log10(xs[i]))" for i in eachindex(xs)],
         ytickformat = ys -> [L"%$(round(ys[i],sigdigits=2))" for i in eachindex(ys)])

    scatterlines!(lag_times, fit.resid; color=color3, markersize=5, strokewidth=1, alpha=0.7)

    return fig, fit, scales
end

"""
    fcs_table(model::Function, lag_times, data, θ0; backend::Symbol=:html, kwargs...)

Fit FCS data with `model` using `fcs_fit` and generate a table of the fitted parameters and the goodness of fit, BIC.    
"""
function fcs_table(model::Function, lag_times::AbstractVector, data::AbstractVector, θ0::AbstractVector; 
                   backend::Symbol=:html, kwargs...)
    fit, scales = fcs_fit(model, lag_times, data, θ0; kwargs...)
    fcs_table(model, fit, scales; backend)
end

"""
    fcs_table(fit::LsqFit.LsqFitResult, scales::AbstractVector; backend::Symbol=:html)

Generate a table of the fitted parameters corresponding to an FCS `LsqFitResult`, `fit`, and the goodness of fit, BIC.    
"""
function fcs_table(model::Function, fit::LsqFit.LsqFitResult, scales::AbstractVector; backend::Symbol=:html)
    vals = parameters(fit, scales)
    errs = errors(fit, scales)

    mname = nameof(model)  # Symbol if model is a named function
    model_sym = mname isa Symbol ? mname : :unknown
    parameter_list = infer_parameter_list(model_sym, vals)

    n = min(length(parameter_list), length(vals), length(errs))
    data = hcat(parameter_list[1:n], vals[1:n], errs[1:n])

    bic_val = FCSFitting.bic(fit)
    bic_line = @sprintf("BIC = %.5g", bic_val)

    column_labels = ["Parameters", "Values", "Std. Dev."]

    pretty_table(
        data;
        backend,
        column_labels = column_labels,
        source_notes = bic_line,
        source_note_alignment = :c,
        alignment = [:l, :r, :r],
        formatters = [(v,i,j)->(j ∈ (2,3) && v isa Number ? @sprintf("%.4g", v) : v)]
    )
end

# Some utility and goodness of fit functions
parameters(fit::LsqFit.LsqFitResult, scale) = fit.param .* scale
errors(fit::LsqFit.LsqFitResult, scale) = stderror(fit) .* scale
function aic(fit::LsqFit.LsqFitResult)
    k = length(coef(fit)); N = nobs(fit)
    σ2 = rss(fit) / N
    return 2k + N * log(σ2) + (2k^2 + 2k) / (N - k - 1)
end
function bic(fit::LsqFit.LsqFitResult)
    k = length(coef(fit)); N = nobs(fit)
    σ2 = rss(fit) / N
    return N * log(σ2) + N * k * log(N) / (N - k - 2)
end