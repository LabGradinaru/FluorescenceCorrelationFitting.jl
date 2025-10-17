module FCSFittingPrettyTablesExt

using PrettyTables
using LsqFit

import FCSFitting: sigstr, fcs_table, infer_parameter_list, 
                   τD, parameters, errors, SI_PREFIXES, aic, 
                   aicc, bic, bicc, chi_squared, ljung_box, ww_test


"""
    fcs_table(model, fit, scales; backend=:html, n_diff=nothing, diffusivity=nothing, gof_metric=bic)

Render a **parameter table** from an `LsqFitResult`, including uncertainties and a goodness-of-fit metric.

# Arguments
- `model::Function` — Used to determine `model_name` for labeling.
- `fit::LsqFit.LsqFitResult` — Result from `fcs_fit`.
- `scales::AbstractVector` — Multiplicative scaling from fit space to physical space.

# Keywords
- `backend::Symbol=:html` — `PrettyTables` backend (`:html`, `:unicode`, `:latex`, etc.).
- `n_diff::Union{Nothing,Int}` — Required for `*_mdiff` models to label multi-species parameters.
- `diffusivity::Union{Nothing,Real}` — If provided, both `τ_D` and `w0` are displayed.
- `offset::Union{Nothing,Real} — If provided, the offset is removed from the display.`
- `gof_metric::Function=bic` — A function `gof_metric(fit)::Real` (e.g., `aic`, `aicc`, `bic`, `bicc`).
- `units::Union{Nothing, AbstractVector{String}}` — If provided, rescales parameter values to the 
                                                    corresponding SI prefix

# Output
Prints a table with columns:
- `"Parameters"` — Human-readable names from `infer_parameter_list(...)`,
- `"Values"` — `parameters(fit, scales)`,
- `"Std. Dev."` — `errors(fit, scales)`,

and a source note with the chosen GoF metric.

# Returns
- The return value of `pretty_table(...)` after printing the table.

# Notes
If `diffusivity` is provided, `τ_D` is computed and inserted at the top; the simple error propagation
assumes no uncertainty in `diffusivity`.
"""
function fcs_table(model::Function, fit::LsqFit.LsqFitResult, scales::AbstractVector; 
                   backend::Symbol=:html, n_diff::Union{Nothing,Int}=nothing, 
                   diffusivity::Union{Nothing, Real}=nothing, 
                   offset::Union{Nothing, Real}=nothing,
                   gof_metric::Function=bic,
                   units::Union{Nothing, AbstractVector{String}}=nothing)

    vals = parameters(fit, scales)
    errs = errors(fit, scales)

    mname = nameof(model)  # Symbol if model is a named function
    model_sym = mname isa Symbol ? mname : :unknown
    
    # Build parameter list (names) in the same order as values
    parameter_list = infer_parameter_list(model_sym, vals; n_diff, diffusivity, offset)

    # Trim to the common length
    n = min(length(parameter_list), length(vals), length(errs))
    parameter_list = parameter_list[1:n]
    vals = vals[1:n]
    errs = errs[1:n]

    # argument checks for SI prefix rescaling
    (units === nothing) && (units = fill("",n))
    length(units) < n && throw(ArgumentError("`units` length ($(length(units))) < number of displayed parameters ($n)."))
    # Validate keys and build multipliers
    bad = [u for u in units[1:n] if !haskey(SI_PREFIXES, u)]
    !isempty(bad) && throw(ArgumentError("Unknown SI prefixes in `units`."))
    # Apply scaling to values and errors
    multipliers = getindex.(Ref(SI_PREFIXES), units[1:n])
    @inbounds for i in 1:n
        vals[i] *= multipliers[i]
        errs[i] *= multipliers[i]
    end
    # Decorate displayed unit labels with the chosen prefix where present.
    # We only touch simple base units [s] and [m]; leave anything else as-is.
    @inbounds for i in 1:n
        u = units[i]
        if u != ""
            # Add prefix before the base symbol when found.
            # e.g. "[s]" -> "[μs]" and "[m]" -> "[nm]".
            parameter_list[i] = replace(parameter_list[i],
                "[s]" => "[" * u * "s]",
                "[m]" => "[" * u * "m]",
            )
        end
    end

    data = hcat(parameter_list, vals, errs)
    # evaluate goodness of fit metric and add it to the table
    gof_val = gof_metric(fit)
    gof_line = " $(nameof(gof_metric)) = $(sigstr(gof_val, 6)) "

    # PrettyTables call
    column_labels = ["Parameters", "Values", "Std. Dev."]

    pretty_table(
        data;
        backend,
        column_labels = column_labels,
        source_notes = gof_line,
        source_note_alignment = :c,
        alignment = [:l, :r, :r],
        formatters = [(v,i,j)->(j ∈ (2,3) && v isa Number ? sigstr(v, 4) : v)],
    )
end

end # module