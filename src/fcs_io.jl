const SI_PREFIXES = Dict(
    ""  => 1.0,
    "d" => 1e1,
    "c" => 1e2,
    "m" => 1e3,
    "μ" => 1e6,
    "u" => 1e6,
    "n" => 1e9,
    "p" => 1e12
)

"""
    FCSChannel(name, τ, G, σ)

A single correlation **channel** for FCS data.

# Fields
- `name::String` — Channel label (e.g., `"G_DD"`, `"G_DA"`, or `"G[1]"`).
- `τ::Vector{Float64}` — Lag times in seconds (same length as `G`).
- `G::Vector{Float64}` — Correlation values.
- `σ::Union{Nothing,Vector{Float64}}` — Optional per-lag standard deviations (same length as `G` if present).

# Notes
All vectors are assumed to be aligned element-wise (`τ[i]` ↔ `G[i]` ↔ `σ[i]`).
"""
struct FCSChannel
    name::String                 # e.g. "G_DD" or "G_DA"
    τ::Vector{Float64}           # lag times (s)
    G::Vector{Float64}           # correlation values
    σ::Union{Nothing,Vector{Float64}}  # std dev per lag (optional)
end
"""
    FCSData(channels, metadata, source)

A container for **multi-channel** FCS data plus provenance.

# Fields
- `channels::Vector{FCSChannel}` — One or more correlation channels sharing a `τ` grid.
- `metadata::Dict{String,Any}` — Arbitrary key-value info (sample, T, NA, λ, pinhole, detector, etc.).
- `source::String` — Provenance string (e.g., file path or `"in-memory"`).
"""
struct FCSData
    channels::Vector{FCSChannel}
    metadata::Dict{String,Any}   # sample, T, NA, λ, pinhole, detector, etc.
    source::String               # filepath or “in-memory”
end

function read_fcs(path::AbstractString; start_idx::Union{Nothing,Int}=nothing,
                  end_idx::Union{Nothing,Int}=nothing,
                  colspec=:auto, metadata=Dict{String,Any}())

    raw = readdlm(path)
    r1 = isnothing(start_idx) ? 1 : start_idx
    r2 = isnothing(end_idx)   ? size(raw,1) : end_idx
    M  = raw[r1:r2, :]

    # Basic inference: 1st col = τ, next 4 = Gs, next 4 = σs (if present)
    if colspec === :auto
        ncol = size(M,2)
        τ = vec(M[:,1])
        chans = FCSChannel[]
        # try pairs (G,σ) for columns 2..n
        i = 2
        k = 1
        while i <= ncol
            G = vec(M[:,i])
            σ = (i+4 <= ncol && ncol >= 9) ? vec(M[:, i+4]) : nothing
            push!(chans, FCSChannel("G[$k]", τ, G, σ))
            i += 1
            k += 1
            if k > 4 && ncol <= 9; break; end
        end
        return FCSData(chans, metadata, String(path))
    else
        # explicit mapping
        names = String[]
        chans = FCSChannel[]
        τ = vec(M[:, first(first(colspec)) == :τ ? last(first(colspec)) : error("τ col missing")])
        # build channels
        for tup in colspec
            sym, idx = tup
            if sym === :τ; continue; end
            if sym === :G
                name = get(tup, 3, "G")
                σidx  = get(tup, 4, nothing)
                σ = isnothing(σidx) ? nothing : vec(M[:, σidx])
                push!(chans, FCSChannel(name, τ, vec(M[:,idx]), σ))
            end
        end
        return FCSData(chans, metadata, String(path))
    end
end


"""
    infer_parameter_list(model_name, params; n_diff=nothing)

Infer the names of parameters used in the fitting based on the model name and
parameter vector length. The returned names follow the same ordering as the
model parameter vectors:
- base parameters
- all dynamic times (τ_dyn)
- all dynamic fractions (K_dyn)
"""
function infer_parameter_list(model_name::Symbol, params::AbstractVector; 
                              n_diff::Union{Nothing,Int}=nothing, 
                              diffusivity::Union{Nothing,Real}=nothing,
                              offset::Union{Nothing,Real}=nothing)
    L = length(params)
    column_names = String[]

    if model_name === :fcs_2d
        m = isnothing(offset) ? _ndyn_from_len(L - 3) : _ndyn_from_len(L - 2)
        append!(column_names, ["Diffusion time τ_D [s]"])
        !isnothing(diffusivity) && (append!(column_names, ["Beam width w_0 [m]"]))
        append!(column_names, ["Current amplitude G(0)"])
        isnothing(offset) && (append!(column_names, ["Offset G(∞)"]))
        append!(column_names, ["Dynamic time $(i) (τ_dyn) [s]" for i in 1:m])
        append!(column_names, ["Dynamic fraction $(i) (K_dyn)" for i in 1:m])
    elseif model_name === :fcs_2d_mdiff
        isnothing(n_diff) && throw(ArgumentError("n_diff required for fcs_2d_mdiff"))
        n = n_diff
        base = isnothing(offset) ? 2n + 2 : 2n + 1
        m = _ndyn_from_len(L - base)

        append!(column_names, ["Diffusion time $(i) τ_D[$i] [s]" for i in 1:n])
        append!(column_names, ["Population fraction $(i) w[$i]" for i in 1:n])
        append!(column_names, ["Current amplitude G(0)"])
        isnothing(offset) && (append!(column_names, ["Offset G(∞)"]))
        append!(column_names, ["Dynamic time $(i) (τ_dyn) [s]" for i in 1:m])
        append!(column_names, ["Dynamic fraction $(i) (K_dyn)" for i in 1:m])
    elseif model_name == :fcs_2d_anom
        m = isnothing(offset) ? _ndyn_from_len(L - 4) : _ndyn_from_len(L - 3)
        append!(column_names, ["Diffusion time τ_D [s]"])
        isnothing(diffusivity) && (append!(column_names, ["Beam width w_0 [m]"]))
        append!(column_names, ["Current amplitude G(0)"])
        isnothing(offset) && (append!(column_names, ["Offset G(∞)"]))
        append!(column_names, ["Anomolous exponent α"])
        append!(column_names, ["Dynamic time $(i) (τ_dyn) [s]" for i in 1:m])
        append!(column_names, ["Dynamic fraction $(i) (K_dyn)" for i in 1:m])
    elseif model_name === :fcs_3d
        m = isnothing(offset) ? _ndyn_from_len(L - 4) : _ndyn_from_len(L - 3)
        append!(column_names, ["Diffusion time τ_D [s]"])
        !isnothing(diffusivity) && (append!(column_names, ["Beam width w_0 [m]"]))
        append!(column_names, ["Current amplitude G(0)"])
        isnothing(offset) && (append!(column_names, ["Offset G(∞)"]))
        append!(column_names, ["Structure factor κ"])
        append!(column_names, ["Dynamic time $(i) (τ_dyn) [s]" for i in 1:m])
        append!(column_names, ["Dynamic fraction $(i) (K_dyn)" for i in 1:m])
    elseif model_name === :fcs_3d_mdiff
        isnothing(n_diff) && throw(ArgumentError("n_diff required for fcs_3d_mdiff"))
        n = n_diff
        base = isnothing(offset) ? 2n + 3 : 2n + 2
        m = _ndyn_from_len(L - base)

        append!(column_names, ["Diffusion time $(i) τ_D[$i] [s]" for i in 1:n])
        append!(column_names, ["Population fraction $(i) w[$i]" for i in 1:n])
        append!(column_names, ["Current amplitude G(0)"])
        isnothing(offset) && (append!(column_names, ["Offset G(∞)"]))
        append!(column_names, ["Structure factor κ"])
        append!(column_names, ["Dynamic time $(i) (τ_dyn) [s]" for i in 1:m])
        append!(column_names, ["Dynamic fraction $(i) (K_dyn)" for i in 1:m])
    else
        return String[]
    end

    return column_names
end

"""
    fcs_plot(model, ch, θ0; residuals=true, color1=:deepskyblue3, color2=:orangered2, color3=:steelblue4, kwargs...)

Fit and plot an FCS channel with optional residuals.

# Arguments
- `model::Function` — Model with signature `model(τ, θ; diffusivity=nothing, ...) -> Ĝ`.
- `ch::FCSChannel` — Data to fit.
- `θ0::AbstractVector` — Initial parameter guess.

# Keywords
- `residuals::Bool=true` — Plot residuals panel if `true`.
- `color1`, `color2`, `color3` — Colors for data, fit, residuals.
- `kwargs...` — Passed to `fcs_fit` (e.g., `σ=ch.σ`, `diffusivity`, bounds, etc.).

# Returns
- `(fig::Makie.Figure, fit::LsqFit.LsqFitResult, scales::AbstractVector)`

# Notes
Delegates to the internal `_fcs_plot` methods. Uses log-scaled τ axis.
"""
function fcs_plot(model::Function, ch::FCSChannel, θ0::AbstractVector; 
                  residuals::Bool=true, color1=:deepskyblue3, 
                  color2=:orangered2, color3=:steelblue4, kwargs...) 
    if residuals
        _fcs_plot(model, ch, θ0, color1, color2, color3; kwargs...)
    else
        _fcs_plot(model, ch, θ0, color1, color2; kwargs...)
    end
end

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
    fig = isnothing(fig) ? Figure(size=(700, 600), fontsize=fontsize) : fig

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
                     diffusivity = get(kwargs, :diffusivity, nothing),
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
                   color1::Symbol, color2::Symbol; fig::Union{Nothing,Makie.Figure}=nothing, 
                   fontsize::Int = 20, kwargs...)
    fit, scales = fcs_fit(model, ch.τ, ch.G, θ0; σ = ch.σ, kwargs...)

    fig = isnothing(fig) ? Figure(size=(700, 600), fontsize=fontsize) : fig

    axes_in_fig = [obj for obj in fig.content if obj isa Makie.Axis]

    if length(axes_in_fig) >= 1
        ax = axes_in_fig[1]
    else
        ax = Axis(fig[1,1];
                 xticklabelsvisible = false,
                 ylabel = L"\mathrm{Correlation} \; G(\tau)",
                 ytickformat = ys -> [L"%$(round(ys[i],sigdigits=2))" for i in eachindex(ys)],
                 xscale = log10)
    end

    scatter!(ax, ch.τ, ch.G; markersize=10, color=color1, strokewidth=1, strokecolor=:black, alpha=0.7)
    lines!(ax, ch.τ, model(ch.τ, fit.param .* scales; 
                           diffusivity = get(kwargs, :diffusivity, nothing),
                           n_diff = get(kwargs, :n_diff, nothing),
                           offset = get(kwargs, :offset, nothing)); 
           linewidth=3, color=color2, alpha=0.9)

    return fig, fit, scales
end

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

    # If diffusivity is provided, insert τ_D (and its error) at the top
    # Assumes no error in the diffusivity
    if !isnothing(diffusivity) 
        insert!(vals, 1, τD(diffusivity, vals[1]))
        insert!(errs, 1, errs[1] * vals[1] / (2diffusivity))
    end
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
    gof_line = " $(nameof(gof_metric)) " * @sprintf("= %.6g ", gof_val)

    column_labels = ["Parameters", "Values", "Std. Dev."]

    pretty_table(
        data;
        backend,
        column_labels = column_labels,
        source_notes = gof_line,
        source_note_alignment = :c,
        alignment = [:l, :r, :r],
        formatters = [(v,i,j)->(j ∈ (2,3) && v isa Number ? @sprintf("%.4g", v) : v)]
    )
end


"""
    parameters(fit, scale) -> Vector

Return **physical-space** parameter estimates as `fit.param .* scale`.

# Arguments
- `fit::LsqFit.LsqFitResult` — Nonlinear least-squares fit result.
- `scale::AbstractVector` — Multiplicative scaling vector (same length as `fit.param`).

# Returns
- `Vector{Float64}` of scaled parameters.
"""
parameters(fit::LsqFit.LsqFitResult, scale) = fit.param .* scale

"""
    errors(fit, scale) -> Vector

Return **standard deviations** of parameters in physical units: `stderror(fit) .* scale`.

# Arguments
- `fit::LsqFit.LsqFitResult` — Nonlinear least-squares fit result.
- `scale::AbstractVector` — Multiplicative scaling vector (same length as `fit.param`).

# Returns
- `Vector{Float64}` of scaled standard errors.

# Notes
Relies on `LsqFit.stderror`; assumes a well-posed covariance estimate.
"""
errors(fit::LsqFit.LsqFitResult, scale) = stderror(fit) .* scale

"""
    aic(fit) -> Real

Akaike Information Criterion for a least-squares fit.

# Definition
`AIC = 2k + N*log(σ²)`, where:
- `k = length(coef(fit))` is the number of fitted parameters,
- `N = nobs(fit)` is the number of observations,
- `σ² = rss(fit)/N` is the residual variance estimate.

# Returns
- Lower is better (relative comparison across models on the same dataset).
"""
function aic(fit::LsqFit.LsqFitResult)
    k, N = length(coef(fit)), nobs(fit)
    σ2 = rss(fit) / N
    return 2k + N*log(σ2)
end
"""
    aicc(fit) -> Real

Small-sample corrected AIC.

# Definition
`AICc = AIC + (2k(k+1)) / (N - k - 1)`.

# Returns
- Recommended when `N / k` is modest.
"""
function aicc(fit::LsqFit.LsqFitResult)
    k, N = length(coef(fit)), nobs(fit)
    a = aic(fit)
    return a + (2k*(k+1)) / (N - k - 1)
end
"""
    bic(fit) -> Real

Bayesian Information Criterion for a least-squares fit.

# Definition
`BIC = k*log(N) + N*log(σ²)`, with
- `k = length(coef(fit))`,
- `N = nobs(fit)`,
- `σ² = rss(fit)/N`.

# Returns
- Lower is better; BIC penalizes model complexity more strongly than AIC.
"""
function bic(fit::LsqFit.LsqFitResult)
    k, N = length(coef(fit)), nobs(fit)
    σ2 = rss(fit) / N
    return k*log(N) + N*log(σ2)
end
"""
    bicc(fit) -> Real

Bias-corrected BIC variant.

# Definition
`BICc = N*log(σ²) + N*k*log(N)/(N - k - 2)`.

# Returns
- A more conservative penalty when `N` is not ≫ `k`.
"""
function bicc(fit::LsqFit.LsqFitResult)
    k = length(coef(fit)); N = nobs(fit)
    σ2 = rss(fit) / N
    return N * log(σ2) + N * k * log(N) / (N - k - 2)
end