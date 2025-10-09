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
    sigstr(x::Real, s::Integer=5) -> String

Return a compact string with `s` significant digits, using scientific
notation for very small/large magnitudes (≈ like `"%.sg"`).
"""
function sigstr(x::Real, s::Integer=5)
    isnan(x)  && return "NaN"
    isinf(x)  && return x > 0 ? "Inf" : "-Inf"
    x == 0    && return "0"

    ax = abs(float(x))
    # Use scientific notation outside a "nice" range
    if ax < 1e-3 || ax ≥ 1e4
        e = floor(Int, log10(ax))
        mant = x / 10.0^e
        mant = round(mant, sigdigits=s)
        # Handle 9.999... → 10.0 rollover
        if abs(mant) ≥ 10
            mant /= 10
            e += 1
        end
        # Trim trailing zeros and lone dot
        ms = string(mant)
        occursin('.', ms) && (ms = rstrip(rstrip(ms, '0'), '.'))
        return ms * "e$(e ≥ 0 ? "+" : "")$e"
    else
        y = round(x, sigdigits=s)
        ys = string(y)
        occursin('.', ys) ? rstrip(rstrip(ys, '0'), '.') : ys
    end
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
    _fcs_plot(model, τ, G, θ0; kwargs...) -> fig, fit, scales

Requires `CairoMakie` (and `LaTeXStrings` for math labels).
"""
_fcs_plot(args...; kwargs...) =
    error("`_fcs_plot` requires CairoMakie (and LaTeXStrings). Load them: `using CairoMakie, LaTeXStrings`.")

"""
    resid_acf_plot(resid; kwargs...) -> fig

Requires `CairoMakie`. Activate by `using CairoMakie`.
"""
resid_acf_plot(args...; kwargs...) =
    error("`resid_acf_plot` requires CairoMakie. Load it: `using CairoMakie`.")

"""
    fcs_table(model, fit, scales; kwargs...) -> pretty output

Requires `PrettyTables`. Activate by `using PrettyTables`.
"""
fcs_table(args...; kwargs...) = 
    error("`fcs_table` requires PrettyTables. Load it: `using PrettyTables`.")

"""
    read_fcs(path; kwargs...) -> FCSData

Requires `DelimitedFiles`. Activate by `using DelimitedFiles` in the session.
"""
read_fcs(::Any; kwargs...) = 
    error("`read_fcs` requires DelimitedFiles. Load it first: `using DelimitedFiles`.")



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