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

const G0_NAME = "Current amplitude"
const OFF_NAME = "Offset"
const ANOM_NAME = "Anomalous exponent"
const STRUCT_NAME = "Structure factor"
const DIFFTIME_NAME = "Residence time"
const DIFFFRAC_NAME = "Diffusion population fraction"
const BEAM_NAME = "Beam width"
const DYNTIME_NAME = "Dynamic lifetime"
const DYNFRAC_NAME = "Dynamic population fraction"


"""
    expected_parameter_names(spec) -> Vector{String}

Return the human-readable parameter ordering required by `FCSModel(spec)`.
Names reflect how `_eval` interprets the parameter vector given the `spec`, including:
- current correlation
- optional offset,
- structure factor (3D),
- diffusion slots (either τD or w0 depending on `diffusivity`),
- anomalous exponents according to `anom`,
- mixture weights (n_diff-1),
- dynamics (τ_dyn..., K_dyn...) in that order.

This does **not** validate lengths; it just labels the sequence.
Refer to `infer_parameter_names` if the given parameter vector interpretation is of interest.
"""
function expected_parameter_names(spec::FCSModelSpec)
    names = _no_dynamics_params(spec)
    push!(names, DYNTIME_NAME * " [1:m] [s]")
    push!(names, DYNFRAC_NAME * " [1:m]")
    return names
end

"""
    infer_parameter_names(spec, params) -> Vector{String}

Infer the names of parameters used in the fitting based on the model name and
parameter vector length. Names reflect how `_eval` interprets the vector 
`params` given the `spec`:
- current correlation
- optional offset,
- structure factor (3D),
- diffusion slots (either τD or w0 depending on `diffusivity`),
- anomalous exponents according to `anom`,
- mixture weights (n_diff-1),
- dynamics (τ_dyn..., K_dyn...) in that order.
"""
function infer_parameter_names(spec::FCSModelSpec, params::AbstractVector)
    L = length(params)
    names = _no_dynamics_params(spec)
    m = _ndyn_from_len(L - length(names))

    append!(names, [DYNTIME_NAME * " $i [s]" for i in 1:m])
    append!(names, [DYNFRAC_NAME * " $i" for i in 1:m])

    return names
end

function _no_dynamics_params(spec::FCSModelSpec)
    names = String[]
    push!(names, G0_NAME)
    isnothing(spec.offset) && push!(names, OFF_NAME)
    spec.dim === d3 && push!(names, STRUCT_NAME)

    # τD slots (or w0 if D fixed)
    base_label = BEAM_NAME
    base_unit = "[m]"
    if isnothing(spec.diffusivity)
        base_label = DIFFTIME_NAME
        base_unit = "[s]"
    end
    for i in 1:spec.n_diff
        push!(names, spec.n_diff == 1 ? base_label*" "*base_unit : "$(base_label) $i $(base_unit)")
    end

    # anomalous exponents
    if spec.anom === globe
        push!(names, ANOM_NAME)
    elseif spec.anom === perpop
        for i in 1:spec.n_diff
            push!(names, ANOM_NAME * " $i")
        end
    end

    # weights (n_diff - 1)
    if spec.n_diff > 1
        for i in 1:(spec.n_diff-1)
            push!(names, DIFFFRAC_NAME * " $i")
        end
    end

    return names
end

"""
    sigstr(x, s=5) -> String

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
    fcs_plot(spec, τ, g, fit, scales) -> (fig, fit, scales)    
    fcs_plot(spec, ch, fit, scales) -> (fig, fit, scales)
    fcs_plot(spec, τ, data, p0) -> (fig, fit, scales)
    fcs_plot(spec, ch, p0) -> (fig, fit, scales)
    fcs_plot(model, τ, data, p0) -> (fig, fit, scales)
    fcs_plot(model, ch, p0) -> (fig, fit, scales)
    
Plot the autocorrelation data and its fit by the input FCSModel or FCSModelSpec. 
Optionally, the weighted residuals between the data and fit are included as a second panel 
along the bottom.

# Example
```julia
using CairoMakie, LaTeXStrings, FCSFitting

# Synthetic example parameters and data: [g0, n_exp_terms, τD, τ_dyn, K_dyn]
initial_parameters = [1.0, 5.0, 2e-7, 1e-7, 0.1]
t = range(1e-7, 1e-2; length=256)
g = model(spec, initial_parameters, t) .+ 0.02 .* randn(length(t))

# Organize data into a channel for easier handling
channel = FCSChannel("sample", t, g, nothing)

fig, fit, scales = fcs_plot(spec, channel, initial_parameters)
save("corr1.png", fig)
```

# Keyword arguments
- `residuals=true`: Include bottom residuals panel if `true`.
- `color1`, `color2`, `color3`: Plot colors for data, fit, and residuals, respectively.
- `kwargs...`: Passed to `fcs_fit`

# Notes
- Delegates to the internal `_fcs_plot` methods and subsequently to `fcs_fit`.
- Uses log-scaled τ axis.
"""
function fcs_plot end

function fcs_plot(spec::FCSModelSpec, τ::AbstractVector, data::AbstractVector, 
                  fit::LsqFit.LsqFitResult, scales::AbstractVector; 
                  residuals::Bool=true, color1=:deepskyblue3, 
                  color2=:orangered2, color3=:steelblue4, kwargs...)
    if residuals
        _fcs_plot(spec, τ, data, fit, scales, color1, color2, color3; kwargs...)
    else
        _fcs_plot(spec, τ, data, fit, scales, color1, color2; kwargs...)
    end
end

function fcs_plot(spec::FCSModelSpec, ch::FCSChannel, fit::LsqFit.LsqFitResult, 
                  scales::AbstractVector; kwargs...)
    return fcs_plot(spec, ch.τ, ch.G, fit, scales; kwargs...)
end

function fcs_plot(spec::FCSModelSpec, τ::AbstractVector, data::AbstractVector,
                  p0::AbstractVector; kwargs...)
    fit, scales = fcs_fit(spec, τ, data, p0; kwargs...)
    return fcs_plot(spec, τ, data, fit, scales; kwargs...)
end

fcs_plot(spec::FCSModelSpec, ch::FCSChannel, p0::AbstractVector; kwargs...) = 
    fcs_plot(spec, ch.τ, ch.G, p0; σ=ch.σ, kwargs...)

function fcs_plot(m::FCSModel, τ::AbstractVector, data::AbstractVector, 
                  p0::AbstractVector; kwargs...)
    fit, scales = fcs_fit(m, τ, data, p0; kwargs...)
    return fcs_plot(m.spec, τ, data, fit, scales; kwargs...)
end

function fcs_plot(m::FCSModel, ch::FCSChannel, p0::AbstractVector; kwargs...)
    fit, scales = fcs_fit(m, ch, p0; kwargs...)
    return fcs_plot(m.spec, ch, fit, scales; kwargs...)
end


"""
    _fcs_plot(spec, τ, G, θ0; kwargs...) -> fig, fit, scales

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