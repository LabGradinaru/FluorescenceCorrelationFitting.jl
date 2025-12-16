const G0_NAME = "Current amplitude"
const OFF_NAME = "Offset"
const ANOM_NAME = "Anomalous exponent"
const STRUCT_NAME = "Structure factor"
const DIFFTIME_NAME = "Residence time"
const DIFFFRAC_NAME = "Diffusion population fraction"
const BEAM_NAME = "Beam width"
const DIFF_NAME = "Diffusivity"
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
- diffusion slots (either τD, w0 or D depending on `diffusivity` and `beamwidth`),
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
    !hasoffset(spec) && push!(names, OFF_NAME)
    dim(spec) === d3 && push!(names, STRUCT_NAME)

    # τD slots (or w0/D if D/w0 fixed)
    hd = hasdiffusivity(spec);  hw = haswidth(spec)
    if !hd || !hw
        n = n_diff(spec)
        base_label = DIFFTIME_NAME
        base_unit = "[s]"
        if hasdiffusivity(spec)
            base_label = BEAM_NAME
            base_unit = "[m]"
        end
        if haswidth(spec)
            base_label = DIFF_NAME
            base_unit = "[m²/s]"
        end
        @simd for i in 1:n
            push!(names, n == 1 ? base_label*" "*base_unit : "$(base_label) $i $(base_unit)")
        end
    end

    # anomalous exponents
    if anom(spec) === globe
        push!(names, ANOM_NAME)
    elseif anom(spec) === perpop
        @simd for i in 1:n
            push!(names, ANOM_NAME * " $i")
        end
    end

    # weights (n - 1)
    if n > 1
        @simd for i in 1:(n-1)
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
    isnan(x) && return "NaN"
    isinf(x) && return x > 0 ? "Inf" : "-Inf"
    x == 0 && return "0"

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


