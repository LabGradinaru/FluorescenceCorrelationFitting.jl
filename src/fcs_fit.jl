"""
    log_lags(n_points, τmin, τmax)

Strictly increasing integer log-spaced lags in [τmin, τmax], zero-based.
Returns *fewer* than n_points if there aren't enough distinct integers.
"""
function log_lags(n_points::Int, τmin::Int, τmax::Int)
    @assert n_points ≥ 1
    @assert 0 ≤ τmin ≤ τmax
    # Work on [τmin+1, τmax+1] in log space, then subtract 1 to get zero-based
    r = range(log10(τmin + 1), log10(τmax + 1); length=n_points)
    lags = round.(Int, 10 .^ r .- 1)
    # Clamp and enforce strict monotonicity
    @inbounds for i in eachindex(lags)
        lags[i] = clamp(lags[i], τmin, τmax)
        if i > 1 && lags[i] ≤ lags[i-1]
            lags[i] = min(lags[i-1] + 1, τmax)
        end
    end
    # Drop duplicates if τ-range is too small
    return unique(lags)
end

"""
    build_scales_from_p0(p0; noscale_idx=Int[], zero_sub=1.0) -> (θ0, scales)

Construct a scale vector so that, ideally, `θ0 = ones` and `p = scales .* θ`
reproduces `p0`.

- Indices in `noscale_idx` are set to scale `1.0` (e.g., mixture weights, K_dyn).
- Zero entries in `p0` use `zero_sub` to avoid zero scale (so that the corresponding
  `θ0[i] = p0[i]/zero_sub`, often `0`).

Returns `(θ0, scales)`.
"""
function build_scales_from_p0(p0::AbstractVector{<:Real};
                              noscale_idx::AbstractVector{<:Integer}=Int[],
                              zero_sub::Real=1.0)
    L = length(p0)
    s = similar(p0, Float64)
    @inbounds for i in 1:L
        if any(==(i), noscale_idx)
            s[i] = 1.0
        else
            s[i] = (p0[i] == 0) ? float(zero_sub) : float(p0[i])
        end
    end
    θ0 = p0 ./ s
    return θ0, s
end

"""
    build_scales(params; zero_sub=1.0) -> (θ0, scales)

Compatibility wrapper for older code; does not protect any indices.
Prefer `build_scales_from_p0` when you know which indices should not be scaled.
"""
build_scales(params::AbstractVector{<:Real}; zero_sub::Real=1.0) =
    build_scales_from_p0(params; noscale_idx=Int[], zero_sub)

"""
    infer_noscale_indices(spec::FCSModelSpec, p0) -> Vector{Int}

Return 1-based indices in `p0` that should **not** be scaled:
- mixture weights (`n_diff-1` of them, if any),
- dynamic fractions `K_dyn` (the last `m` dynamic slots).

This mirrors the parameter layout used by `_eval(spec, t, p; scales)`.
"""
function infer_noscale_indices(spec::FCSModelSpec, p0::AbstractVector)
    L = length(p0)
    idxs = Int[]

    idx = 2 # g0
    isnothing(spec.offset) && (idx += 1) # offset
    (spec.dim.sym === :d3) && (idx += 1) # κ

    # 4) diffusion times (or w0 when diffusivity is fixed)
    n = spec.n_diff
    τ_end = idx + n - 1
    τ_end ≤ L || return idxs  # let caller fail later; here we just avoid OOB
    idx = τ_end + 1

    # 5) anomalous exponents
    if spec.anom.sym === :global
        idx += 1
    elseif spec.anom.sym === :perpop
        α_end = idx + n - 1
        α_end ≤ L || return idxs
        idx = α_end + 1
    end

    # 6) weights (n-1)
    if n > 1
        w_start = idx
        w_end = w_start + (n - 1) - 1
        w_end ≤ L || return idxs
        append!(idxs, w_start:w_end)
        idx = w_end + 1
    end

    # 7) dynamics: τ_dyn[1..m], K_dyn[1..m]
    total_extra = L - (idx - 1)
    m = _ndyn_from_len(total_extra)
    if m > 0
        # τ_dyn: idx .. idx+m-1
        # K_dyn: idx+m .. idx+2m-1  ← these should NOT be scaled
        k_start = idx + m
        k_end = idx + 2m - 1
        append!(idxs, k_start:k_end)
    end

    return idxs
end

"""
    fcs_fit(spec, times, data, p0) -> (fit, scales)
    fcs_fit(spec, channel, p0) -> (fit, scales)
    fcs_fit(model, times, data, p0) -> (fit, scales)
    fcs_fit(model, channel, p0) -> (fit, scales)

Fit FCS data, in the form of a pair of lag times and the correlation curve, 
based on a given `FCSModel` or its specifications, `FCSModelSpec`
using `LsqFit.curve_fit`, with parameter normalization. 
`p0` is an initial model parameter guess (see Example).

# Example 

```julia
# 3D "Brownian" diffusion with one kinetic (exponential) term and an offset.
diffusivity   = 5e-11         # m^2/s
offset        = 0.0
spec = FCSModelSpec(dim = :d3, anom = :none, offset = offset, diffusivity = diffusivity)

# Synthetic example parameters: [g0, n_exp_terms, τD, τ_dyn, K_dyn]
initial_parameters = [1.0, 5.0, 2e-7, 1e-7, 0.1]

# t: lag‑time vector (s); g: experimental correlation values
# Example stub (replace with real data):
t = range(1e-7, 1e-2; length=256)
g = model(spec, initial_parameters, t) .+ 0.02 .* randn(length(t))

fit, scale = fcs_fit(spec, t, g, initial_parameters)
```
# Keyword Arguments
- `σ=nothing`: Standard deviation of each data point in the correlation. 
               If no weight, `wt`, is provided, 

TODO!

# Notes

- If `scales` is `nothing`, they are inferred from `p0` so that `θ0 ≈ ones`, while
  *not* scaling mixture weights and K-dynamics (found via `infer_noscale_indices`).
- If `σ` is given and `wt` is not, weights are set to `1 ./ σ.^2`. Otherwise the
  provided `wt` is used; if both are `nothing`, the fit is unweighted.
- Bounds (`lower`, `upper`) are given in **physical** units and internally normalized
  to `θ`-space by dividing by `scales`.
"""
function fcs_fit end

function fcs_fit(spec::FCSModelSpec, times::AbstractVector, 
                 data::AbstractVector, p0::AbstractVector;
                 σ::Union{Nothing,AbstractVector}=nothing,
                 wt::Union{Nothing,AbstractVector}=nothing,
                 scales::Union{Nothing,AbstractVector}=nothing,
                 zero_sub::Real=1.0, lower=nothing, upper=nothing,
                 kwargs...)
    # basic consistency checks
    N = length(times)
    N == length(data) || throw(ArgumentError("Lag times and correlation values must be of equal length."))
    if wt !== nothing
        length(wt) == N || throw(ArgumentError("Weights must have same size as lag times and data."))
    end
    if σ !== nothing
        length(σ) == N || throw(ArgumentError("Standard deviations must have same size as lag times and data."))
    end

    # prefer explicit `wt`; otherwise derive from σ; otherwise unweighted.
    local weights = wt
    if weights === nothing 
        if σ !== nothing
            weights = @. 1 / σ^2
        else
            weights = ones(N)
        end
    end

    # Indices that should not be scaled (weights + K_dyn)
    noscale_idx = infer_noscale_indices(spec, p0)

    # Build scales if not provided; get normalized θ0
    if scales === nothing
        θ0, scales_ = build_scales_from_p0(p0; noscale_idx, zero_sub)
    else
        length(scales) == length(p0) ||
            throw(ArgumentError("Provided scales length mismatch."))
        θ0 = p0 ./ scales
        scales_ = scales
    end

    # Two-arg model for LsqFit that maps θ → p, then evaluates the generic model
    model = FCSModel(; spec, scales=scales_)

    # Normalize bounds to θ-space if provided
    normalize_bounds(b) = b === nothing ? nothing :
        (length(b) == length(scales_) ? b ./ scales_ :
         throw(ArgumentError("lower/upper must have length $(length(scales_))")))
    lowerθ = normalize_bounds(lower)
    upperθ = normalize_bounds(upper)

    # Fit
    x = collect(times)
    fit = if (lowerθ !== nothing) && (upperθ !== nothing)
        curve_fit(model, x, data, weights, θ0; lower=lowerθ, upper=upperθ, kwargs...)
    elseif (lowerθ !== nothing)
        curve_fit(model, x, data, weights, θ0; lower=lowerθ, kwargs...)
    elseif (upperθ !== nothing)
        curve_fit(model, x, data, weights, θ0; upper=upperθ, kwargs...)
    else
        curve_fit(model, x, data, weights, θ0; kwargs...)
    end

    return fit, scales_
end

function fcs_fit(m::FCSModel, times::AbstractVector,
                 data::AbstractVector, p0::AbstractVector; kwargs...)
    # If scales are pre-attached to the model, reuse them
    if m.scales === nothing
        fit, scales = fcs_fit(m.spec, times, data, p0; kwargs...)
        return fit, scales
    else
        # Reuse the scales in `m` and bypass auto-scaling
        return fcs_fit(m.spec, times, data, p0; scales=m.scales, kwargs...)
    end
end

fcs_fit(spec::FCSModelSpec, ch::FCSChannel, p0::AbstractVector; kwargs...) = 
    fcs_fit(spec, ch.τ, ch.G, p0; σ=ch.σ, kwargs...)

fcs_fit(m::FCSModel, ch::FCSChannel, p0::AbstractVector; kwargs...) = 
    fcs_fit(m, ch.τ, ch.G, p0; σ=ch.σ, kwargs...)