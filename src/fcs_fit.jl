"""
    FCSFitResult{P,R,J,W,T,S}(
        param,resid,jacobian,converged,trace,wt,spec,scales
    )
Container for the result of fitting with the model specified by `spec` based on `LsqFit.LsqFitResult`. 


FCSFitResult(lsf::LsqFit.LsqFitResult, spec, scales)

Wrap an `LsqFit.LsqFitResult` together with the model `spec` and `scales`
into an `FCSFitResult` that supports `StatsAPI` methods.
"""
struct FCSFitResult{P,R,J,W <: AbstractArray,T,S} <: StatsAPI.StatisticalModel
    param::P
    resid::R
    jacobian::J
    converged::Bool
    trace::T
    wt::W
    spec::FCSModelSpec
    scales::S
end

function FCSFitResult(lsf::LsqFit.LsqFitResult, spec::FCSModelSpec, scales)
    wt = (hasproperty(lsf, :wt) && getproperty(lsf, :wt) !== nothing) ? 
          getproperty(lsf, :wt) : nothing

    return FCSFitResult(
        lsf.param, lsf.resid, lsf.jacobian, lsf.converged, 
        lsf.trace, wt, spec, scales,
    )
end

"""
    _to_lfr(fit::FCSFitResult) -> LsqFit.LsqFitResult

Convert back to an `LsqFit.LsqFitResult` to reuse routines like `stderror`.
"""
function _to_lfr(fit::FCSFitResult)
    return LsqFit.LsqFitResult(
        fit.param, fit.resid, fit.jacobian, fit.converged, fit.trace, fit.wt
    )
end

StatsAPI.coef(ffr::FCSFitResult) = ffr.param .* ffr.scales
StatsAPI.nobs(ffr::FCSFitResult) = length(ffr.resid)
StatsAPI.rss(ffr::FCSFitResult) = sum(abs2, ffr.resid)
StatsAPI.weights(ffr::FCSFitResult) = ffr.wt
StatsAPI.residuals(ffr::FCSFitResult) = ffr.resid

"""
    StatsAPI.stderror(fit::FCSFitResult)

Standard errors of **physical-space** parameters.
"""
StatsAPI.stderror(fit::FCSFitResult; kwargs...) =
    LsqFit.stderror(_to_lfr(fit)) .* fit.scales

"""
    StatsAPI.dof(ffr::FCSFitResult)

Residual degrees of freedom = nobs - number of free parameters.
"""
StatsAPI.dof(ffr::FCSFitResult) = StatsAPI.nobs(ffr) - length(ffr.param)

"""
    mse(ffr::FCSFitResult)

Mean squared error based on residual DOF.
"""
mse(ffr::FCSFitResult) = StatsAPI.rss(ffr) / StatsAPI.dof(ffr)

"""
    isconverged(ffr::FCSFitResult)

Convenience boolean flag.
"""
isconverged(ffr::FCSFitResult) = ffr.converged

"""
    StatsAPI.offset(ffr::FCSFitResult)

Return the fitted (or fixed) offset in physical units.
If the spec fixes the offset, use that; otherwise assume the
offset is the second free parameter (g0 is first).
"""
function StatsAPI.offset(ffr::FCSFitResult)
    spec = ffr.spec
    # Prefer a fixed offset in the spec (if present)
    if hasproperty(spec, :offset) && getproperty(spec, :offset) !== nothing
        return getproperty(spec, :offset)
    end
    # Otherwise, interpret p[2] as the offset and scale it
    return ffr.param[2] * ffr.scales[2]
end

function StatsAPI.loglikelihood(fit::FCSFitResult)
    w = fit.wt
    N = nobs(fit)
    # Guard against pathological inputs
    if N == 0 || any(!isfinite, fit.resid) || any(!isfinite, w) || any(≤(0), w)
        return -Inf
    end

    # assume sample from an iid Gaussian
    σ2 = rss(fit) / N
    return -0.5 * (N * log(2π * σ2) + N)
end

r2(fit::FCSFitResult; variant::Symbol=:McFadden) = StatsAPI.r2(fit, variant)


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
    !hasoffset(spec) && (idx += 1) # offset
    (dim(spec) === d3) && (idx += 1) # κ

    # 4) diffusion times (or w0 when diffusivity is fixed)
    n = n_diff(spec)
    τ_end = idx + n - 1
    τ_end ≤ L || return idxs  # let caller fail later; here we just avoid OOB
    idx = τ_end + 1

    # 5) anomalous exponents
    if anom(spec) === globe
        idx += 1
    elseif anom(spec) === perpop
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
    fcs_fit(spec, times, data, p0) -> FCSFitResult
    fcs_fit(spec, channel, p0) -> FCSFitResult
    fcs_fit(model, times, data, p0) -> FCSFitResult
    fcs_fit(model, channel, p0) -> FCSFitResult

Fit FCS data, in the form of a pair of lag times and the correlation curve, 
based on a given `FCSModel` or its specifications, `FCSModelSpec` using 
`LsqFit.curve_fit`, with parameter normalization. 
`p0` is an initial model parameter guess (see Example).

# Example 
```julia
# 3D "Brownian" diffusion with one kinetic (exponential) term and an offset.
diffusivity = 5e-11         # m^2/s
offset = 0.0
spec = FCSModelSpec(dim = d3, anom = none, offset = offset, diffusivity = diffusivity)

# Synthetic example parameters and data: [g0, n_exp_terms, τD, τ_dyn, K_dyn]
initial_parameters = [1.0, 5.0, 2e-7, 1e-7, 0.1]
t = range(1e-7, 1e-2; length=256)
g = model(spec, initial_parameters, t) .+ 0.02 .* randn(length(t))

fit = fcs_fit(spec, t, g, initial_parameters)
```

# Keyword Arguments
- `σ=nothing`: Standard deviation of each data point in the correlation. 
               If no weight, `wt`, is provided, `wt = 1 ./ σ.^2` is used
- `wt=nothing`: Per-data point weighting. If `nothing`, each component of `data`
                is weighted equally during the fit.
- `scales=nothing`: Per-parameter scaling to convert to an order-1 parameter. 
                    If `nothing`, inferred from `p0` so that `θ0 = p0 / scales ≈ ones`,
                    while *not* scaling mixture weights
- `zero_sub=1.0`: Scale used in the scaling vector when `p0` contains a zero
- `lower=nothing`: Lower bound for each element of the parameter vector. If nothing, 
                   all values are unbounded from below
- `upper=nothing`: Upper bound for each element of the parameter vector. If nothing, 
                   all values are unbounded from above
- `kwargs`: Passed to `LsqFit.curve_fit`
"""
function fcs_fit end

function fcs_fit(spec::FCSModelSpec, τ::AbstractVector, data::AbstractVector, p0::AbstractVector;
                 σ::Union{Nothing,AbstractArray}=nothing, wt::Union{Nothing,AbstractArray}=nothing,
                 scales::Union{Nothing,AbstractVector}=nothing, zero_sub::Real=1.0,
                 lower=nothing, upper=nothing, kwargs...)

    N = length(τ)
    N == length(data) || throw(ArgumentError("Lag times and correlation values must be of equal length."))
    wt === nothing || (length(wt) == N || throw(ArgumentError("Weights must match data length.")))
    σ  === nothing || (length(σ)  == N || throw(ArgumentError("σ must match data length.")))

    # prefer explicit weights; else inverse-variance from σ; else ones
    weights = wt === nothing ? (σ === nothing ? ones(N) : @. 1 / σ^2) : wt

    # build scales (protect weights & K_dyn)
    noscale_idx = infer_noscale_indices(spec, p0)
    θ0, scales_ = if scales === nothing
        build_scales_from_p0(p0; noscale_idx, zero_sub)
    else
        length(scales) == length(p0) || throw(ArgumentError("Provided scales length mismatch."))
        (p0 ./ scales, scales)
    end

    model = FCSModel(; spec, scales=scales_)

    normalize_bounds(b) = b === nothing ? nothing :
        (length(b) == length(scales_) ? b ./ scales_ :
         throw(ArgumentError("lower/upper must have length $(length(scales_))")))

    lowerθ = normalize_bounds(lower)
    upperθ = normalize_bounds(upper)

    x = collect(τ)
    fit = if (lowerθ !== nothing) && (upperθ !== nothing)
        curve_fit(model, x, data, weights, θ0; lower=lowerθ, upper=upperθ, kwargs...)
    elseif (lowerθ !== nothing)
        curve_fit(model, x, data, weights, θ0; lower=lowerθ, kwargs...)
    elseif (upperθ !== nothing)
        curve_fit(model, x, data, weights, θ0; upper=upperθ, kwargs...)
    else
        curve_fit(model, x, data, weights, θ0; kwargs...)
    end

    return FCSFitResult(fit, spec, scales_)
end

fcs_fit(spec::FCSModelSpec, ch::FCSChannel, p0::AbstractVector; kwargs...) =
    fcs_fit(spec, ch.τ, ch.G, p0; σ=ch.σ, kwargs...)

function fcs_fit(m::FCSModel, τ::AbstractVector, data::AbstractVector, p0::AbstractVector; kwargs...)
    if m.scales === nothing
        return fcs_fit(m.spec, τ, data, p0; kwargs...)
    else
        return fcs_fit(m.spec, τ, data, p0; scales=m.scales, kwargs...)
    end
end

fcs_fit(m::FCSModel, ch::FCSChannel, p0::AbstractVector; kwargs...) =
    fcs_fit(m, ch.τ, ch.G, p0; σ=ch.σ, kwargs...)