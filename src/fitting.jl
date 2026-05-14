const BOLTZMANN = 1.380649e-23 # Boltzmann constant in SI units
const AVAGADROS = 6.022141e23 # Avagadro's number in SI units

const SI_PREFIXES = Dict(
    "" => 1.0,
    "d" => 1e1,
    "c" => 1e2,
    "m" => 1e3,
    "μ" => 1e6,
    "u" => 1e6,
    "n" => 1e9,
    "A" => 1e10,
    "p" => 1e12
)


"""
    FCSFitResult{P,R,J,W,T,S}(
        param,resid,jacobian,converged,trace,wt,spec,scales
    )
Container for the result of fitting with the model specified by `spec` based on `LsqFit.LsqFitResult`. 


FCSFitResult(lsf::LsqFit.LsqFitResult, spec, scales)

Wrap an `LsqFit.LsqFitResult` together with the model `spec` and `scales`
into an `FCSFitResult` that supports `StatsAPI` methods.
"""
struct FCSFitResult{P,R,J,W<:AbstractArray,T,S} <: StatsAPI.StatisticalModel
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
StatsAPI.dof(ffr::FCSFitResult) = StatsAPI.nobs(ffr) - length(ffr.param)
StatsAPI.nobs(ffr::FCSFitResult) = length(ffr.resid)
StatsAPI.residuals(ffr::FCSFitResult) = ffr.resid
StatsAPI.rss(ffr::FCSFitResult) = sum(abs2, ffr.resid)
StatsAPI.weights(ffr::FCSFitResult) = ffr.wt
StatsAPI.stderror(fit::FCSFitResult; kwargs...) =
    LsqFit.stderror(_to_lfr(fit)) .* fit.scales
mse(ffr::FCSFitResult) = StatsAPI.rss(ffr) / StatsAPI.dof(ffr)
isconverged(ffr::FCSFitResult) = ffr.converged

function StatsAPI.loglikelihood(fit::FCSFitResult)
    r = fit.resid
    N = nobs(fit)
    N == 0 && return -Inf

    w = fit.wt === nothing ? ones(eltype(r), N) : fit.wt
    length(w) == N || throw(ArgumentError("fit.wt must match residual length"))
    (all(isfinite, r) && all(isfinite, w) && all(>(0), w)) || return -Inf

    # Weighted RSS with *unweighted* residuals (LsqFit stores unweighted r)
    rss_w = sum(@. w * r^2)
    σ2 = rss_w / N
    (σ2 > 0 && isfinite(σ2)) || return -Inf

    const_term = N*log(2π) - sum(log, w)    # = N*log(2π) when w ≡ 1
    return -0.5 * (const_term + N*log(σ2) + N)
end

bic(args...; kwargs...) = StatsAPI.bic(args...; kwargs...)
aic(args...; kwargs...) = StatsAPI.aic(args...; kwargs...)
aicc(args...; kwargs...) = StatsAPI.aicc(args...; kwargs...)


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
    τ_end ≤ L || return idxs  # quick return and let caller fail later; here we just avoid OOB
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

Fit FCS data, in the form of a pair of lag times and the correlation curve, 
based on specifications, `FCSModelSpec` using  `LsqFit.curve_fit`, 
with parameter normalization. 
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
- `wt_threshold=1e10`: Maximum value a data point can be weighed at. 
                       This is set to avoid data with very few or no points having weights which are unreasonably large.
- `lower=nothing`: Lower bound for each element of the parameter vector. If nothing, 
                   all values are unbounded from below
- `upper=nothing`: Upper bound for each element of the parameter vector. If nothing, 
                   all values are unbounded from above
- `kwargs`: Passed to `LsqFit.curve_fit`
"""
function fcs_fit end

function fcs_fit(spec::FCSModelSpec, τ::AbstractVector, data::AbstractVector, p0::AbstractVector;
                 σ::Union{Nothing,AbstractArray}=nothing, wt::Union{Nothing,AbstractArray}=nothing,
                 scales::Union{Nothing,AbstractVector}=nothing, zero_sub::Real=1.0, wt_threshold::Real=1e10,
                 lower=nothing, upper=nothing, kwargs...)

    N = length(τ)
    N == length(data) || throw(ArgumentError("Lag times and correlation values must be of equal length."))
    wt === nothing || (length(wt) == N || throw(ArgumentError("Weights must match data length.")))
    σ  === nothing || (length(σ) == N || throw(ArgumentError("σ must match data length.")))

    # prefer explicit weights; else inverse-variance from σ; else ones
    weights = wt === nothing ? (σ === nothing ? ones(N) : @. 1 / σ^2) : wt
    weights[weights .> wt_threshold] .= wt_threshold

    # build scales (protect weights & K_dyn)
    noscale_idx = infer_noscale_indices(spec, p0)
    θ0, scales_ = if scales === nothing
        build_scales_from_p0(p0; noscale_idx, zero_sub)
    else
        length(scales) == length(p0) || throw(ArgumentError("Provided scales length mismatch."))
        (p0 ./ scales, scales)
    end

    model = FCSModel(spec, τ, p0, scales=scales_)

    normalize_bounds(b) = b === nothing ? nothing :
        (length(b) == length(scales_) ? b ./ scales_ :
         throw(ArgumentError("lower/upper must have length $(length(scales_))")))

    lowerθ = normalize_bounds(lower);  upperθ = normalize_bounds(upper)

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


# ─────────────────────────────────────────────────────────────────────────────
# Global (multi-channel) simultaneous fitting
# ─────────────────────────────────────────────────────────────────────────────

"""
    GlobalFCSFitResult

Result of simultaneously fitting multiple FCS channels with one set of
parameters constrained to be identical ("global") across all channels.

The combined normalized parameter vector is laid out as:
`[shared_params, local_params_ch1, local_params_ch2, ...]`
where the shared params occupy `param[1:n_shared]`.

Use [`shared_coef`](@ref), [`channel_coef`](@ref), and [`channel_result`](@ref)
to inspect results per channel.  For correct standard errors of the shared
parameters use `stderror(r)` directly on the `GlobalFCSFitResult`.
"""
struct GlobalFCSFitResult
    param::Vector{Float64}            # combined θ (normalized), length = n_shared + Σlocal
    resid::Vector{Float64}            # concatenated residuals across all channels
    jacobian::Matrix{Float64}         # combined ∂resid/∂θ
    converged::Bool
    trace::Any
    wt::Vector{Float64}               # concatenated weights (never nothing)
    specs::Vector{Any}                # FCSModelSpec instances, one per channel
    combined_scales::Vector{Float64}  # physical_param = θ .* combined_scales
    n_shared::Int                     # number of shared (global) parameters
    local_lengths::Vector{Int}        # length of each channel's local-param block
    n_obs::Vector{Int}                # number of observations per channel
    i_shared_in_full::Vector{Vector{Int}}  # for ch k: positions of shared params in full param vec
    i_local_in_full::Vector{Vector{Int}}   # for ch k: positions of local  params in full param vec
    channel_scales::Vector{Vector{Float64}} # full per-channel scales (length = full param vector)
end

function GlobalFCSFitResult(lsf::LsqFit.LsqFitResult, specs, combined_scales,
                            n_shared, local_lengths, n_obs,
                            i_shared_in_full, i_local_in_full, channel_scales,
                            fallback_wt)
    wt_stored = (hasproperty(lsf, :wt) && lsf.wt !== nothing) ?
                 Vector{Float64}(lsf.wt) : Vector{Float64}(fallback_wt)
    return GlobalFCSFitResult(
        Vector{Float64}(lsf.param), Vector{Float64}(lsf.resid),
        Matrix{Float64}(lsf.jacobian), lsf.converged, lsf.trace, wt_stored,
        collect(Any, specs), Vector{Float64}(combined_scales),
        n_shared, Vector{Int}(local_lengths), Vector{Int}(n_obs),
        i_shared_in_full, i_local_in_full,
        [Vector{Float64}(s) for s in channel_scales],
    )
end

function _to_lfr(fit::GlobalFCSFitResult)
    return LsqFit.LsqFitResult(
        fit.param, fit.resid, fit.jacobian, fit.converged, fit.trace, fit.wt
    )
end

StatsAPI.coef(r::GlobalFCSFitResult) = r.param .* r.combined_scales
StatsAPI.nobs(r::GlobalFCSFitResult) = sum(r.n_obs)
StatsAPI.dof(r::GlobalFCSFitResult) = StatsAPI.nobs(r) - length(r.param)
StatsAPI.residuals(r::GlobalFCSFitResult) = r.resid
StatsAPI.rss(r::GlobalFCSFitResult) = sum(abs2, r.resid)
StatsAPI.weights(r::GlobalFCSFitResult) = r.wt
StatsAPI.stderror(r::GlobalFCSFitResult; kwargs...) =
    LsqFit.stderror(_to_lfr(r)) .* r.combined_scales
isconverged(r::GlobalFCSFitResult) = r.converged
mse(r::GlobalFCSFitResult) = StatsAPI.rss(r) / StatsAPI.dof(r)

function StatsAPI.loglikelihood(fit::GlobalFCSFitResult)
    r = fit.resid
    N = StatsAPI.nobs(fit)
    N == 0 && return -Inf
    w = fit.wt
    (all(isfinite, r) && all(isfinite, w) && all(>(0), w)) || return -Inf
    rss_w = sum(@. w * r^2)
    σ2 = rss_w / N
    (σ2 > 0 && isfinite(σ2)) || return -Inf
    const_term = N*log(2π) - sum(log, w)
    return -0.5 * (const_term + N*log(σ2) + N)
end

"""
    shared_coef(r::GlobalFCSFitResult) -> Vector{Float64}

Return the physical (unscaled) values of the globally shared parameters.
"""
shared_coef(r::GlobalFCSFitResult) =
    r.param[1:r.n_shared] .* r.combined_scales[1:r.n_shared]

"""
    channel_coef(r::GlobalFCSFitResult, k::Int) -> Vector{Float64}

Return the full physical parameter vector for channel `k`, combining the
global shared values with that channel's local fitted values.
"""
function channel_coef(r::GlobalFCSFitResult, k::Int)
    1 ≤ k ≤ length(r.specs) || throw(ArgumentError("k=$k out of range 1..$(length(r.specs))."))
    n_full  = r.n_shared + r.local_lengths[k]
    full_θ  = Vector{Float64}(undef, n_full)
    loff    = r.n_shared + (k > 1 ? sum(r.local_lengths[1:k-1]) : 0)
    @inbounds for (j, i) in enumerate(r.i_shared_in_full[k])
        full_θ[i] = r.param[j]
    end
    @inbounds for (j, i) in enumerate(r.i_local_in_full[k])
        full_θ[i] = r.param[loff + j]
    end
    return full_θ .* r.channel_scales[k]
end

"""
    channel_result(r::GlobalFCSFitResult, k::Int) -> FCSFitResult

Reconstruct a per-channel `FCSFitResult` for channel `k` from the global fit.

!!! note
    Standard errors of **shared** parameters in the returned result use only
    channel `k`'s observations and are therefore conservative (larger than the
    joint estimate).  Use `stderror(r)[1:r.n_shared]` on the `GlobalFCSFitResult`
    for correct shared-parameter uncertainties.
"""
function channel_result(r::GlobalFCSFitResult, k::Int)
    1 ≤ k ≤ length(r.specs) || throw(ArgumentError("k=$k out of range 1..$(length(r.specs))."))
    n_full = r.n_shared + r.local_lengths[k]
    full_θ = Vector{Float64}(undef, n_full)
    loff = r.n_shared + (k > 1 ? sum(r.local_lengths[1:k-1]) : 0)

    @inbounds for (j, i) in enumerate(r.i_shared_in_full[k])
        full_θ[i] = r.param[j]
    end
    @inbounds for (j, i) in enumerate(r.i_local_in_full[k])
        full_θ[i] = r.param[loff + j]
    end

    obs_start = (k > 1 ? sum(r.n_obs[1:k-1]) : 0) + 1
    obs_end = sum(r.n_obs[1:k])
    ch_resid = r.resid[obs_start:obs_end]
    ch_wt = r.wt[obs_start:obs_end]

    # Per-channel jacobian in the channel's full-θ space.
    # Columns from the global jacobian are rearranged to match the channel's
    # full parameter ordering.  Other channels' columns are zero and dropped.
    J_global = r.jacobian
    ch_jac = zeros(r.n_obs[k], n_full)
    @inbounds for (j, i) in enumerate(r.i_shared_in_full[k])
        ch_jac[:, i] .= J_global[obs_start:obs_end, j]
    end
    @inbounds for (j, i) in enumerate(r.i_local_in_full[k])
        ch_jac[:, i] .= J_global[obs_start:obs_end, loff + j]
    end

    ch_lsf = LsqFit.LsqFitResult(full_θ, ch_resid, ch_jac, r.converged, r.trace, ch_wt)
    return FCSFitResult(ch_lsf, r.specs[k], r.channel_scales[k])
end


"""
    fcs_fit(specs, τs, datas, p0s; shared=:τD, ...) -> GlobalFCSFitResult
    fcs_fit(specs, channels, p0s; shared=:τD, ...) -> GlobalFCSFitResult

Simultaneously fit multiple FCS correlation curves while constraining one set
of parameters to be identical ("global") across all channels.

The two primary use cases are:
- **Same molecule, different fluorophore spectra**: channels share the same
  diffusion time(s).  Set each spec with `diffusivity` and `width` both
  unset (default) so that `τD` is the free parameter, and use `shared=:τD`.
- **Different molecules, same optical setup**: channels share beam width
  `w₀` but have independent diffusivities.  Fix `diffusivity` in each spec
  (set `diffusivity=D_k`) so `w₀` is the free parameter, and use `shared=:w0`.

# Arguments
- `specs`: `Vector` of `FCSModelSpec`, one per channel.
- `τs` / `channels`: Lag-time vectors or `FCSChannel` objects per channel.
- `datas`: Correlation data vectors (not needed with the `FCSChannel` dispatch).
- `p0s`: Initial parameter guess vectors, one per channel.

# Keyword Arguments
- `shared=:τD`: Which physical quantity is constrained globally:
  - `:τD` — share diffusion time(s); specs must not fix both `diffusivity` and `width`.
  - `:w0` / `:width` — share beam waist(s); all specs must fix `diffusivity`.
  - `:D`  / `:diffusivity` — share diffusivity; all specs must fix `width`.
- `σs=nothing`: Per-channel uncertainty vectors (a `Vector` of vectors or `nothing`).
- `wts=nothing`: Per-channel weight vectors; overrides `σs` when provided.
- `zero_sub=1.0`, `lower=nothing`, `upper=nothing`: As in single-channel `fcs_fit`.
- `kwargs`: Forwarded to `LsqFit.curve_fit`.

# Returns
A [`GlobalFCSFitResult`](@ref).  Inspect it with [`shared_coef`](@ref),
[`channel_coef`](@ref), and [`channel_result`](@ref).
"""
function fcs_fit(specs::AbstractVector{<:FCSModelSpec},
                 τs::AbstractVector{<:AbstractVector},
                 datas::AbstractVector{<:AbstractVector},
                 p0s::AbstractVector{<:AbstractVector};
                 shared::Symbol = :τD,
                 σs = nothing,
                 wts = nothing,
                 zero_sub::Real = 1.0,
                 lower = nothing,
                 upper = nothing,
                 kwargs...)

    K = length(specs)
    K == length(τs) == length(datas) == length(p0s) ||
        throw(ArgumentError("specs, τs, datas, and p0s must all have the same length."))
    K ≥ 2 || throw(ArgumentError("Global fitting requires at least 2 channels."))

    shared ∈ (:τD, :w0, :width, :D, :diffusivity) ||
        throw(ArgumentError("shared must be one of :τD, :w0, :width, :D, :diffusivity; got :$shared."))

    # Validate per-channel spec consistency with the chosen shared quantity,
    # compute per-channel scales and layouts
    full_scales = Vector{Vector{Float64}}(undef, K)
    full_θ0s = Vector{Vector{Float64}}(undef, K)
    layouts = Vector{ParamLayout}(undef, K)

    for k in 1:K
        length(τs[k]) == length(datas[k]) ||
            throw(ArgumentError("Channel $k: lag times and data have different lengths."))

        if shared ∈ (:w0, :width)
            hasdiffusivity(specs[k]) ||
                throw(ArgumentError("shared=:$shared requires all specs to have a fixed diffusivity. Channel $k does not."))
        elseif shared ∈ (:D, :diffusivity)
            haswidth(specs[k]) ||
                throw(ArgumentError("shared=:$shared requires all specs to have a fixed beam width. Channel $k does not."))
        end

        noscale_idx = infer_noscale_indices(specs[k], p0s[k])
        θ0k, sk = build_scales_from_p0(p0s[k]; noscale_idx, zero_sub)
        full_θ0s[k] = θ0k
        full_scales[k] = sk
        layouts[k] = ParamLayout(specs[k], sk, p0s[k])

        isempty(layouts[k].i_τD) &&
            throw(ArgumentError("Channel $k has no free τD-slot parameters to share " *
                "(both diffusivity and beam width are fixed in its spec)."))
    end

    n_shared = length(layouts[1].i_τD)
    for k in 2:K
        length(layouts[k].i_τD) == n_shared ||
            throw(ArgumentError("All channels must have the same number of shared parameters. " *
                "Channel 1 has $n_shared, channel $k has $(length(layouts[k].i_τD))."))
    end

    # Align τD-slot scales: override channels 2..K to match channel 1's τD scales
    # so the shared θ values are interpreted identically by every channel model.
    shared_scales_ref = full_scales[1][layouts[1].i_τD]
    for k in 2:K
        full_scales[k][layouts[k].i_τD] .= shared_scales_ref
        full_θ0s[k] = Vector{Float64}(p0s[k]) ./ full_scales[k]
    end

    # Index maps: which positions in each channel's full param vec are shared vs local
    i_shared_in_full = [collect(Int, layouts[k].i_τD) for k in 1:K]
    i_local_in_full = [[i for i in 1:length(p0s[k]) if i ∉ layouts[k].i_τD] for k in 1:K]
    local_lengths = length.(i_local_in_full)

    # Build combined initial θ and scales
    shared_θ0 = full_θ0s[1][i_shared_in_full[1]]
    local_θ0s = [full_θ0s[k][i_local_in_full[k]] for k in 1:K]
    combined_θ0 = vcat(shared_θ0, local_θ0s...)

    local_scales_k = [full_scales[k][i_local_in_full[k]] for k in 1:K]
    combined_scales_ = vcat(shared_scales_ref, local_scales_k...)

    # Per-channel FCSModels (each stores its own full scales)
    τs_f64 = [Vector{Float64}(τ) for τ in τs]
    channel_models = [FCSModel(specs[k], τs_f64[k], p0s[k]; scales=full_scales[k]) for k in 1:K]

    # Callable for the joint optimization
    n_obs = length.(datas)
    global_model = _build_global_model(
        channel_models, τs_f64, n_shared, local_lengths,
        i_shared_in_full, i_local_in_full,
    )

    # Concatenated data and weights
    combined_data = vcat(datas...)
    combined_wt = if wts !== nothing
        vcat(wts...)
    elseif σs !== nothing
        vcat([σs[k] === nothing ? ones(Float64, n_obs[k]) : @. 1 / σs[k]^2 for k in 1:K]...)
    else
        ones(Float64, sum(n_obs))
    end

    x_dummy = collect(1:sum(n_obs))  # GlobalFCSModel ignores x; LsqFit needs it

    normalize_bounds(b) = b === nothing ? nothing :
        (length(b) == length(combined_scales_) ? b ./ combined_scales_ :
         throw(ArgumentError("lower/upper must have length $(length(combined_scales_))")))
    lowerθ = normalize_bounds(lower)
    upperθ = normalize_bounds(upper)

    fit = if lowerθ !== nothing && upperθ !== nothing
        curve_fit(global_model, x_dummy, combined_data, combined_wt, combined_θ0;
                  lower=lowerθ, upper=upperθ, kwargs...)
    elseif lowerθ !== nothing
        curve_fit(global_model, x_dummy, combined_data, combined_wt, combined_θ0;
                  lower=lowerθ, kwargs...)
    elseif upperθ !== nothing
        curve_fit(global_model, x_dummy, combined_data, combined_wt, combined_θ0;
                  upper=upperθ, kwargs...)
    else
        curve_fit(global_model, x_dummy, combined_data, combined_wt, combined_θ0; kwargs...)
    end

    return GlobalFCSFitResult(fit, specs, combined_scales_, n_shared, local_lengths, n_obs,
                              i_shared_in_full, i_local_in_full,
                              [full_scales[k] for k in 1:K], combined_wt)
end

function fcs_fit(specs::AbstractVector{<:FCSModelSpec},
                 channels::AbstractVector{<:FCSChannel},
                 p0s::AbstractVector{<:AbstractVector};
                 kwargs...)
    τs = [ch.τ for ch in channels]
    datas = [ch.G for ch in channels]
    σs = any(!isnothing(ch.σ) for ch in channels) ? [ch.σ for ch in channels] : nothing
    return fcs_fit(specs, τs, datas, p0s; σs, kwargs...)
end