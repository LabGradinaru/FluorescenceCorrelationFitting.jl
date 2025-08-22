"""
    τD_from_D(D, w0)

Convert diffusion coefficient `D` and lateral waist `w0` to the lateral diffusion time τD.
"""
@inline τD_from_D(D::Real, w0::Real) = w0^2 / (4D)

# clamp into (ε, 1-ε) to keep fractions valid without NaNs/Infs in fits
@inline function clamp01(x::T) where T
    epsT = eps(T)
    return clamp(x, epsT, one(T) - epsT)
end

# broadcastable vectors from Real or AbstractVector
@inline _asvec(x::AbstractVector) = x
@inline _asvec(x::Real) = (x,)

# normalize mixture weights to sum to 1 without mutating the input
function _normalize_weights(ws::AbstractVector)
    T = promote_type(eltype(ws), Float64)
    w = T.(ws)
    s = sum(w)
    s > 0 ? (w ./ s) : fill(inv(length(w)), length(w))
end

# multiplicative blinking/dynamics factor ∏_j [1 + K_j * exp(-t/τ_j)]
# returns 1 if no (τ,K) provided 
function _dynamics_factor(t, τs::AbstractVector, Ks::AbstractVector, ics::AbstractVector{Int})
    length(τs) == length(Ks) || throw(ArgumentError("τs and Ks must have same length"))
    sum(ics) == length(τs) || throw(ArgumentError("The number of components must match τs and Ks")) 
    if isempty(τs)
        return t isa AbstractVector ? ones(promote_type(eltype(t), Float64), length(t)) : one(promote_type(typeof(t), Float64))
    end
    if t isa AbstractVector
        T = promote_type(eltype(t), eltype(τs), eltype(Ks))
        out = ones(T, length(t))
        idx = 1
        @inbounds for i in eachindex(ics)
            nic = ics[i]
            end_idx = idx+nic-1
            @. out *= (one(T) + sum(Ks[idx:end_idx] * exp(- t/τs[idx:end_idx])))
        end
        return out
    else
        T = promote_type(typeof(t), eltype(τs), eltype(Ks))
        out = one(T)
        idx = 1
        @inbounds for i in eachindex(ics)
            nic = ics[i]
            end_idx = idx+nic-1
            out *= (one(T) + sum(Ks[idx:end_idx] * exp(- t/τs[idx:end_idx])))
        end
        return out
    end
end

# sum_i w_i * kernel(t, param_i), for t scalar or vector
function _mdiff(t, params::AbstractVector, weights::AbstractVector, kernel::Function)
    length(params) == length(weights) || throw(ArgumentError("params and weights must have same length"))
    w = _normalize_weights(weights)
    if t isa AbstractVector
        T = promote_type(eltype(t), eltype(params), eltype(weights))
        out = zeros(T, length(t))
        @inbounds for i in eachindex(params)
            @. out += T(w[i]) * kernel(t, T(params[i]))
        end
        return out
    else
        T = promote_type(typeof(t), eltype(params), eltype(weights))
        s = zero(T)
        @inbounds for i in eachindex(params)
            s += T(w[i]) * kernel(T(t), T(params[i]))
        end
        return s
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Base kernels (unit-amplitude, zero-offset)
# ─────────────────────────────────────────────────────────────────────────────

"2D diffusion kernel: 1 / (1 + t/τD)"
@inline function udc_2d(t, τD) 
    if t isa AbstractVector 
        @. inv(1 + t/τD) 
    else
        inv(1 + t/τD)
    end
end

"2D anomalous diffusion kernel: 1 / (1 + (t/τD)^α)"
@inline function udc_2d_anom(t, τD, α)
    if t isa AbstractVector
        @. inv(1 + (t/τD)^α)
    else
        inv(1 + (t/τD)^α)
    end
end

"3D diffusion kernel with structure factor s = z0/w0: 1 / ((1 + t/τD) * sqrt(1 + t/(s^2 τD)))"
@inline function udc_3d(t, τD, s)
    if t isa AbstractVector
        @. inv( (1 + t/τD) * sqrt(1 + t/(s^2 * τD)) )
    else
        inv( (1 + t/τD) * sqrt(1 + t/(s^2 * τD)) )
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Public models (with amplitude g0 and offset, plus optional blinking)
# ─────────────────────────────────────────────────────────────────────────────

@inline function _ndyn_from_len(total_extra::Int)
    total_extra ≥ 0 || throw(ArgumentError("parameter vector too short"))
    rem(total_extra, 2) == 0 || throw(ArgumentError("τ_dyn and K_dyn must have equal length"))
    total_extra ÷ 2
end

"""
    fcs_2d(t, p; scales, ics)

Single-component 2D diffusion with optional multiplicative dynamics (triplet/blinking).
The parameters vector `p` should be organized as

*   `p[1]` → τD; the diffusion time
*   `p[2]` → g0; the zero-lag autocorrelation
*   `p[3]` → offset; the offset of the correlation from 0
*   `p[4:m]` → τ_dyn; the dynamic lifetimes
*   `p[m+1:N]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime

`scales` converts from the normalized units of the input to match the units of time.
`ics` dictates the number of independent components for each dynamic contributor.

# Examples
`fcs_2d(times, [1e-4, 1.0, 0.0, 1e-6, 1e-7, 0.1, 0.1]; ics=[1,1])`

would attempt to fit the the regular diffusion model with initial parameters `τD` = 100 μs,
`g0` = 1.0, `offset` = 0.0 and two independent dynamic components, multiplying the diffusion model as
`(1 + T * exp(- t/ τ1)) * (1 + K * exp(- t/ τ2))`
"""
function fcs_2d(t::Union{Real,AbstractVector{<:Real}}, p::AbstractVector{<:Real}; 
                scales::Union{Nothing, AbstractVector}=nothing, 
                ics::Union{Nothing, AbstractVector{Int}}=nothing)
    L = length(p)
    isnothing(scales) && (scales = ones(L))
    L == length(scales) || throw(ArgumentError("Scaling and parameter vector must be of the same length."))
    L ≥ 3 || throw(ArgumentError("need at least 3 params: τD, g0, offset"))
    scaled_p = scales .* p

    m = _ndyn_from_len(L - 3)
    isnothing(ics) && (ics = ones(Int, m))
    sum(ics) == m || throw(ArgumentError("The number of dynamic components must be consistent among the parameters `p` and `ics`."))

    dyn = (m == 0) ? 1.0 : _dynamics_factor(t, @view(scaled_p[4:3+m]), @view(scaled_p[4+m:3+2m]), ics)
    udc = udc_2d(t, scaled_p[1])
    if t isa AbstractVector
        @. scaled_p[3] + scaled_p[2] * udc * dyn
    else
        scaled_p[3] + scaled_p[2] * udc * dyn
    end
end

"""
    fcs_2d_mixture(t,p)

Mixture of `n` 2D diffusion components with weights that are normalized internally.
The parameters vector `p` should be organized as

*   `p[1:n]` → τDs; the diffusion times of each diffuser
*   `p[n+1:2n]` → weights: the fraction of diffuser in each population
*   `p[2n+1]` → g0; the zero-lag autocorrelation
*   `p[2n+2]` → offset; the offset of the correlation from 0
*   `p[2n+3:2n+2+m]` → τ_dyn; the dynamic lifetimes
*   `p[2n+3+m:end]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime
"""
function fcs_2d_mdiff(t::Union{Real,AbstractVector{<:Real}}, p::AbstractVector{<:Real};
                      n_diff::Integer=1, scales::Union{Nothing,AbstractVector}=nothing,
                      ics::Union{Nothing, AbstractVector{Int}}=nothing)
    n_diff ≥ 1 || throw(ArgumentError("n_diff must be ≥ 1"))
    L = length(p)
    isnothing(scales) && (scales = ones(L))
    L == length(scales) || throw(ArgumentError("Scaling and parameter vector must be of the same length."))

    n = n_diff
    base = 2n + 2
    L ≥ base || throw(ArgumentError("p too short for n_diff=$n (need ≥ $(base))"))

    scaled_p = scales .* p
    m = _ndyn_from_len(L - base)
    isnothing(ics) && (ics = ones(Int, m))
    sum(ics) == m || throw(ArgumentError("The number of dynamic components must be consistent among the parameters `p` and `ics`."))

    τDs = @view scaled_p[1:n]
    wts = @view scaled_p[n+1:2n]
    g0 = scaled_p[2n+1]
    offset = scaled_p[2n+2]
    τdyn = m == 0 ? Float64[] : collect(@view scaled_p[base+1 : base+m])
    Kdyn = m == 0 ? Float64[] : collect(@view scaled_p[base+m+1 : base+2m])

    dyn = (m == 0) ? 1.0 : _dynamics_factor(t, τdyn, Kdyn, ics)
    mix = _mdiff(t, τDs, wts, (tt,τ)->udc_2d(tt,τ))
    if t isa AbstractVector
        @. offset + g0 * mix * dyn
    else
        offset + g0 * mix * dyn
    end
end

"""
    fcs_3d(t; τD, s, g0=1, offset=0, τ_dyn=[], K_dyn=[])

Single-component 3D diffusion with optional dynamics.
The parameters vector `p` should be organized as

*   `p[1]` → τD; the diffusion time
*   `p[2]` → g0; the zero-lag autocorrelation
*   `p[3]` → offset; the offset of the correlation from 0
*   `p[4]` → s; the structure factor `s = z0/w0`
*   `p[5:m]` → τ_dyn; the dynamic lifetimes
*   `p[m+1:N]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime
"""
function fcs_3d(t::Union{Real,AbstractVector{<:Real}}, p::AbstractVector{<:Real};
                scales::Union{Nothing, AbstractVector}=nothing,
                ics::Union{Nothing, AbstractVector{Int}}=nothing)
    L = length(p)
    isnothing(scales) && (scales = ones(L))
    L ≥ 4 || throw(ArgumentError("need at least 4 params: τD, g0, offset, s"))
    L == length(scales) || throw(ArgumentError("Scaling and parameter vector must be of the same length."))
    scaled_p = scales .* p

    m = _ndyn_from_len(L - 4)
    isnothing(ics) && (ics = ones(Int, m))
    sum(ics) == m || throw(ArgumentError("The number of dynamic components must be consistent among the parameters `p` and `ics`."))

    dyn = (m == 0) ? 1.0 : _dynamics_factor(t, @view(scaled_p[5:4+m]), @view(scaled_p[5+m:4+2m]), ics)
    udc = udc_3d(t, scaled_p[1], scaled_p[4])
    if t isa AbstractVector
        @. scaled_p[3] + scaled_p[2] * udc * dyn
    else
        scaled_p[3] + scaled_p[2] * udc * dyn
    end
end

"""
    fcs_3d_mixture(t; τDs, s, weights, g0=1, offset=0, τ_dyn=[], K_dyn=[])

Mixture of `n` 3D diffusion components sharing the same structure factor `s`.

*   `p[1:n]` → τDs; the diffusion times of each diffuser
*   `p[n+1:2n]` → weights: the fraction of diffuser in each population
*   `p[2n+1]` → g0; the zero-lag autocorrelation
*   `p[2n+2]` → offset; the offset of the correlation from 0
*   `p[2n+3]` → s; the structure factor `s = z0/w0`
*   `p[2n+3:2n+2+m]` → τ_dyn; the dynamic lifetimes
*   `p[2n+3+m:end]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime
"""
function fcs_3d_mdiff(t::Union{Real,AbstractVector{<:Real}}, p::AbstractVector{<:Real};
                      n_diff::Integer=1, scales::Union{Nothing,AbstractVector}=nothing,
                      ics::Union{Nothing, AbstractVector{Int}}=nothing)
    n_diff ≥ 1 || throw(ArgumentError("n_diff must be ≥ 1"))
    L = length(p)
    isnothing(scales) && (scales = ones(L))
    L == length(scales) || throw(ArgumentError("Scaling and parameter vector must be of the same length."))

    n = n_diff
    base = 2n + 3
    L ≥ base || throw(ArgumentError("p too short for n_diff=$n (need ≥ $(base))"))

    scaled_p = scales .* p
    m = _ndyn_from_len(L - base)
    isnothing(ics) && (ics = ones(Int, m))
    sum(ics) == m || throw(ArgumentError("The number of dynamic components must be consistent among the parameters `p` and `ics`."))

    τDs = @view scaled_p[1:n]
    wts = @view scaled_p[n+1:2n]
    g0 = scaled_p[2n+1]
    offset = scaled_p[2n+2]
    s = scaled_p[2n+3]
    τdyn = m == 0 ? Float64[] : collect(@view scaled_p[base+1 : base+m])       # 2n+4 : 2n+3+m
    Kdyn = m == 0 ? Float64[] : collect(@view scaled_p[base+m+1 : base+2m])    # 2n+4+m : 2n+3+2m

    dyn = (m == 0) ? 1.0 : _dynamics_factor(t, τdyn, Kdyn, ics)
    mix = _mdiff(t, τDs, wts, (tt,τ)->udc_3d(tt, τ, s))
    if t isa AbstractVector
        @. offset + g0 * mix * dyn
    else
        offset + g0 * mix * dyn
    end
end