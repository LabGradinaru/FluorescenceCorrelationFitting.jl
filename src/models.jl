"""
    τD(D, w0)
Convert diffusion coefficient `D` and lateral waist `w0` to the lateral diffusion time τD.
"""
@inline τD(D::Real, w0::Real) = w0^2 / (4D)
"""
    diffusivity(τD, w0)
Convert diffusion time `τD` and beam waist `w0` to the diffusivity.
"""
@inline diffusivity(τD::Real, w0::Real) = w0^2 / (4τD)

"""
    Veff(w0, z0)
Calculate the effective volume from the measured FCS parameters.
"""
@inline Veff(w0::Real, z0::Real) = π^(3/2) * w0^2 * z0

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

# determine the number of dynamic components based on the parameter vector length
@inline function _ndyn_from_len(total_extra::Int)
    total_extra ≥ 0 || throw(ArgumentError("parameter vector too short"))
    rem(total_extra, 2) == 0 || throw(ArgumentError("τ_dyn and K_dyn must have equal length"))
    total_extra ÷ 2
end

# multiplicative blinking/dynamics factor ∏_j [1 + K_j * exp(-t/τ_j)]
# returns 1 if no (τ,K) provided 
function _dynamics_factor(t, τs::AbstractVector, Ks::AbstractVector, ics::AbstractVector{Int})
    length(τs) == length(Ks) || throw(ArgumentError("τs and Ks must have same length"))
    sum(ics) == length(τs) || throw(ArgumentError("The number of components must match τs and Ks"))

    if isempty(τs)
        return t isa AbstractVector ?
            ones(promote_type(eltype(t), Float64), length(t)) :
            one(promote_type(typeof(t), Float64))
    end

    if t isa AbstractVector
        T = promote_type(eltype(t), eltype(τs), eltype(Ks))
        out = ones(T, length(t))
        idx = 1
        @inbounds for i in eachindex(ics)
            nic = ics[i]
            # accumulate s(t) = Σ_k K_k * exp(-t/τ_k) over this block
            s = zeros(T, length(t))
            @inbounds for j = 1:nic
                k = idx + j - 1
                τ = τs[k]
                K = Ks[k]
                @. s += K * (exp(-t/τ) - 1)
            end
            @. out *= (one(T) + s)
            idx += nic  # advance to next block
        end
        return out
    else
        T = promote_type(typeof(t), eltype(τs), eltype(Ks))
        out = one(T)
        idx = 1
        @inbounds for i in eachindex(ics)
            nic = ics[i]
            s = zero(T)
            @inbounds for j = 1:nic
                k = idx + j - 1
                s += Ks[k] * (exp(-t/τs[k]) - 1)
            end
            out *= (one(T) + s)
            idx += nic  # advance to next block
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

"2D diffusion kernel"
@inline function udc_2d(t, τD) 
    if t isa AbstractVector 
        @. inv(1 + t/τD) 
    else
        inv(1 + t/τD)
    end
end

"2D anomalous diffusion kernel"
@inline function udc_2d_anom(t, τD, α)
    if t isa AbstractVector
        @. inv(1 + (t/τD)^α)
    else
        inv(1 + (t/τD)^α)
    end
end

"3D diffusion kernel with structure factor κ = z0/w0"
@inline function udc_3d(t, τD, κ)
    if t isa AbstractVector
        @. inv( (1 + t/τD) * sqrt(1 + t/(κ^2 * τD)) )
    else
        inv( (1 + t/τD) * sqrt(1 + t/(κ^2 * τD)) )
    end
end

"3D anomalous diffusion kernel with structure factor κ = z0/w0"
@inline function udc_3d_anom(t, τD, α)
    if t isa AbstractVector
        @. inv((1 + (t/τD)^α) * sqrt(1 + (t/τD)^α / κ^2))
    else
        inv((1 + (t/τD)^α) * sqrt(1 + (t/τD)^α / κ^2))
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Public models (with amplitude g0 and offset, plus optional blinking)
# ─────────────────────────────────────────────────────────────────────────────

"""
    fcs_2d(t, p; scales, ics, diffusivity)

Single-component 2D diffusion with optional multiplicative dynamics (triplet/blinking).
The parameters vector `p` should be organized as

*   `p[1]` → τD; the diffusion time
*   `p[2]` → g0; the zero-lag autocorrelation
*   `p[3]` → offset; the offset of the correlation from 0
*   `p[4:m]` → τ_dyn; the dynamic lifetimes
*   `p[m+1:N]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime

`scales` converts from the normalized units of the input to match the units of time.
`ics` dictates the number of independent components for each dynamic contributor.
`diffusivity` can be provided as a fixed parameter (e.g., for callibration), in which case
`p[1]` is interpretted as the 1/e radius, w0.
`offset` can similarly be fixed, in which case all of the above shift up by 1.


# Examples

Evaluate the kernel of a 2d diffusion, `1 / (1 + t/τD)` from times `1e-6` to `1e-5`
with `τD` = 1 ms multiplied by `g0` = 1.0 and two independent dynamic components,
`(1 - T + T * exp(- t/ τ1)) * (1 - K + K * exp(- t/ τ2))` with `T = K = 0.1` and `τ1 = 1e-4`, `τ2 = 1e-6` seconds:

```jldoctest
julia> fcs_2d(1e-6:1e-6:1e-5, [1e-3, 1.0, 0.0, 1e-4, 1e-6, 0.1, 0.1])

10-element Vector{Float64}:
 1.1382968204673092
 1.11065863303906
 1.099208790202291
 1.0937116359843249
 1.0904087865516843
 1.0879201516749173
 1.085738900216725
 1.0836788468371112
 1.0816715434967463
 1.0796917748436876
```

As above but with two dependent dynamic components, `(1 + T * exp(- t/ τ1) + K * exp(- t/ τ2) - T - K)`:

```jldoctest
julia> fcs_2d(1e-6:1e-6:1e-5, [1e-3, 0.5, 0.0, 1e-4, 1e-6, 0.1, 0.1]; ics=[2])

10-element Vector{Float64}:
 1.1346582692228384
 1.1093347262019329
 1.098727078954773
 1.093536362354687
 1.0903450120895324
 1.0878969468947233
 1.085730456988233
 1.0836757747038235
 1.0816704256764438
 1.079691368115418
```
"""
function fcs_2d(t::Union{Real,AbstractVector{<:Real}}, p::AbstractVector{<:Real}; 
                scales::Union{Nothing, AbstractVector}=nothing, 
                ics::Union{Nothing, AbstractVector{Int}}=nothing,
                diffusivity::Union{Nothing,Real}=nothing,
                offset::Union{Nothing,Real}=nothing)
    L = length(p)
    isnothing(scales) && (scales = ones(L))
    L == length(scales) || throw(ArgumentError("Scaling and parameter vector must be of the same length."))
    
    n_default_params::Int = isnothing(offset) ? 3 : 2
    L ≥ n_default_params || throw(ArgumentError("Need at least $n_default_params input parameters"))

    scaled_p = scales .* p

    m = _ndyn_from_len(L - n_default_params)
    isnothing(ics) && (ics = ones(Int, m))
    sum(ics) == m || throw(ArgumentError("The number of dynamic components must be consistent among the parameters `p` and `ics`."))

    dyn = (m == 0) ? 1.0 : 
        _dynamics_factor(t, @view(scaled_p[n_default_params+1:n_default_params+m]), 
                         @view(scaled_p[n_default_params+1+m:n_default_params+2m]), ics)

    udc = isnothing(diffusivity) ? udc_2d(t, scaled_p[1]) : udc_2d(t, τD(diffusivity, scaled_p[1]))
    if t isa AbstractVector
        if isnothing(offset)
            @. scaled_p[3] + scaled_p[2] * udc * dyn
        else
            @. offset + scaled_p[2] * udc * dyn
        end
    else
        if isnothing(offset)
            scaled_p[3] + scaled_p[2] * udc * dyn
        else
            @. offset + scaled_p[2] * udc * dyn
        end
    end
end


"""
    fcs_2d_mdiff(t, p; scales, ics, diffusivity)

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
    fcs_2d_anom(t, p; scales, ics, diffusivity)

Single-component 2D anomolous diffusion with optional multiplicative dynamics (triplet/blinking).
The parameters vector `p` should be organized as

*   `p[1]` → τD; the diffusion time
*   `p[2]` → g0; the zero-lag autocorrelation
*   `p[3]` → offset; the offset of the correlation from 0
*   `p[4]` → α; the anomalous exponent 
*   `p[5:m]` → τ_dyn; the dynamic lifetimes
*   `p[m+1:N]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime
"""
function fcs_2d_anom(t::Union{Real,AbstractVector{<:Real}}, p::AbstractVector{<:Real}; 
                     scales::Union{Nothing, AbstractVector}=nothing, 
                     ics::Union{Nothing, AbstractVector{Int}}=nothing,
                     diffusivity::Union{Nothing,Real}=nothing,
                     offset::Union{Nothing,Real}=nothing)
    L = length(p)
    isnothing(scales) && (scales = ones(L))
    
    n_default_params::Int = isnothing(offset) ? 4 : 3
    L ≥ n_default_params || throw(ArgumentError("Need at least $n_default_params input parameters."))

    L == length(scales) || throw(ArgumentError("Scaling and parameter vector must be of the same length."))
    scaled_p = scales .* p

    m = _ndyn_from_len(L - n_default_params)
    isnothing(ics) && (ics = ones(Int, m))
    sum(ics) == m || throw(ArgumentError("The number of dynamic components must be consistent among the parameters `p` and `ics`."))

    dyn = (m == 0) ? 1.0 : 
        _dynamics_factor(t, @view(scaled_p[n_default_params+1:n_default_params+m]), 
                         @view(scaled_p[n_default_params+1+m:n_default_params+2m]), ics)

    udc = isnothing(diffusivity) ? 
        udc_2d_anom(t, scaled_p[1], scaled_p[n_default_params]) : 
        udc_2d_anom(t, τD(diffusivity, scaled_p[1]), scaled_p[n_default_params])
    if t isa AbstractVector
        if isnothing(offset)
            @. scaled_p[3] + scaled_p[2] * udc * dyn
        else
            @. offset + scaled_p[2] * udc * dyn
        end
    else
        if isnothing(offset)
            scaled_p[3] + scaled_p[2] * udc * dyn
        else
            @. offset + scaled_p[2] * udc * dyn
        end
    end
end


"""
    fcs_3d(t, p; scales, ics, diffusivity)

Single-component 3D diffusion with optional dynamics.
The parameters vector `p` should be organized as

*   `p[1]` → τD; the diffusion time
*   `p[2]` → g0; the zero-lag autocorrelation
*   `p[3]` → offset; the offset of the correlation from 0
*   `p[4]` → κ; the structure factor `κ = z0/w0`
*   `p[5:m]` → τ_dyn; the dynamic lifetimes
*   `p[m+1:N]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime
"""
function fcs_3d(t::Union{Real,AbstractVector{<:Real}}, p::AbstractVector{<:Real};
                scales::Union{Nothing, AbstractVector}=nothing,
                ics::Union{Nothing, AbstractVector{Int}}=nothing,
                diffusivity::Union{Nothing,Real}=nothing,
                offset::Union{Nothing,Real}=nothing)
    L = length(p)
    isnothing(scales) && (scales = ones(L))
    
    n_default_params::Int = isnothing(offset) ? 4 : 3
    L ≥ n_default_params || throw(ArgumentError("Need at least $n_default_params input parameters."))

    L == length(scales) || throw(ArgumentError("Scaling and parameter vector must be of the same length."))
    scaled_p = scales .* p

    m = _ndyn_from_len(L - n_default_params)
    isnothing(ics) && (ics = ones(Int, m))
    sum(ics) == m || throw(ArgumentError("The number of dynamic components must be consistent among the parameters `p` and `ics`."))

    dyn = (m == 0) ? 1.0 : 
        _dynamics_factor(t, @view(scaled_p[n_default_params+1:n_default_params+m]), 
                         @view(scaled_p[n_default_params+1+m:n_default_params+2m]), ics)

    udc = isnothing(diffusivity) ? 
        udc_3d(t, scaled_p[1], scaled_p[n_default_params]) : 
        udc_3d(t, τD(diffusivity, scaled_p[1]), scaled_p[n_default_params])
    if t isa AbstractVector
        if isnothing(offset)
            @. scaled_p[3] + scaled_p[2] * udc * dyn
        else
            @. offset + scaled_p[2] * udc * dyn
        end
    else
        if isnothing(offset)
            scaled_p[3] + scaled_p[2] * udc * dyn
        else
            @. offset + scaled_p[2] * udc * dyn
        end
    end
end


"""
    fcs_3d_mdiff(t, p; scales, ics, diffusivity)

Mixture of `n` 3D diffusion components sharing the same structure factor `κ`.

*   `p[1:n]` → τDs; the diffusion times of each diffuser
*   `p[n+1:2n]` → weights: the fraction of diffuser in each population
*   `p[2n+1]` → g0; the zero-lag autocorrelation
*   `p[2n+2]` → offset; the offset of the correlation from 0
*   `p[2n+3]` → κ; the structure factor `κ = z0/w0`
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