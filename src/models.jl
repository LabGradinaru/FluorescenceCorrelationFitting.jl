const BOLTZMANN = 1.380649e-23 # Boltzmann constant in SI units
const AVAGADROS = 6.022141e23 # Avagadro's number in SI units

# Common error messages
const w0_ERROR = "w0 must be positive"
const κ_ERROR = "κ must be positive"
const D_ERROR = "Diffusivity must be positive"
const NDIFF_ERROR = "n_diff must be ≥ 1"
const SP_ERROR = "Scaling and parameter vectors must be of the same length."
PNDIFF_ERROR(x) = "Parameter vector too short for " * nameof(x) * "=$x"
const WEIGHTS_ERROR = "Sum of weights must be ≤ 1"
const PICS_ERROR = "Mismatch between dynamics in the parameter vector and the independent components."
NPARAM_ERROR(x) = "Need at least " * x * " input parameters."


# ─────────────────────────────────────────────────────────────────────────────
# Convenience calculators
# ─────────────────────────────────────────────────────────────────────────────

"""
    τD(D, w0)
Convert diffusion coefficient `D` and lateral waist `w0` to the lateral diffusion time τD.
"""
@inline function τD(D::Real, w0::Real)
    w0 > 0 || throw(ArgumentError(w0_ERROR))
    D  > 0 || throw(ArgumentError(D_ERROR))
    return w0^2 / (4D)
end
"""
    diffusivity(τD, w0)
Convert diffusion time `τD` and beam waist `w0` to the diffusivity.
"""
@inline function diffusivity(τD::Real, w0::Real)
    w0 > 0 || throw(ArgumentError(w0_ERROR))
    τD > 0 || throw(ArgumentError("τD must be positive"))
    return w0^2 / (4τD)
end

"""
    volume(w0, κ)
Calculate the effective volume from fitted FCS parameters.
"""
@inline function volume(w0::Real, κ::Real)
    w0 > 0 || throw(ArgumentError(w0_ERROR))
    κ > 0 || throw(ArgumentError(κ_ERROR))
    return π^(3/2) * w0^3 * κ
end
"""
    area(w0)
Calculate the area formed by the beam waist `w0`.
"""
@inline function area(w0::Real) 
    w0 > 0 || throw(ArgumentError(w0_ERROR))
    return π * w0^2
end

"""
    concentration(w0, κ, g0; Ks=[], ics=[0])

Estimate the **molar concentration** (in mol/L) from FCS fit parameters.

# Arguments
- `w0::Real`: Lateral 1/e² Gaussian waist of the detection PSF (meters).
- `κ::Real`: Axial structure factor `κ = wz / w0` (dimensionless).
- `g0::Real`: Fitted correlation amplitude **at τ→0** (dimensionless).
              In standard FCS models, the measured `g0` is inflated by blinking (“dark states”).
- `Ks::AbstractVector` (keyword): Dark-state **equilibrium fractions** for the kinetic terms
   used in the model (each in `[0,1)`), ordered exactly as in your dynamics kernel.
- `ics::AbstractVector{Int}` (keyword): Block sizes describing how `Ks` (and their times)
   are grouped into **independent** multiplicative blinking factors. For example,
   `ics = [2, 1]` means the first blinking block has 2 components, the second block has 1.
"""
@inline function concentration(w0::Real, κ::Real, g0::Real; 
                               Ks::AbstractVector = [],
                               ics::AbstractVector{Int} = [0])
    g0 > 0 || throw(ArgumentError("g0 must be positive"))
    κ > 0 || throw(ArgumentError(κ_ERROR))
    w0 > 0 || throw(ArgumentError(w0_ERROR))
    all(0 .<= Ks .< 1) || throw(ArgumentError("All Ks must lie in [0,1)"))

    # Validate / normalize ics
    isempty(Ks) ?
        (ics == [0]) || throw(ArgumentError("With Ks=[], use ics=[0]")) :
        sum(ics) == length(Ks) || throw(ArgumentError("sum(ics) must equal length(Ks)"))

    # Blink prefactor at τ→0: B(0) = ∏_blocks (1 + Σ_i n_i),  n_i = K_i/(1-K_i)
    B0 = one(Float64)
    idx = 1
    for b in eachindex(ics)
        nb = ics[b]
        nb == 0 && continue

        s = 0.0
        @inbounds for j = 1:nb
            K = float(Ks[idx + j - 1])
            s += K / (1 - K)
        end
        B0 *= (1 + s)
        idx += nb
    end

    Neff = B0 / g0
    return Neff / (AVAGADROS * volume(w0, κ) * 1000) # 1000: m^3 → L
end
"""
    surface_density(w0, κ, g0; Ks=[], ics=[0])

Estimate the **molar surface density** (in mol/m^2) from FCS fit parameters.
Analogue to `concentration` when the a 2d fit is performed.
"""
@inline function surface_density(w0::Real, g0::Real; 
                                 Ks::AbstractVector = [],
                                 ics::AbstractVector{Int} = [0])
    g0 > 0 || throw(ArgumentError("g0 must be positive"))
    w0 > 0 || throw(ArgumentError(w0_ERROR))
    all(0 .<= Ks .< 1) || throw(ArgumentError("All Ks must lie in [0,1)"))

    # Validate / normalize ics
    isempty(Ks) ?
        (ics == [0]) || throw(ArgumentError("With Ks=[], use ics=[0]")) :
        sum(ics) == length(Ks) || throw(ArgumentError("sum(ics) must equal length(Ks)"))

    # Blink prefactor at τ→0: B(0) = ∏_blocks (1 + Σ_i n_i),  n_i = K_i/(1-K_i)
    B0 = one(Float64)
    idx = 1
    for b in eachindex(ics)
        nb = ics[b]
        nb == 0 && continue

        s = 0.0
        @inbounds for j = 1:nb
            K = float(Ks[idx + j - 1])
            s += K / (1 - K)
        end
        B0 *= (1 + s)
        idx += nb
    end

    Neff = B0 / g0
    return Neff / (AVAGADROS * area(w0))
end

"""
    hydrodynamic(τD, w0; T=293.0, η=1.0016e-3)
Calculate the effective hydrodynamic radius of a molecule from its characteristic 
diffusion time and beam waist using the Stokes-Einstein relation.
"""
@inline hydrodynamic(τD::Real, w0::Real; kwargs...) =
    hydrodynamic(diffusivity(τD, w0); kwargs...)
"""
    hydrodynamic(D; T=293.0, η=1.0016e-3, D_err=nothing)
Calculate the effective hydrodynamic radius of a molecule from its diffusion
coefficient using the Stokes-Einstein relation.

If an error in the diffusivity is provided, returns the error in Rh estimate
"""
@inline function hydrodynamic(D::Real; T::Real=293.0, η=1.0016e-3, 
                              D_err::Union{Nothing,Real}=nothing)
    D > 0 || throw(ArgumentError(D_ERROR))
    T > 0 || throw(ArgumentError("Temperature must be positive."))
    η > 0 || throw(ArgumentError("Viscosity must be positive."))
    isnothing(D_err) || D_err > 0 || throw(ArgumentError("D_err must be positive if not nothing."))
    
    _scale = BOLTZMANN * T / (6π * η)
    if isnothing(D_err)
        _scale / D
    else
        _scale / D, _scale * D_err / D^2
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

# determine the number of dynamic components based on the parameter vector length
@inline function _ndyn_from_len(total_extra::Int)
    total_extra ≥ 0 || throw(ArgumentError("p too short."))
    rem(total_extra, 2) == 0 || throw(ArgumentError("τ_dyn and K_dyn must have the same length."))
    total_extra ÷ 2
end

# multiplicative blinking/dynamics factor ∏_j [1 + K_j * exp(-t/τ_j)]
# returns 1 if no (τ,K) provided 
function _dynamics_factor(t, τs::AbstractVector, Ks::AbstractVector, ics::AbstractVector{Int})
    length(τs) == length(Ks) || throw(ArgumentError("τs and Ks must have same length."))
    sum(ics) == length(τs) || throw(ArgumentError("The number of components must match τs and Ks"))
    all(0 .<= Ks .< 1) || throw(ArgumentError("All Ks must lie in [0,1)"))

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
function _mdiff(t, τDs::AbstractVector, wts::AbstractVector, kernel::Function)
    n = length(τDs)
    n ≥ 1 || throw(ArgumentError("Need at least one τD"))
    length(wts) + 1 == n || throw(ArgumentError("There must be one less weight than there are τDs."))
    
    if n == 1
        w_full = (1.0,)
    else
        sum(wts) ≤ 1 || throw(ArgumentError(WEIGHTS_ERROR))
        w_full = (vcat(wts, 1 - sum(wts)))::Vector{Float64}
    end

    if t isa AbstractVector
        T = promote_type(eltype(t), eltype(τDs), eltype(wts))
        out = zeros(T, length(t))
        @inbounds for i in eachindex(τDs)
            wi = (n == 1) ? 1.0 : w_full[i]
            out .+= wi .* kernel(t, τDs[i])
        end
        return out
    else
        T = promote_type(typeof(t), eltype(τDs), eltype(wts))
        mix = zero(T)
        @inbounds for i in eachindex(τDs)
            wi = (n == 1) ? 1.0 : w_full[i]
            mix += wi * kernel(t, τDs[i])
        end
        return mix
    end
end

# Weighted mixture of n anomalous 2D diffusers with per-component αᵢ.
# τDs :: length-n vector of τD[i]
# wts :: length-(n-1) vector for the first n-1 weights; last is implied: 1 - sum(wts)
# αs  :: length-n vector of α[i]
@inline function _mdiff_anom(t, τDs::AbstractVector, αs::AbstractVector, wts::AbstractVector, kernel::Function)
    n = length(τDs)
    length(αs) == n || throw(ArgumentError("length(αs) must equal length(τDs)"))
    length(wts) + 1 == n || throw(ArgumentError("There must be one less weight than there are τDs and αs."))
    
    if n == 1
        w_full = (1.0,)
    else
        sum(wts) ≤ 1 || throw(ArgumentError(WEIGHTS_ERROR))
        w_full = (vcat(wts, 1 - sum(wts)))::Vector{Float64}
    end

    if t isa AbstractVector
        out = zeros(length(t))
        @inbounds for i in eachindex(τDs)
            wi = (n == 1) ? 1.0 : w_full[i]
            out .+= wi .* kernel(t, τDs[i], αs[i])
        end
        return out
    else
        out = 0.0
        @inbounds for i in eachindex(τDs)
            wi = (n == 1) ? 1.0 : w_full[i]
            out += wi * kernel(t, τDs[i], αs[i])
        end
        return out
    end
end


# ─────────────────────────────────────────────────────────────────────────────
# Base kernels (unit-amplitude, zero-offset)
# ─────────────────────────────────────────────────────────────────────────────

"2D diffusion kernel"
@inline udc_2d(t, τD) = @. inv(1 + t/τD) 

"2D anomalous diffusion kernel with exponent α"
@inline udc_2d_anom(t, τD, α) = @. inv(1 + (t/τD)^α)

"3D diffusion kernel with structure factor κ = z0/w0"
@inline udc_3d(t, τD, κ) = @. inv((1 + t/τD) * sqrt(1 + t/(κ^2 * τD)))

"3D anomalous diffusion kernel with anomalous exponent α and structure factor κ = z0/w0"
@inline udc_3d_anom(t, τD, κ, α) = @. inv((1 + (t/τD)^α) * sqrt(1 + (t/τD)^α / κ^2))


# ─────────────────────────────────────────────────────────────────────────────
# Public models (with amplitude g0 and offset, plus optional blinking)
# ─────────────────────────────────────────────────────────────────────────────

"""
    fcs_2d(t, p; scales, ics, diffusivity, offset)

Single-component 2D diffusion with optional multiplicative dynamics (triplet/blinking).
The parameters vector `p` (no fixed offset) should be organized as

*   `p[1]` → g0; the zero-lag autocorrelation
*   `p[2]` → offset; the offset of the correlation from 0
*   `p[3]` → τD; the diffusion time
*   `p[4:3+m]` → τ_dyn; the dynamic lifetimes
*   `p[4+m:3+2m]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime

If `offset` is provided, `p[2]` is omitted and indices after it shift left by one.
If `diffusivity` is provided, `p[3]` is interpretted as the 1/e radius, w0.
`scales` converts from the normalized units of the input to match the units of time.
`ics` dictates the number of independent components for each dynamic contributor.


# Examples

Evaluate the kernel of a 2d diffusion, `1 / (1 + t/τD)` from times `1e-6` to `1e-5`
with `τD` = 1 ms multiplied by `g0` = 1.0 and two independent dynamic components,
`(1 - T + T * exp(- t/ τ1)) * (1 - K + K * exp(- t/ τ2))` with `T = K = 0.1` and `τ1 = 1e-4`, `τ2 = 1e-6` seconds:

```jldoctest
julia> fcs_2d(1e-6:1e-6:1e-5, [1.0, 0.0, 1e-3, 1e-4, 1e-6, 0.1, 0.1])

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
julia> fcs_2d(1e-6:1e-6:1e-5, [0.5, 0.0, 1e-3, 1e-4, 1e-6, 0.1, 0.1]; ics=[2])

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
    L == length(scales) || throw(ArgumentError(SP_ERROR))
    
    base::Int = isnothing(offset) ? 3 : 2 # base includes: g0, (offset), τD
    L ≥ base || throw(ArgumentError(NPARAM_ERROR))

    sp = scales .* p
    m = _ndyn_from_len(L - base)
    isnothing(ics) && (ics = ones(Int, m))
    sum(ics) == m || throw(ArgumentError(PICS_ERROR))

    dyn = (m == 0) ? 1.0 : 
        _dynamics_factor(t, @view(sp[base+1:base+m]), 
                         @view(sp[base+m+1:base+2m]), ics)
    
    g0 = sp[1]
    τDe = isnothing(diffusivity) ? sp[base] : τD(diffusivity, sp[base])
    udc = udc_2d(t, τDe)
    if isnothing(offset)
        @. sp[2] + g0 * udc * dyn
    else
        @. offset + g0 * udc * dyn
    end
end

"""
    fcs_2d_mdiff(t, p; scales, ics, offset)

Mixture of `n` 2D diffusion components with weights that are normalized internally.
The parameters vector `p` should be organized as

*   `p[1]` → g0; the zero-lag autocorrelation
*   `p[2]` → offset; the offset of the correlation from 0
*   `p[3:n+2]` → τDs; the diffusion times of each diffuser
*   `p[n+3:2n+1]` → ws; the fraction of diffuser in the first n-1 populations.
                    sum of weights constrained by unity so only n-1 dof are required
*   `p[2n+2:2n+1+m]` → τ_dyn; the dynamic lifetimes
*   `p[2n+2+m:end]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime

If `offset` is provided, `p[2]` is omitted and indices after it shift left by one.
"""
function fcs_2d_mdiff(t::Union{Real,AbstractVector{<:Real}}, p::AbstractVector{<:Real};
                      n_diff::Integer=1, scales::Union{Nothing,AbstractVector}=nothing,
                      ics::Union{Nothing, AbstractVector{Int}}=nothing,
                      offset::Union{Nothing, Real}=nothing)
    n_diff ≥ 1 || throw(ArgumentError(NDIFF_ERROR))
    L = length(p)
    isnothing(scales) && (scales = ones(L))
    L == length(scales) || throw(ArgumentError(SP_ERROR))

    base = isnothing(offset) ? 2 : 1 # base includes: g0, (offset)
    L ≥ base || throw(ArgumentError(PNDIFF_ERROR(n_diff)))
    sp = scales .* p

    τDs = collect(@view sp[base+1:base+n_diff])
    w_end = base + n_diff + (n_diff > 1 ? (n_diff-1) : 0)
    wts = (n_diff == 1) ? Float64[] : collect(@view sp[base+n_diff+1:w_end])
    (n_diff == 1 || sum(wts) ≤ 1) || throw(ArgumentError(WEIGHTS_ERROR))
    
    m = _ndyn_from_len(L - w_end)
    isnothing(ics) && (ics = ones(Int, m))
    sum(ics) == m || throw(ArgumentError(PICS_ERROR))
    τdyn = m == 0 ? Float64[] : collect(@view sp[w_end+1:w_end+m])
    Kdyn = m == 0 ? Float64[] : collect(@view sp[w_end+m+1:w_end+2m])

    g0 = sp[1]
    dyn = (m == 0) ? 1.0 : _dynamics_factor(t, τdyn, Kdyn, ics)
    mix = _mdiff(t, τDs, wts, (tt,τ)->udc_2d(tt,τ))

    if isnothing(offset)
        @. sp[2] + g0 * mix * dyn
    else
        @. offset + g0 * mix * dyn
    end
end

"""
    fcs_2d_anom(t, p; scales, ics, diffusivity, offset)

Single-component 2D anomolous diffusion with optional multiplicative dynamics (triplet/blinking).
The parameters vector `p` should be organized as

*   `p[1]` → g0; the zero-lag autocorrelation
*   `p[2]` → offset; the offset of the correlation from 0
*   `p[3]` → τD; the diffusion time
*   `p[4]` → α; the anomalous exponent 
*   `p[5:m]` → τ_dyn; the dynamic lifetimes
*   `p[m+1:N]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime

If `offset` is provided, `p[2]` is omitted and indices after it shift left by one.
"""
function fcs_2d_anom(t::Union{Real,AbstractVector{<:Real}}, p::AbstractVector{<:Real}; 
                     scales::Union{Nothing, AbstractVector}=nothing, 
                     ics::Union{Nothing, AbstractVector{Int}}=nothing,
                     diffusivity::Union{Nothing,Real}=nothing,
                     offset::Union{Nothing,Real}=nothing)
    L = length(p)
    isnothing(scales) && (scales = ones(L))
    L == length(scales) || throw(ArgumentError(SP_ERROR))

    base::Int = isnothing(offset) ? 4 : 3 # base includes: g0, (offset), τD, α
    L ≥ base || throw(ArgumentError(NPARAM_ERROR))
    sp = scales .* p

    m = _ndyn_from_len(L - base)
    isnothing(ics) && (ics = ones(Int, m))
    sum(ics) == m || throw(ArgumentError(PICS_ERROR))
    τdyn = m == 0 ? Float64[] : collect(@view sp[base+1:base+m])
    Kdyn = m == 0 ? Float64[] : collect(@view sp[base+m+1:base+2m])

    g0 = sp[1]
    dyn = (m == 0) ? 1.0 : _dynamics_factor(t, τdyn, Kdyn, ics)
    τDe = isnothing(diffusivity) ? sp[base-1] : τD(diffusivity, sp[base-1])
    udc = udc_2d_anom(t, τDe, sp[base])

    if isnothing(offset)
        @. sp[2] + g0 * udc * dyn
    else
        @. offset + g0 * udc * dyn
    end
end

"""
    fcs_2d_anom_mdiff(t, p; n_diff=1, scales, ics, offset)

Mixture of `n_diff` 2D anomalous diffusers with **per-population αᵢ**
(e.g., “free” α≈1 and “confined” α<1 fractions), plus optional blinking dynamics.

# Parameters (no fixed offset)
*   `p[1]` → g0; the zero-lag autocorrelation
*   `p[2]` → offset; the offset of the correlation from 0
*   `p[3:n+2]` → τDs; the diffusion times of each diffuser
*   `p[n+3:2n+2]` → αs; the anomalous exponent for each population
*   `p[2n+3:3n+1]` → weights; the fraction of diffuser in the first n-1 populations.
                    sum of weights constrained by unity so only n-1 dof are required
*   `p[3n+2:3n+m+1]` → τ_dyn; the dynamic lifetimes
*   `p[3n+m+2:3n+2m+1]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime

If `offset` is provided, `p[2]` is omitted and indices after it shift left by one.
"""
function fcs_2d_anom_mdiff(t::Union{Real,AbstractVector{<:Real}}, p::AbstractVector{<:Real};
                           n_diff::Integer=1, scales::Union{Nothing,AbstractVector}=nothing,
                           ics::Union{Nothing, AbstractVector{Int}}=nothing,
                           offset::Union{Nothing, Real}=nothing)
    n_diff ≥ 1 || throw(ArgumentError(NDIFF_ERROR))
    L = length(p)
    isnothing(scales) && (scales = ones(L))
    L == length(scales) || throw(ArgumentError(SP_ERROR))

    base = isnothing(offset) ? 2 : 1 # base includes: g0, (offset)
    scaled_p = scales .* p

    # diffusion times
    τD_end = base + n_diff
    L ≥ τD_end || throw(ArgumentError("p too short for $n_diff diffusion times"))
    τDs = collect(@view scaled_p[base+1:τD_end])

    # anomalous exponents
    α_end = base + 2n_diff
    L ≥ α_end || throw(ArgumentError("p too short for $n_diff anomalous exponents"))
    αs = collect(@view scaled_p[base+n_diff+1:α_end])
    
    # weights
    w_end = α_end + (n_diff > 1 ? (n_diff-1) : 0)
    wts = n_diff == 1 ? Float64[] : collect(@view scaled_p[α_end+1:w_end])
    sum(wts) ≤ 1 || throw(ArgumentError(WEIGHTS_ERROR))

    # dynamics
    m = _ndyn_from_len(L - w_end)
    isnothing(ics) && (ics = ones(Int, m))
    sum(ics) == m || throw(ArgumentError(PICS_ERROR))
    τdyn = m == 0 ? Float64[] : collect(@view scaled_p[w_end+1:w_end+m])
    Kdyn = m == 0 ? Float64[] : collect(@view scaled_p[w_end+m+1:w_end+2m])

    g0 = scaled_p[1]
    dyn = m == 0 ? 1.0 : _dynamics_factor(t, τdyn, Kdyn, ics)
    mix = _mdiff_anom(t, τDs, αs, wts, (t,τ,α)->udc_2d_anom(t,τ,α))

    if isnothing(offset)
        @. scaled_p[2] + g0 * mix * dyn
    else
        @. offset + g0 * mix * dyn
    end
end

"""
    fcs_3d(t, p; scales, ics, diffusivity, offset)

Single-component 3D diffusion with optional dynamics.
The parameters vector `p` should be organized as

*   `p[1]` → g0; the zero-lag autocorrelation
*   `p[2]` → offset; the offset of the correlation from 0
*   `p[3]` → κ; the structure factor `κ = z0/w0`
*   `p[4]` → τD; the diffusion time
*   `p[5:4+m]` → τ_dyn; the dynamic lifetimes
*   `p[5+m:4+2m]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime
"""
function fcs_3d(t::Union{Real,AbstractVector{<:Real}}, p::AbstractVector{<:Real};
                scales::Union{Nothing, AbstractVector}=nothing,
                ics::Union{Nothing, AbstractVector{Int}}=nothing,
                diffusivity::Union{Nothing,Real}=nothing,
                offset::Union{Nothing,Real}=nothing)
    L = length(p)
    isnothing(scales) && (scales = ones(L))
    L == length(scales) || throw(ArgumentError(SP_ERROR))

    base = isnothing(offset) ? 4 : 3 # base includes: g0, (offset), κ, τD
    L ≥ base || throw(ArgumentError(NPARAM_ERROR))
    sp = scales .* p

    # dynamics
    m = _ndyn_from_len(L - base)
    isnothing(ics) && (ics = ones(Int, m))
    sum(ics) == m || throw(ArgumentError(PICS_ERROR))
    τ_dyn = m == 0 ? Float64[] : collect(@view(sp[base+1:base+m]))
    K_dyn = m == 0 ? Float64[] : collect(@view(sp[base+1+m:base+2m]))

    g0 = sp[1]
    dyn = (m == 0) ? 1.0 : _dynamics_factor(t, τ_dyn, K_dyn, ics)
    τDe = isnothing(diffusivity) ? sp[base] : τD(diffusivity, sp[base])
    udc = udc_3d(t, τDe, sp[base-1])

    if isnothing(offset)
        @. sp[2] + g0 * udc * dyn
    else
        @. offset + g0 * udc * dyn
    end
end

"""
    fcs_3d_mdiff(t, p; scales, ics, offset)

Mixture of `n` 3D diffusion components sharing the same structure factor `κ`.

*   `p[1]` → g0; the zero-lag autocorrelation
*   `p[2]` → offset; the offset of the correlation from 0
*   `p[3]` → κ; the structure factor `κ = z0/w0`
*   `p[4:n+3]` → τDs; the diffusion times of each diffuser
*   `p[n+4:2n+2]` → weights; the fraction of diffuser in the first n-1 populations.
                    sum of weights constrained by unity so only n-1 dof are required
*   `p[2n+3:2n+2+m]` → τ_dyn; the dynamic lifetimes
*   `p[2n+3+m:end]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime
"""
function fcs_3d_mdiff(t::Union{Real,AbstractVector{<:Real}}, p::AbstractVector{<:Real};
                      n_diff::Integer=1, scales::Union{Nothing,AbstractVector}=nothing,
                      ics::Union{Nothing, AbstractVector{Int}}=nothing,
                      offset::Union{Nothing,Real}=nothing)
    n_diff ≥ 1 || throw(ArgumentError(NDIFF_ERROR))
    L = length(p)
    isnothing(scales) && (scales = ones(L))
    L == length(scales) || throw(ArgumentError(SP_ERROR))

    base = isnothing(offset) ? 3 : 2 # base includes: g0, (offset), κ
    L ≥ base + n_diff || throw(ArgumentError(PNDIFF_ERROR(n_diff)))
    sp = scales .* p

    # diffusion
    τDs = collect(@view sp[base+1:base+n_diff])
    diff_idx = base+2n_diff-1
    
    w_end = base + n_diff + (n_diff > 1 ? (n_diff-1) : 0)
    wts = (n_diff == 1) ? Float64[] : collect(@view sp[base+n_diff+1 : w_end])
    (n_diff == 1 || sum(wts) ≤ 1) || throw(ArgumentError(WEIGHTS_ERROR))

    # dynamics
    m = _ndyn_from_len(L - w_end)
    isnothing(ics) && (ics = ones(Int, m))
    sum(ics) == m || throw(ArgumentError(PICS_ERROR))
    τdyn = m == 0 ? Float64[] : collect(@view sp[w_end+1:w_end+m])
    Kdyn = m == 0 ? Float64[] : collect(@view sp[w_end+m+1:w_end+2m])

    g0 = sp[1]
    κ = sp[base]
    dyn = (m == 0) ? 1.0 : _dynamics_factor(t, τdyn, Kdyn, ics)
    mix = _mdiff(t, τDs, wts, (tt,τ)->udc_3d(tt, τ, κ))

    if isnothing(offset)
        @. sp[2] + g0 * mix * dyn
    else
        @. offset + g0 * mix * dyn
    end
end