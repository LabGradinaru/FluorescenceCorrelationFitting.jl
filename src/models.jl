const BOLTZMANN = 1.380649e-23 # Boltzmann constant in SI units
const AVAGADROS = 6.022141e23 # Avagadro's number in SI units


# ─────────────────────────────────────────────────────────────────────────────
# Convenience calculators
# ─────────────────────────────────────────────────────────────────────────────

const w0_ERROR = "w0 must be positive"
const κ_ERROR = "κ must be positive"
const D_ERROR = "Diffusivity must be positive"

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
    surface_density(w0, g0; Ks=[], ics=[0])

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
# Low-level kernel helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    dynamics_factor(t, τs, Ks, ics)

Compute the contribution of exponential-kernel dynamics to the correlation.

- `τs` and `Ks` have equal length, with all `0 ≤ K < 1`.
- `ics` lists the component counts per block (sum(ics) == length(τs)).
- `t` may be a scalar `Number` or an `AbstractVector`.

Returns a scalar if `t` is a scalar, or a vector of the same length as `t` otherwise.
"""
function dynamics_factor(t, τs::AbstractVector, Ks::AbstractVector, ics::AbstractVector{<:Integer})
    length(τs) == length(Ks) || throw(ArgumentError("τs and Ks must have same length."))
    sum(ics) == length(τs) || throw(ArgumentError("The number of components (sum(ics)) must match length(τs)=length(Ks)."))
    all(0 .<= Ks .< 1) || throw(ArgumentError("All Ks must lie in [0, 1)."))

    T = promote_type(_eltype(t), _eltype(τs), _eltype(Ks))
    return _dynamics_factor(t, τs, Ks, ics, T)
end

function _dynamics_factor(t::Number, τs::AbstractVector, Ks::AbstractVector, 
                          ics::AbstractVector{<:Integer}, T)
    isempty(τs) && return one(T)  
    out = one(T)
    idx = 1
    @inbounds for nb in ics
        s = zero(T)
        @inbounds for j = 1:nb
            k = idx + j - 1
            τ = τs[k]; K = Ks[k]
            s += T(dyn(t, τ, K))
        end
        out *= (one(T) + s)
        idx += nb
    end
    return out
end

function _dynamics_factor(t::AbstractVector, τs::AbstractVector, Ks::AbstractVector, 
                          ics::AbstractVector{<:Integer}, T)
    isempty(τs) && return ones(T, length(t))

    out = ones(T, length(t))
    s = similar(out);  fill!(s, zero(T))  # work buffer

    idx = 1
    @inbounds for nb in ics
        fill!(s, zero(T))
        @inbounds for j = 1:nb
            k = idx + j - 1
            τ = τs[k]; K = Ks[k]
            @simd for n in eachindex(t)
                s[n] += T(dyn(t[n], τ, K))
            end
        end
        @inbounds for n in eachindex(out)
            out[n] *= (one(T) + s[n])
        end
        idx += nb
    end
    return out
end

"""
    diff_factor(t, κ, τs, αs, wts)

Compute the contribution of diffusion-kernel dynamics to the correlation.
"""
function diff_factor(t, κ::Union{Nothing,Real}, τs::AbstractVector, 
                     αs::Union{Nothing,AbstractVector}, wts::AbstractVector)
    n = length(τs)
    if !isnothing(αs)
        length(αs) == n || throw(ArgumentError("There must be as many diffusion times as there are anomalous exponents."))
    end
    length(wts) + 1 == n || throw(ArgumentError("There must be one less weight than there are diffusion times."))

    if n == 1
        w_full = (1.0,)
    else
        sum(wts) ≤ 1 || throw(ArgumentError("Sum of diffusion population weights must be ≤ 1"))
        w_full = (vcat(wts, 1 - sum(wts)))::Vector{Float64}
    end

    T = promote_type(_eltype(t), _eltype(τs), _eltype(wts))
    return _diff_factor(t, κ, τs, αs, w_full, T)
end

function _diff_factor(t::Number, κ::Union{Nothing,Real}, τs::AbstractVector, 
                      αs::Union{Nothing,AbstractVector}, wts::AbstractVector, T)
    out = zero(T)
    @inbounds for i in eachindex(τs)
        α = isnothing(αs) ? nothing : αs[i]
        out += T(wts[i] * _diff(t, κ, τs[i], α))
    end
    return out
end

function _diff_factor(t::AbstractVector, κ::Union{Nothing,Real}, τs::AbstractVector, 
                      αs::Union{Nothing,AbstractVector}, wts::AbstractVector, T)
    out = zeros(T, length(t))
    @inbounds for i in eachindex(τs)
        α = isnothing(αs) ? nothing : αs[i]
        τ = τs[i]; wt = wts[i]
        @simd for n in eachindex(t)
            out[n] += T(wt * _diff(t[n], κ, τ, α))
        end
    end
    return out
end

# determine the number of dynamic components based on the parameter vector length
@inline function _ndyn_from_len(total_extra::Int)
    total_extra ≥ 0 || throw(ArgumentError("p too short."))
    rem(total_extra, 2) == 0 || throw(ArgumentError("τ_dyn and K_dyn must have the same length."))
    total_extra ÷ 2
end

_eltype(x) = x isa AbstractArray ? eltype(x) : typeof(x)

# ─────────────────────────────────────────────────────────────────────────────
# Base kernels (unit-amplitude, zero-offset)
# ─────────────────────────────────────────────────────────────────────────────

"""Handler for diffusion kernels based on which arguments are not Nothing."""
@inline function _diff(t, κ, τ, α)
    if isnothing(κ)
        if isnothing(α)
            udc_2d(t, τ)
        else
            udc_2d(t, τ, α)
        end
    else
        if isnothing(α)
            udc_3d(t, τ, κ)
        else
            udc_3d(t, τ, κ, α)
        end
    end
end

"2D diffusion kernel"
udc_2d(t, τ) = @. inv(1 + t/τ) 

"2D anomalous diffusion kernel with exponent α"
udc_2d(t, τ, α) = @. inv(1 + (t/τ)^α)

"3D diffusion kernel with structure factor κ = z0/w0"
udc_3d(t, τ, κ) = @. inv((1 + t/τ) * sqrt(1 + t/(κ^2 * τ)))

"3D anomalous diffusion kernel with anomalous exponent α and structure factor κ = z0/w0"
udc_3d(t, τ, κ, α) = @. inv((1 + (t/τ)^α) * sqrt(1 + (t/τ)^α / κ^2))

"Kernel of a dynamic species of fraction `f` and lifetime `τ`."
dyn(t, τ, f) = @. f * (exp(-t/τ) - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Public fitting models and types
# ─────────────────────────────────────────────────────────────────────────────

struct Dimension
    sym::Symbol
    function Dimension(sym::Symbol)
        sym ∈ (:d2, :d3) || throw(DomainError(sym, "Dimension must be :d2 or :d3"))
        new(sym)
    end
end
Dimension(x::AbstractString) = Dimension(Symbol(lowercase(x)))
Dimension(x::Dimension) = x
Base.convert(::Type{Dimension}, x) = Dimension(x) 

struct Scope
    sym::Symbol
    function Scope(sym::Symbol)
        sym ∈ (:none, :global, :perpop) || throw(DomainError(sym, "Scope must be :none, :global, or :perpop"))
        new(sym)
    end
end
Scope(x::AbstractString) = Scope(Symbol(lowercase(x)))
Scope(x::Scope) = x
Base.convert(::Type{Scope}, x) = Scope(x)

# TODO: would be better user experience to have a named tuple wrapper around this that parses into the Spec

Base.@kwdef struct FCSModelSpec
    dim::Dimension
    anom::Scope
    offset::Union{Nothing,Float64} = nothing
    diffusivity::Union{Nothing,Float64} = nothing
    n_diff::Int = 1
    ics::Union{Nothing,Vector{Int}} = nothing
end
function FCSModelSpec(dim, anom, offset, diffusivity, n_diff, ics::Union{Nothing,AbstractVector})
    ndim = convert(Dimension, dim)
    nanom = convert(Scope, anom)

    noff = isnothing(offset) ? nothing : Float64(offset)
    nD = isnothing(diffusivity) ? nothing : Float64(diffusivity)

    nndiff = Int(n_diff)
    nndiff ≥ 1 || throw(ArgumentError("n_diff must be ≥ 1"))

    nics = nothing
    if !isnothing(ics)
        nics = Int.(collect(ics))
        any(<(0), nics) && throw(ArgumentError("ics entries must be ≥ 0"))
    end

    if (nanom.sym === :perpop && nndiff == 1)
        throw(ArgumentError("Anom spec scope :perpop requires n_diff > 1"))
    end

    return FCSModelSpec(; dim=ndim, anom=nanom, offset=noff, diffusivity=nD, n_diff=nndiff, ics=nics)
end

Base.@kwdef mutable struct FCSModel <: Function
    spec::FCSModelSpec
    scales::Union{Nothing,AbstractVector} = nothing
end
(m::FCSModel)(t, p) = _eval(m.spec, t, p; scales=m.scales)

"""
    _eval(spec, t, p; scales=nothing)


"""
function _eval(spec::FCSModelSpec, t, p::AbstractVector; scales=nothing)
    L = length(p)
    isnothing(scales) && (scales = ones(L))
    L == length(scales) || throw(ArgumentError("Scaling and parameter vectors must be of the same length."))
    sp = scales .* p # scaled parameter vector
    
    # current correlation (at lag time = 0)
    g0 = sp[1]

    # check if the offset is to be allowed to vary in the model
    # if it is, set the current parameter vector to be `off`
    idx = 2
    off = isnothing(spec.offset) ? (sp[idx]; idx += 1; sp[idx-1]) : spec.offset

    # if the model is in three dimensions, collect the structure factor κ
    κ = nothing
    if spec.dim.sym == :d3
        κ = sp[idx]; idx += 1
    end

    # diffusion times
    # if diffusivity is set to a value, the parameters in these 
    # slots are assumed to be corresponding to w0
    n_diff = spec.n_diff
    τD_end = idx + n_diff - 1
    L ≥ τD_end || throw(ArgumentError("Parameter vector too short for $n_diff diffusion times."))
    τDslots = @view sp[idx:τD_end]
    τDvec = isnothing(spec.diffusivity) ? collect(τDslots) : [τD(spec.diffusivity, w0) for w0 in τDslots]

    # anomalous exponents
    α = nothing
    α_scope = spec.anom.sym
    if α_scope == :global
        α = sp[idx] * ones(n_diff); idx += 1
    elseif α_scope == :perpop
        α_end = idx + n_diff - 1
        L ≥ α_end || throw(ArgumentError("Parameter vector too short for $n_diff unique anomalous exponents."))
        α = collect(@view sp[idx:α_end])
        idx += n_diff
    end

    # diffusion population weights
    #TODO: in the future, weights being bounds by unity should be imposed as a non-linear constraint instead
    if n_diff == 1
        wts = Float64[]
    else
        w_end = idx + n_diff - 2
        L ≥ w_end || throw(ArgumentError("Parameter vector too short for $(n_diff-1) population weights."))
        wts = collect(@view sp[idx:w_end])
        sum(wts) ≤ 1 || throw(ArgumentError("Sum of diffusion population weights must be ≤ 1"))
        idx = w_end + 1
    end

    # dynamic contributions to the correlation
    # TODO: as above. naively, this seems much more challenging since it is based on ics within FCSModelSpec
    m = _ndyn_from_len(L - w_end)
    ics = something(spec.ics, ones(Int, m))
    sum(ics) == m || throw(ArgumentError("Mismatch between dynamics expected in the parameter vector and the independent components."))
    
    τdyn = m == 0 ? Float64[] : collect(@view sp[idx:idx+m-1])
    Kdyn = m == 0 ? Float64[] : collect(@view sp[idx+m:idx+2m-1])

    dyn = m == 0 ? 1.0 : dynamics_factor(t, τdyn, Kdyn, ics)
    diff = diff_factor(t, κ, τDvec, α, wts)

    return @. off + g0 * diff * dyn
end