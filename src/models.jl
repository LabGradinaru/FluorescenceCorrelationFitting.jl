@enum Dim::UInt8 d2 d3 # Spatial dimension
@enum Scope::UInt8 none globe perpop # Scope for a variable

"""
    FCSModelSpec{D, S, OFF, DIFF, NDIFF}(offset, diffusivity, ics)
    FCSModelSpec(; dim, anom, offset, diffusivity, n_diff, ics)
    FCSModelSpec(nt)
    FCSModelSpec(d)

Construct an object which specifies the model for fitting `FCSModelSpec`. 
The `D` parameter is the spatial dimension, `Dim`, which the diffusion occurs in
and dictates the diffusion kernel which is used. The `S<:Scope` parameter dictates 
the presence and scope of the anomalous exponent. `OFF` and `DIFF` determine 
if offset and diffusion, respectively, are allowed to vary during fitting. 
`NDIFF` a type corresponding to the number of diffusive components.

# Examples
```julia
spec = FCSModelSpec(; dim=d2, anom=none, n_diff=1)  # 2D, normal diffusion, one diffuser
spec = FCSModelSpec(; dim=d3, anom=global, n_diff=2)  # 3D, one α shared across 2 diffusers
spec = FCSModelSpec(; dim=d3, anom=perpop, n_diff=2, offset=0.0) # α₁,α₂; fixed offset
spec = FCSModelSpec(; dim=d3, diffusivity=5e-11, n_diff=1)  # treat τD slot as w0
```
"""
struct FCSModelSpec{D,S,OFF,DIFF,NDIFF}
    offset::Float64
    diffusivity::Float64
    ics::Vector{Int}
end

# utilities to access the type-encoded flags/values
dim(::FCSModelSpec{D}) where {D} = D
anom(::FCSModelSpec{D,S}) where {D,S} = S
hasoffset(::FCSModelSpec{D,S,OFF}) where {D,S,OFF} = OFF
hasdiffusivity(::FCSModelSpec{D,S,OFF,DIFF}) where {D,S,OFF,DIFF} = DIFF
n_diff(::FCSModelSpec{D,S,OFF,DIFF,NDIFF}) where {D,S,OFF,DIFF,NDIFF} = NDIFF

# convenience methods for translating Symbols/ Strings to enums
_scope_from(x) = x isa Scope ? x :
                 x === :global ? globe :
                 x isa Symbol ? Scope(x) :
                 Scope(Symbol(x))
_dim_from(x) = x isa Dim ? x : Dim(Symbol(x))

function FCSModelSpec(; dim::Dim=d3, anom::Scope=none,
                      offset::Union{Nothing,Real}=nothing,
                      diffusivity::Union{Nothing,Real}=nothing,
                      n_diff::Integer=1, ics=nothing)
    OFF = offset !== nothing
    DIFF = diffusivity !== nothing
    
    N = Int(n_diff)
    N ≥ 1 || throw(ArgumentError("n_diff must be ≥ 1"))

    Dval = _dim_from(dim)
    Sval = _scope_from(anom)
    FCSModelSpec{Dval,Sval,OFF,DIFF,N}(
        offset === nothing ? 0.0 : float(offset),
        diffusivity === nothing ? 0.0 : float(diffusivity),
        ics === nothing ? Int[] : Int.(ics)
    )
end

FCSModelSpec(nt::NamedTuple) = FCSModelSpec(; nt...)
FCSModelSpec(d::Dict) = FCSModelSpec(; (Symbol(k)=>v for (k,v) in d)...)


Base.@kwdef mutable struct FCSModel <: Function
    spec::FCSModelSpec
    scales::Union{Nothing,AbstractVector} = nothing
end
(m::FCSModel)(t, p) = _eval(m.spec, t, p; scales=m.scales)


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
    off = hasoffset(spec) ? spec.offset : (sp[idx]; idx += 1; sp[idx-1])

    # if the model is in three dimensions, collect the structure factor κ
    κ = dim(spec) === d3 ? (sp[idx]; idx += 1; sp[idx-1]) : nothing

    # diffusion times
    # if diffusivity is set to a value, the parameters in these 
    # slots are assumed to be corresponding to w0
    n = n_diff(spec)
    τD_end = idx + n - 1
    L ≥ τD_end || throw(ArgumentError("Parameter vector too short for $n diffusion times."))
    τDslots = @view sp[idx:τD_end]
    τDvec = hasdiffusivity(spec) ? [τD(spec.diffusivity, w0) for w0 in τDslots] : collect(τDslots) 
    idx += n

    # anomalous exponents
    α = nothing
    α_scope = anom(spec)
    if α_scope == globe
        α = sp[idx] * ones(n); idx += 1
    elseif α_scope == perpop
        α_end = idx + n - 1
        L ≥ α_end || throw(ArgumentError("Parameter vector too short for $n unique anomalous exponents."))
        α = collect(@view sp[idx:α_end])
        idx += n
    end

    # diffusion population weights
    #TODO: in the future, weights being bounds by unity should be imposed as a non-linear constraint instead
    if n == 1
        wts = Float64[]
    else
        w_end = idx + n - 2
        L ≥ w_end || throw(ArgumentError("Parameter vector too short for $(n-1) population weights."))
        wts = collect(@view sp[idx:w_end])
        sum(wts) ≤ 1 || throw(ArgumentError("Sum of diffusion population weights must be ≤ 1"))
        idx = w_end + 1
    end

    # dynamic contributions to the correlation
    # TODO: as above. naively, this seems much more challenging however, since it is based on ics within FCSModelSpec
    m = _ndyn_from_len(L - (idx - 1))
    ics = isempty(spec.ics) ? ones(Int, m) : spec.ics
    sum(ics) == m || throw(ArgumentError("Mismatch between dynamics expected in the parameter vector and the independent components."))
    
    τdyn = m == 0 ? Float64[] : collect(@view sp[idx:idx+m-1])
    Kdyn = m == 0 ? Float64[] : collect(@view sp[idx+m:idx+2m-1])

    dyn = m == 0 ? 1.0 : dynamics_factor(t, τdyn, Kdyn, ics)
    diff = diff_factor(t, κ, τDvec, α, wts)

    return @. off + g0 * diff * dyn
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
        w_full = ones(1)
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
    total_extra ≥ 0 || throw(ArgumentError("Parameter vector too short."))
    rem(total_extra, 2) == 0 || throw(ArgumentError("Dynamic lifetimes and fractions vectors must have the same length."))
    total_extra ÷ 2
end

_eltype(x) = x isa AbstractArray ? eltype(x) : typeof(x)


# ─────────────────────────────────────────────────────────────────────────────
# Base kernels (unit-amplitude, zero-offset)
# ─────────────────────────────────────────────────────────────────────────────

"""Handler for diffusion kernels based on which arguments are not Nothing."""
@inline function _diff(t, κ, τ, α)
    if isnothing(κ)
        isnothing(α) ? udc_2d(t, τ) : udc_2d(t, τ, α)
    else
        isnothing(α) ? udc_3d(t, τ, κ) : udc_3d(t, τ, κ, α)
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