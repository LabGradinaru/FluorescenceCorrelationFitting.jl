@enum Dim::UInt8 d2 d3 # Spatial dimension
@enum Scope::UInt8 none globe perpop # Scope for a variable

"""
    FCSModelSpec{D, S, OFF, DIFF, RAD, ND}(offset, diffusivity, ics)
    FCSModelSpec(; dim, anom, offset, diffusivity, width, n_diff, ics)
    FCSModelSpec(nt)
    FCSModelSpec(d)

Construct an object which specifies the model for fitting `FCSModelSpec`. 
The `D` parameter is the spatial dimension, `Dim`, which the diffusion occurs in
and dictates the diffusion kernel which is used. The `S<:Scope` parameter dictates 
the presence and scope of the anomalous exponent. `OFF`, `DIFF` and `RAD` are Boolean 
types that determine if offset, diffusion and beam width, respectively, are allowed to vary during fitting. 
`ND` is a type of the form `Val{N}` for an integer `N` corresponding to the number of diffusive components.

# Examples
```julia
spec = FCSModelSpec(; dim=d2, anom=none, n_diff=1)  # 2D, normal diffusion, one diffuser
spec = FCSModelSpec(; dim=d3, anom=globe, n_diff=2)  # 3D, one α shared across 2 diffusers
spec = FCSModelSpec(; dim=d3, anom=perpop, n_diff=2, offset=0.0) # α₁,α₂; fixed offset
spec = FCSModelSpec(; dim=d3, diffusivity=5e-11, n_diff=1)  # treat τD slot as w0
```
"""
struct FCSModelSpec{D,S,OFF,DIFF,RAD,ND}
    offset::Float64
    diffusivity::Float64
    beamwidth::Float64
    ics::Vector{Int}
end

function FCSModelSpec(; dim::Dim=d3, anom::Scope=none,
                      offset::Union{Nothing,Real}=nothing,
                      diffusivity::Union{Nothing,Real}=nothing,
                      width::Union{Nothing,Real}=nothing,
                      n_diff::Integer=1, ics=nothing)
    OFF = offset !== nothing
    DIFF = diffusivity !== nothing
    RAD = width !== nothing
    
    N = Int(n_diff)
    N ≥ 1 || throw(ArgumentError("n_diff must be ≥ 1"))

    Dval = _dim_from(dim)
    Sval = _scope_from(anom)
    FCSModelSpec{Dval,Sval,OFF,DIFF,RAD,Val{N}}(
        OFF ? float(offset) : 0.0,
        DIFF ? float(diffusivity) : 0.0,
        RAD ? float(width) : 0.0,
        ics === nothing ? Int[] : Int.(ics)
    )
end

FCSModelSpec(nt::NamedTuple) = FCSModelSpec(; nt...)
FCSModelSpec(d::Dict) = FCSModelSpec(; (Symbol(k)=>v for (k,v) in d)...)

# utilities to access the type-encoded flags/values
dim(::FCSModelSpec{D}) where {D} = D
anom(::FCSModelSpec{D,S}) where {D,S} = S
hasoffset(::FCSModelSpec{D,S,OFF}) where {D,S,OFF} = OFF
hasdiffusivity(::FCSModelSpec{D,S,OFF,DIFF}) where {D,S,OFF,DIFF} = DIFF
haswidth(::FCSModelSpec{D,S,OFF,DIFF,RAD}) where {D,S,OFF,DIFF,RAD} = RAD
n_diff(::FCSModelSpec{D,S,OFF,DIFF,RAD,Val{N}}) where {D,S,OFF,DIFF,RAD,N} = N

# convenience methods for translating Symbols/ Strings to enums
_scope_from(x) = x isa Scope ? x :
                 x === :global ? globe :
                 x isa Symbol ? Scope(x) :
                 Scope(Symbol(x))
_dim_from(x) = x isa Dim ? x : Dim(Symbol(x))



const DYN_COMP_ERROR = ArgumentError("Mismatch between dynamics expected in the parameter vector and the independent components.")

"""
    ParamLayout(i_g0, i_off, i_kappa, i_tauD, i_alpha, i_wts, i_taudyn, i_Kdyn)
    ParamLayout(spec, scales, params)

Container for ranges within which given parameters are stored in a parameter vector.
"""
struct ParamLayout
    i_g0::Int
    i_off::Int # 0 if fixed
    i_κ::Int # 0 if 2D
    i_τD::UnitRange{Int} # empty if both diffusivity and w0 are fixed
    i_α::UnitRange{Int} # empty if none
    i_wts::UnitRange{Int} # empty if n==1
    i_τdyn::UnitRange{Int}
    i_Kdyn::UnitRange{Int}
end

function ParamLayout(spec::FCSModelSpec, scales::AbstractVector, params::AbstractVector{T}) where {T}
    L = length(params)
    length(scales) === L || throw(ArgumentError("Scaling and parameter vectors must be of the same length."))
    
    # current amplitude, offset and structure factor handling
    i_g0 = 1

    i_off = 0
    idx = 2
    if !hasoffset(spec)
        i_off = idx
        idx += 1
    end

    i_kappa = 0
    if dim(spec) === d3 
        i_kappa = idx
        idx += 1
    end

    # indices corresponding to diffusion times
    # if both the diffusivity and width are fixed, diffusion time is completely specified
    i_τD = 1:0 # empty UnitRange
    n = n_diff(spec)
    if !hasdiffusivity(spec) || !haswidth(spec)
        τD_end = idx + n - 1
        L ≥ τD_end || throw(ArgumentError("Parameter vector too short for $n diffusion times."))
        i_τD = idx:τD_end
        idx += n
    end

    # indices for anomalous diffusion factors 
    i_α = 1:0 
    α_scope = anom(spec)
    if α_scope == globe
        i_α = idx:idx
        idx += 1
    elseif α_scope == perpop
        α_end = idx + n - 1
        L ≥ α_end || throw(ArgumentError("Parameter vector too short for $n unique anomalous exponents."))
        i_α = idx:α_end
        idx += n
    end

    # indices for diffusive population weights
    i_wts = 1:0
    if n > 1
        w_end = idx + n - 2
        L ≥ w_end || throw(ArgumentError("Parameter vector too short for $(n-1) population weights."))
        i_wts = idx:w_end
        idx = w_end + 1
    end
    
    # indices for dynamic lifetimes and weights
    m = _ndyn_from_len(L - (idx-1))
    ics = isempty(spec.ics) ? ones(Int, m) : spec.ics
    sum(ics) == m || throw(DYN_COMP_ERROR)
    i_τdyn = idx:idx+m-1
    i_Kdyn = idx+m:idx+2m-1

    return ParamLayout(i_g0, i_off, i_kappa, i_τD, i_α, i_wts, i_τdyn, i_Kdyn)
end


"""
    EvalCache{T}(sp, τD, α, wfull, τdyn, Kdyn)

Container for per-evaluation cache buffers to minimize allocations during fitting.
"""
mutable struct EvalCache{T}
    sp::Vector{T}
    τD::Vector{T}
    α::Vector{T}
    wts::Vector{T}
    τdyn::Vector{T}
    Kdyn::Vector{T}
end



"""
    FCSModel{T,SpecT,ScaleT}(spec, layout, scales, ics, cache, dynbuf, work)
    FCSModel(spec, t, p0; scales)

Complete model specification, parameter vector structure and cache required 
for minimal-allocation evaluation. Can be easily created from an FCSModelSpec.
Acts as a functor with generic optimization problem inputs (independent, dependent).
"""
mutable struct FCSModel{T,SpecT,ScaleT} <: Function
    spec::SpecT
    layout::ParamLayout
    scales::ScaleT
    ics::Vector{Int}

    # cache and work buffers for use during evaluation
    cache::EvalCache{T}
    work::Vector{T}
    dynbuf::Vector{T}
end

function FCSModel(spec::FCSModelSpec, t::AbstractVector, p0::AbstractVector; scales=nothing)
    T = promote_type(eltype(t), eltype(p0), Float64)
    L = length(p0)

    scalesT = isnothing(scales) ? nothing : T.(scales)
    layout = ParamLayout(spec, isnothing(scalesT) ? ones(T,L) : scalesT, T.(p0))

    n = n_diff(spec)
    m = length(layout.i_τdyn)

    cache = EvalCache{T}(
        zeros(T, L),
        zeros(T, n),
        anom(spec) === none ? T[] : zeros(T, n),
        n <= 1 ? T[] : zeros(T, n-1),
        zeros(T, m), zeros(T, m)
    )

    ics = isempty(spec.ics) ? ones(Int, m) : Int.(spec.ics)
    return FCSModel(spec, layout, scalesT, ics, cache, zeros(T, length(t)), zeros(T, length(t)))
end

function (m::FCSModel)(t::AbstractVector, p::AbstractVector)
    y = similar(t, promote_type(eltype(t), eltype(p), eltype(m.cache.sp)))
    return eval!(y, m, t, p)
end

@inline function update!(m::FCSModel{T,SpecT}, p::AbstractVector) where {T,SpecT}
    sp = m.cache.sp
    scales = m.scales

    if isnothing(scales)
        @inbounds @simd for i in eachindex(p)
            sp[i] = T(p[i])
        end
    else
        @inbounds @simd for i in eachindex(p)
            sp[i] = T(p[i]) * scales[i]
        end
    end

    # τD: either copy raw τD or compute τD(D, w0), depending on which parameters are fixed
    if hasdiffusivity(m.spec)
        Dfix = m.spec.diffusivity
        if haswidth(m.spec)
            w0fix = m.spec.beamwidth
            fill!(m.cache.τD, τD(Dfix, w0fix))
        else
            @inbounds for (j, i) in enumerate(m.layout.i_τD)
                w0 = sp[i]
                m.cache.τD[j] = τD(Dfix, w0)
            end
        end
    else
        if haswidth(m.spec)
            w0fix = m.spec.beamwidth
            @inbounds for (j, i) in enumerate(m.layout.i_τD)
                diff = sp[i]
                m.cache.τD[j] = τD(diff, w0fix)
            end
        else
            @inbounds for (j, i) in enumerate(m.layout.i_τD)
                m.cache.τD[j] = sp[i]
            end
        end
    end

    # α: expand to length n if present
    if anom(m.spec) === none
        # leave α empty
    elseif anom(m.spec) === globe
        a = sp[first(m.layout.i_α)]
        fill!(m.cache.α, a)
    else # perpop
        @inbounds for (j, i) in enumerate(m.layout.i_α)
            m.cache.α[j] = sp[i]
        end
    end

    # weights (n-1) possibly empty
    if !isempty(m.layout.i_wts)
        sum(m.cache.wts) ≤ 1 || throw(ArgumentError("Sum of diffusion population weights must be ≤ 1"))
        @inbounds for (j, i) in enumerate(m.layout.i_wts)
            m.cache.wts[j] = sp[i]
        end
    end

    # dynamics possibly empty
    if !isempty(m.layout.i_τdyn)
        @inbounds for (j, i) in enumerate(m.layout.i_τdyn)
            m.cache.τdyn[j] = sp[i]
        end
        @inbounds for (j, i) in enumerate(m.layout.i_Kdyn)
            m.cache.Kdyn[j] = sp[i]
        end
    end

    return nothing
end

function eval!(y::AbstractVector, m::FCSModel, t::AbstractVector, p::AbstractVector)
    update!(m, p)

    sp = m.cache.sp
    g0 = sp[m.layout.i_g0]
    off = m.layout.i_off == 0 ? m.spec.offset : sp[m.layout.i_off]
    κ = m.layout.i_κ  == 0 ? nothing : sp[m.layout.i_κ]

    diff_factor!(y, t, κ, m.cache.τD, m.cache.α, m.cache.wts)
    dynamics_factor!(m.dynbuf, m.work, t, m.cache.τdyn, m.cache.Kdyn, m.ics)

    @inbounds @simd for i in eachindex(t)
        y[i] = off + g0 * y[i] * m.dynbuf[i]
    end
    return y
end

# determine the number of dynamic components based on the parameter vector length
@inline function _ndyn_from_len(total_extra::Int)
    total_extra ≥ 0 || throw(ArgumentError("Parameter vector too short."))
    rem(total_extra, 2) == 0 || throw(ArgumentError("Dynamic lifetimes and fractions vectors must have the same length."))
    total_extra ÷ 2
end



# ─────────────────────────────────────────────────────────────────────────────
# Low-level kernel helpers
# ─────────────────────────────────────────────────────────────────────────────

function diff_factor!(out, t, κ, τs, αs, wts)
    T = eltype(out)
    fill!(out, zero(T))

    n = length(τs)
    wsum = zero(T)

    # quick return if there is only one dynamic component
    if n == 1
        τ = τs[1]
        @inbounds @simd for i in eachindex(t)
            out[i] = _diff(t[i], κ, τ, isempty(αs) ? nothing : αs[1])
        end
        return out
    end

    @inbounds for j = 1:n-1
        w = wts[j]
        wsum += w
        τ = τs[j]
        α = isempty(αs) ? nothing : αs[j]
        @simd for i in eachindex(t)
            out[i] += w * _diff(t[i], κ, τ, α)
        end
    end

    lastw = one(T) - wsum
    τ = τs[n]
    α = isempty(αs) ? nothing : αs[n]
    @inbounds @simd for i in eachindex(t)
        out[i] += lastw * _diff(t[i], κ, τ, α)
    end
    return out
end

function diff_factor(t, κ, τs, αs, wts)
    out = similar(t, eltype(t))
    diff_factor!(out, t, κ, τs, αs, wts)
end

function dynamics_factor!(out, work, t, τs, Ks, ics)
    if isempty(τs)
        fill!(out, one(eltype(out)))
        return out
    end

    fill!(out, one(eltype(out)))
    idx = 1
    # loop over each component of ICS (number of dependent dynamics factors)
    @inbounds for nb in ics
        fill!(work, zero(eltype(work)))
        for j ∈ 1:nb
            k = idx + j - 1
            τ = τs[k];  K = Ks[k] # fraction and lifetime for dynamic component
            @simd for n in eachindex(t)
                work[n] += dyn(t[n], τ, K) # Kᵢ (e^{-t/τᵢ} - 1)
            end
        end
        @simd for n in eachindex(t)
            out[n] *= (one(eltype(out)) + work[n]) # 1 + ∑ᵢ Kᵢ (e^{-t/τᵢ} - 1)
        end
        idx += nb
    end
    return out
end

function dynamics_factor(t, τs, Ks, ics)
    out = similar(t, eltype(t))
    work = similar(t, eltype(t))
    dynamics_factor!(out, work, t, τs, Ks, ics)
end


# ─────────────────────────────────────────────────────────────────────────────
# Global (multi-channel) model builder
# ─────────────────────────────────────────────────────────────────────────────

"""
    _build_global_model(channel_models, τs, n_shared, local_lengths,
                        i_shared_in_full, i_local_in_full) -> Function

Build a callable compatible with `LsqFit.curve_fit` for simultaneous fitting
of multiple FCS channels with shared ("global") parameters.

The combined parameter vector has the layout
`[shared_params, local_params_ch1, local_params_ch2, ...]`.
For each channel `k`, the shared params are inserted at the indices
`i_shared_in_full[k]` and the local params at `i_local_in_full[k]`
before the channel's own `FCSModel` is evaluated. The returned predictions
are concatenated in channel order.
"""
function _build_global_model(channel_models,
                              τs::Vector{Vector{Float64}},
                              n_shared::Int,
                              local_lengths::Vector{Int},
                              i_shared_in_full::Vector{Vector{Int}},
                              i_local_in_full::Vector{Vector{Int}})
    K = length(channel_models)
    # Precompute start-of-local-block offset in combined_p for each channel
    local_offsets = Vector{Int}(undef, K)
    local_offsets[1] = n_shared
    for k in 2:K
        local_offsets[k] = local_offsets[k-1] + local_lengths[k-1]
    end

    return function (x::AbstractVector, combined_p::AbstractVector)
        T = eltype(combined_p)
        shared_p = view(combined_p, 1:n_shared)
        results  = Vector{Vector{T}}(undef, K)

        for k in 1:K
            L_local = local_lengths[k]
            loff = local_offsets[k]
            local_p = view(combined_p, loff+1:loff+L_local)

            n_full = n_shared + L_local
            full_p = Vector{T}(undef, n_full)
            @inbounds for (j, i) in enumerate(i_shared_in_full[k])
                full_p[i] = shared_p[j]
            end
            @inbounds for (j, i) in enumerate(i_local_in_full[k])
                full_p[i] = local_p[j]
            end

            results[k] = channel_models[k](τs[k], full_p)
        end

        return vcat(results...)
    end
end



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
dyn(t, τ, f) = f * (exp(-t/τ) - 1)