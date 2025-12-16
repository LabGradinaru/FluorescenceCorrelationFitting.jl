PosError(x) = ArgumentError(string(x, " must be positive."))
const w0_SIGN_ERROR = PosError("w0")
const κ_SIGN_ERROR = PosError("κ")
const D_SIGN_ERROR = PosError("Diffusivity")
const nd_ERROR = ArgumentError("The chosen diffuser must be at a positive index less than the total number of diffusers.")
const w0_REQUIRED_ERROR = ArgumentError("This model does not fix diffusivity, so you must provide `w0=` to compute D.")



"""
    τD(D, w0; scale="")
    τD(spec, fit; nd=1, scale="")

Convert diffusion coefficient `D` and lateral waist `w0` to the lateral diffusion time τD.

# Keyword arguments
- `nd::Int`: If given a fit, refers to the diffusive population (if there are many) 
             which the diffusivity is to be calculated for.
- `scale::String`: The metric prefix by which to scale metres (e.g., `scale = μ` -> [τD] = μs)
"""
function τD end

function τD(D::Real, w0::Real; scale::String="")
    w0 > 0 || throw(w0_SIGN_ERROR)
    D > 0 || throw(D_SIGN_ERROR)
    
    diff_time = w0^2 / (4D)
    haskey(SI_PREFIXES, scale) && (diff_time *= SI_PREFIXES[scale])
    return diff_time
end

function τD(spec::FCSModelSpec{D,S,OFF,false,false}, fit::FCSFitResult; 
            nd::Int = 1, scale::String="") where {D,S,OFF}
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)

    idx = 1
    !hasoffset(spec) && (idx += 1)
    dim(spec) === d3 && (idx += 1)

    diff_time = coef(fit)[idx + nd]
    haskey(SI_PREFIXES, scale) && (diff_time *= SI_PREFIXES[scale])
    return diff_time
end

function τD(spec::FCSModelSpec{D,S,OFF,true,false}, fit::FCSFitResult; 
            nd::Int = 1, kwargs...) where {D,S,OFF}
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)

    idx = 1
    !hasoffset(spec) && (idx += 1)
    dim(spec) === d3 && (idx += 1)

    w0 = coef(fit)[idx + nd]
    return τD(spec.diffusivity, w0; kwargs...)
end

function τD(spec::FCSModelSpec{D,S,OFF,false,true}, fit::FCSFitResult; 
            nd::Int = 1, kwargs...) where {D,S,OFF}
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)

    idx = 1
    !hasoffset(spec) && (idx += 1)
    dim(spec) === d3 && (idx += 1)

    diff = coef(fit)[idx + nd]
    return τD(diff, spec.beamwidth; kwargs...)
end

function τD(spec::FCSModelSpec{D,S,OFF,true,true}, fit::FCSFitResult; 
            kwargs...) where {D,S,OFF}
    return τD(spec.diffusivity, spec.beamwidth; kwargs...)
end


"""
    diffusivity(τD, w0; scale="")
    diffusivity(spec, fit; scale="")
    diffusivity(spec, fit; nd=1, w0_known=nothing, scale="")
    diffusivity(spec, fit; nd=1, scale="")

Return the diffusivity of an individual population given (i) τD and w0 or (ii) model specifications and 
the result of a fit.

# Keyword arguments
- `nd::Int`: If given a fit, refers to the diffusive population (if there are many) 
             which the diffusivity is to be calculated for.
- `scale::String`: The metric prefix by which to scale metres (e.g., `scale = μ` -> [D] = μm²/s)

# Notes
- For models **without** fixed diffusivity, diffusion slots in the fit are either τD or diffusivity.
  In the former case, to get D you must also provide w₀ (because τD → D needs w₀).
  In the latter, the fitted value is simply the diffusivity
"""
function diffusivity end

function diffusivity(diff_time::Real, w0::Real; scale::String="")
    w0 > 0 || throw(w0_SIGN_ERROR)
    diff_time > 0 || throw(PosError("τD"))

    diff = w0^2 / (4diff_time)
    haskey(SI_PREFIXES, scale) && (diff *= SI_PREFIXES[scale]^2)
    return diff
end

function diffusivity(spec::FCSModelSpec{D,S,OFF,true}, fit; scale::String="") where {D,S,OFF}
    diff = spec.diffusivity
    haskey(SI_PREFIXES, scale) && (diff *= SI_PREFIXES[scale]^2)
    return diff
end

function diffusivity(spec::FCSModelSpec{D,S,OFF,false,false}, fit::FCSFitResult;
                     nd::Int=1, w0::Union{Nothing,Real}=nothing, scale::String="") where {D,S,OFF}
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)
    w0 === nothing && throw(w0_REQUIRED_ERROR)

    diff_time = τD(spec, fit; nd, scale)  # get τD in base units
    return diffusivity(diff_time, w0; scale)
end

function diffusivity(spec::FCSModelSpec{D,S,OFF,false,true}, fit::FCSFitResult;
                     nd::Int=1, scale::String="") where {D,S,OFF}
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)
    
    idx = 1
    !hasoffset(spec) && (idx += 1)
    dim(spec) === d3 && (idx += 1)
    diff = coef(fit)[idx+nd]
    haskey(SI_PREFIXES, scale) && (diff *= SI_PREFIXES[scale]^2)
    return diff
end


"""
    Veff(w0, κ; scale="")
    Veff(spec, fit; nd=1, w0=nothing, scale="")

Calculate the effective volume from fitted FCS parameters.
"""
function Veff end

function Veff(w0::Real, κ::Real; scale::String="")
    w0 > 0 || throw(w0_SIGN_ERROR)
    κ > 0 || throw(κ_SIGN_ERROR)

    vol = π^(3/2) * w0^3 * κ
    haskey(SI_PREFIXES, scale) && (vol *= SI_PREFIXES[scale]^3)
    return vol
end

function Veff(spec::FCSModelSpec{d3,S,OFF,FD,FW}, fit::FCSFitResult;
              nd::Int = 1, w0::Union{Nothing,Real}=nothing, scale::String="") where {S,OFF,FD,FW}
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)

    θ = coef(fit)
    idx = 2
    !hasoffset(spec) && (idx += 1)
    κ = θ[idx]

    w0_eff = FW ? spec.beamwidth : FD ? θ[idx + nd] : (w0 === nothing ? throw(w0_REQUIRED_ERROR) : w0)
    return Veff(w0_eff, κ; scale)
end


"""
    Aeff(w0; scale="")
    Aeff(spec, fit; nd=1, w0=nothing, scale="")

Calculate the area formed by the beam waist from fitted FCS parameters.
"""
function Aeff(w0::Real; scale::String="") 
    w0 > 0 || throw(w0_SIGN_ERROR)
    
    area = π * w0^2
    haskey(SI_PREFIXES, scale) && (area *= SI_PREFIXES[scale]^2)
    return area
end

function Aeff(spec::FCSModelSpec{d2,S,OFF,FD,FW,NDIFF}, fit::FCSFitResult;
              nd::Int = 1, w0::Union{Nothing,Real}=nothing, scale::String="") where {S,OFF,FD,FW,NDIFF}
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)

    idx = 1
    !hasoffset(spec) && (idx += 1)

    w0_eff = FW ? spec.beamwidth : FD ? coef(fit)[idx + nd] : (w0 === nothing ? throw(w0_REQUIRED_ERROR) : w0)

    return Aeff(w0_eff; scale)
end


"""
    concentration(g0, κ, w0; Ks=[], ics=[0], scale="L")
    concentration(spec, fit; nd=1, w0=nothing, scale="")

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
function concentration(g0::Real, κ::Real, w0::Real; Ks::AbstractVector = Float64[],
                       ics::AbstractVector{Int} = Int[], scale::String="")
    g0 > 0 || throw(PosError("g0"))
    κ > 0 || throw(κ_SIGN_ERROR)
    w0 > 0 || throw(w0_SIGN_ERROR)
    all(0 .<= Ks .< 1) || throw(ArgumentError("All Ks must lie in [0,1)"))

    # Validate / normalize ics
    isempty(Ks) ?
        (ics == []) || throw(ArgumentError("If there are no dynamic fractions, ics must be empty.")) :
        sum(ics) == length(Ks) || throw(DYN_COMP_ERROR)

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

    # base concentration: mol / L
    conc = B0 * 1e-3 / (g0 * AVAGADROS * Veff(w0, κ))
    if haskey(SI_PREFIXES, scale) # mol/L → (prefix)·mol/L
        conc *= SI_PREFIXES[scale]
    end
    return conc
end

function concentration(spec::FCSModelSpec{d3,S,OFF,FD,FW}, fit::FCSFitResult;
                       nd::Int = 1, w0::Union{Nothing,Real}=nothing, scale::String="") where {S,OFF,FD,FW}
    N = n_diff(spec)
    0 < nd ≤ N || throw(nd_ERROR)
    θ = coef(fit)

    g0 = θ[1]
    idx = 2
    !hasoffset(spec) && (idx += 1)
    κ = θ[idx]; idx += 1

    w0_eff = FW ? spec.beamwidth : FD ? θ[idx + nd - 1] : (w0 === nothing ? throw(w0_REQUIRED_ERROR) : w0)

    # diffusion block length:
    #   primary diffusion params: N unless both D and w0 are fixed
    #   weights: N-1
    primary = (FD && FW) ? 0 : N
    diff_comp = primary + (N - 1)
    if S == globe
        diff_comp += 1
    elseif S == perpop
        diff_comp += N
    end
    idx += diff_comp

    m = _ndyn_from_len(length(θ) - (idx - 1))
    ics = isempty(spec.ics) ? ones(Int, m) : spec.ics
    sum(ics) == m || throw(DYN_COMP_ERROR)

    Kdyn = m == 0 ? Float64[] : collect(@view θ[idx + m : idx + 2m - 1])

    return concentration(g0, κ, w0_eff; Ks=Kdyn, ics, scale)
end


"""
    surface_density(g0, w0; Ks=[], ics=Int[], scale="")
    surface_density(spec, fit; nd=1, w0=nothing, scale="")

Estimate the **molar surface density** (in mol/m^2) from FCS fit parameters.
Analogue to `concentration` when the a 2d fit is performed.
"""
function surface_density(g0::Real, w0::Real; Ks::AbstractVector = Float64[],
                         ics::AbstractVector{Int} = Int[], scale::String="")
    g0 > 0 || throw(PosError("g0"))
    w0 > 0 || throw(w0_SIGN_ERROR)
    all(0 .<= Ks .< 1) || throw(ArgumentError("All Ks must lie in [0,1)"))

    # Validate / normalize ics
    isempty(Ks) ?
        (ics == []) || throw(ArgumentError("If there are no dynamic fractions, ics must be empty.")) :
        sum(ics) == length(Ks) || throw(DYN_COMP_ERROR)

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

    dens = B0 / (g0 * AVAGADROS * Aeff(w0))
    haskey(SI_PREFIXES, scale) && (dens *= SI_PREFIXES[scale])
    return dens
end

function surface_density(spec::FCSModelSpec{d2,S,OFF,FD,FW}, fit::FCSFitResult;
                         nd::Int = 1, w0::Union{Nothing,Real}=nothing, scale::String="") where {S,OFF,FD,FW}
    N = n_diff(spec)
    0 < nd ≤ N || throw(nd_ERROR)
    θ = coef(fit)

    g0 = θ[1]
    idx = 2
    !hasoffset(spec) && (idx += 1)     # skip offset if it exists

    # idx now points at the first diffusion-related slot (if any)
    w0_eff = FW ? spec.beamwidth : FD ? θ[idx + nd - 1] : (w0 === nothing ? throw(w0_REQUIRED_ERROR) : w0)

    primary = (FD && FW) ? 0 : N
    diff_comp = primary + (N - 1)
    if S == globe
        diff_comp += 1
    elseif S == perpop
        diff_comp += N
    end
    idx += diff_comp

    m = _ndyn_from_len(length(θ) - (idx - 1))
    ics = isempty(spec.ics) ? ones(Int, m) : spec.ics
    sum(ics) == m || throw(DYN_COMP_ERROR)

    Ks = m == 0 ? Float64[] : collect(@view θ[idx + m : idx + 2m - 1])

    return surface_density(g0, w0_eff; Ks, ics, scale)
end


"""
    hydrodynamic(D; T=293.0, η=1.0016e-3, scale="")
    hydrodynamic(diff_time, w0; T=293.0, η=1.0016e-3, scale="")

Calculate the effective hydrodynamic radius of a molecule using the Stokes-Einstein relation.

# Keyword Arguments
- `T=293.0`: Temperature (in Kelvin)
- `η=1.0016e-3`: Viscosity of water (Pa⋅s)
"""
function hydrodynamic(D::Real; T=293.0, η=1.0016e-3, scale::String="")
    D > 0 || throw(D_SIGN_ERROR)
    T > 0 || throw(PosError("Temperature"))
    η > 0 || throw(PosError("Viscosity"))

    rh = BOLTZMANN * T / (6π * η * D)
    haskey(SI_PREFIXES, scale) && (rh *= SI_PREFIXES[scale])
    return rh
end

hydrodynamic(diff_time::Real, w0::Real; kwargs...) =
    hydrodynamic(diffusivity(diff_time, w0); kwargs...)