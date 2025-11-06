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


# ─────────────────────────────────────────────────────────────────────────────
# Fitting
# ─────────────────────────────────────────────────────────────────────────────

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

function fcs_fit(m::FCSModel, τ::AbstractVector, data::AbstractVector, p0::AbstractVector; kwargs...)
    if m.scales === nothing
        return fcs_fit(m.spec, τ, data, p0; kwargs...)
    else
        return fcs_fit(m.spec, τ, data, p0; scales=m.scales, kwargs...)
    end
end

fcs_fit(m::FCSModel, ch::FCSChannel, p0::AbstractVector; kwargs...) =
    fcs_fit(m, ch.τ, ch.G, p0; σ=ch.σ, kwargs...)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience calculators
# ─────────────────────────────────────────────────────────────────────────────

PosError(x) = ArgumentError(string(x, " must be positive."))
const w0_SIGN_ERROR = PosError("w0")
const κ_SIGN_ERROR = PosError("κ")
const D_SIGN_ERROR = PosError("Diffusivity")
const nd_ERROR = ArgumentError("The chosen diffuser must be at a positive index less than the total number of diffusers.")
const w0_REQUIRED_ERROR = ArgumentError("This model does not fix diffusivity, so you must provide `w0=` to compute D.")


"""
    τD(D, w0; scale="")

Convert diffusion coefficient `D` and lateral waist `w0` to the lateral diffusion time τD.
"""
function τD(D::Real, w0::Real; scale::String="")
    w0 > 0 || throw(w0_SIGN_ERROR)
    D > 0 || throw(D_SIGN_ERROR)
    
    diff_time = w0^2 / (4D)
    haskey(SI_PREFIXES, scale) && (diff_time *= SI_PREFIXES[scale])
    return diff_time
end

"""
    τD(spec, fit; nd=1, scale="")

Get the nd-th diffusion time from a fitted model.

- If the spec used a **fixed diffusivity**, the corresponding slot in the fit is w₀,
  so we convert w₀ → τD.
- Otherwise, the slot is already τD so is scaled by `scale`
"""
function τD(spec::FCSModelSpec, fit::FCSFitResult; nd::Int = 1, scale::String = "")
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)

    θ = coef(fit)
    idx = 1
    !hasoffset(spec) && (idx += 1)
    dim(spec) === d3 && (idx += 1)

    diff_slot = idx + nd
    if hasdiffusivity(spec) # slot holds w0
        w0 = θ[diff_slot]
        return τD(spec.diffusivity, w0; scale)
    else # slot already holds τD
        diff_time = θ[diff_slot]
        haskey(SI_PREFIXES, scale) && (diff_time *= SI_PREFIXES[scale])
        return diff_time
    end
end

"""
    diffusivity(τD, w0; scale="")

Convert diffusion time `τD` and beam waist `w0` to the diffusivity.
"""
function diffusivity(τD::Real, w0::Real; scale::String="")
    w0 > 0 || throw(w0_SIGN_ERROR)
    τD > 0 || throw(PosError("τD"))

    diff = w0^2 / (4τD)
    # if the user says e.g. "μ", we scale the length unit; D is m^2/s → (prefix m)^2/s
    haskey(SI_PREFIXES, scale) && (diff *= SI_PREFIXES[scale]^2)
    return diff
end

"""
    diffusivity(spec; scale="")

Return the fixed diffusivity from the spec (error if it was not fixed).
"""
function diffusivity(spec::FCSModelSpec{D,S,OFF,true}; scale::String = "") where {D,S,OFF}
    diff = spec.diffusivity
    haskey(SI_PREFIXES, scale) && (diff *= SI_PREFIXES[scale]^2)
    return diff
end

"""
    diffusivity(spec, fit; nd=1, w0_known=nothing, scale="")

For models **without** fixed diffusivity, diffusion slots in the fit are τD’s.
To get D you must also provide w₀ (because τD → D needs w₀).

If your model used fixed diffusivity, use `diffusivity(spec)` instead.
"""
function diffusivity(spec::FCSModelSpec{D,S,OFF,false}, fit::FCSFitResult;
                     nd::Int=1, w0::Union{Nothing,Real}=nothing, scale::String="") where {D,S,OFF}
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)
    w0 === nothing && throw(w0_REQUIRED_ERROR)

    τd = τD(spec, fit; nd, scale = "")  # get τD in base units
    return diffusivity(τd, w0; scale)
end

"""
    Veff(w0, κ; scale="")

Calculate the effective volume from fitted FCS parameters.
"""
function Veff(w0::Real, κ::Real; scale::String="")
    w0 > 0 || throw(w0_SIGN_ERROR)
    κ > 0 || throw(κ_SIGN_ERROR)

    vol = π^(3/2) * w0^3 * κ
    haskey(SI_PREFIXES, scale) && (vol *= SI_PREFIXES[scale]^3)
    return vol
end

"""
    Veff(spec, fit; nd=1, scale="")

3D-only. Pull κ and the nd-th w₀/τD from the fit and compute Veff.

- if diffusivity was fixed → diffusion slot is w₀ → we can compute Veff
- if diffusivity was free → diffusion slot is τD → user must give w₀
"""
function Veff(spec::FCSModelSpec{d3,S,OFF,true}, fit::FCSFitResult;
              nd::Int = 1, scale::String = "") where {S,OFF}
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)
    θ = coef(fit)

    idx = 2
    !hasoffset(spec) && (idx += 1)
    κ = θ[idx]
    w0 = θ[idx + nd]
    return Veff(w0, κ; scale)
end

function Veff(spec::FCSModelSpec{d3,S,OFF,false}, fit::FCSFitResult;
              w0::Union{Nothing,Real}=nothing, scale::String = "") where {S,OFF}
    w0 === nothing && throw(w0_REQUIRED_ERROR)
    θ = coef(fit)

    idx = 2
    !hasoffset(spec) && (idx += 1)
    κ = θ[idx]
    return Veff(w0, κ; scale)
end

"""
    Aeff(w0; scale="")

Calculate the area formed by the beam waist `w0`.
"""
function Aeff(w0::Real; scale::String="") 
    w0 > 0 || throw(w0_SIGN_ERROR)
    
    area = π * w0^2
    haskey(SI_PREFIXES, scale) && (area *= SI_PREFIXES[scale]^2)
    return area
end

"""
    Aeff(spec, fit; nd=1, scale="")

Extract w₀ from the fit and compute the confocal area.
"""
function Aeff(spec::FCSModelSpec{d2,S,OFF,true,NDIFF}, fit::FCSFitResult;
              nd::Int = 1, scale::String = "") where {S,OFF,NDIFF}
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)
    θ = coef(fit)

    idx = 1
    !hasoffset(spec) && (idx += 1)
    w0 = θ[idx + nd]

    return Aeff(w0; scale)
end

"""
    concentration(g0, κ, w0; Ks=[], ics=[0], scale="L")

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

"""
    concentration(spec, fit; nd=1, Ks=[], scale="L")

Pull g₀, κ, w₀ (in the fixed-diffusivity case), and Ks from the fit 
and determine the concentration.
"""
function concentration(spec::FCSModelSpec{d3,S,OFF,true}, fit::FCSFitResult;
                       nd::Int = 1, scale::String = "") where {S,OFF}
    N = n_diff(spec)
    0 < nd ≤ N || throw(nd_ERROR)
    θ = coef(fit)

    g0 = θ[1];  idx = 2
    !hasoffset(spec) && (idx += 1)
    κ = θ[idx];  idx += 1
    w0 = θ[idx+nd-1]

    # total number of components in θ allocated to τD/ w₀ + weights
    diff_comp = 2N - 1
    if S == globe
        diff_comp += 1
    elseif S == perpop
        diff_comp += N
    end
    idx += diff_comp

    m = _ndyn_from_len(length(θ) - (idx - 1))
    ics = isempty(spec.ics) ? ones(Int, m) : spec.ics
    sum(ics) == m || throw(DYN_COMP_ERROR)

    Kdyn = m == 0 ? Float64[] : collect(@view θ[idx+m:idx+2m-1])

    return concentration(g0, κ, w0; Ks=Kdyn, ics, scale)
end

function concentration(spec::FCSModelSpec{d3,S,OFF,false}, fit::FCSFitResult; nd::Int = 1, 
                       w0::Union{Nothing,Real}=nothing, scale::String = "") where {S,OFF}
    N = n_diff(spec)
    0 < nd ≤ N || throw(nd_ERROR)
    w0 === nothing && throw(w0_REQUIRED_ERROR)
    θ = coef(fit)

    g0 = θ[1];  idx = 2
    !hasoffset(spec) && (idx += 1)
    κ = θ[idx];  idx += 1

    diff_comp = 2N - 1
    if S == globe
        diff_comp += 1
    elseif S == perpop
        diff_comp += N
    end
    idx += diff_comp

    m = _ndyn_from_len(length(θ) - (idx - 1))
    ics = isempty(spec.ics) ? ones(Int, m) : spec.ics
    sum(ics) == m || throw(DYN_COMP_ERROR)

    Kdyn = m == 0 ? Float64[] : collect(@view θ[idx+m:idx+2m-1])

    return concentration(g0, κ, w0; Ks=Kdyn, ics, scale)
end

"""
    surface_density(w0, g0; Ks=[], ics=[0], scale="")

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

"""
    surface_density(spec, fit; nd=1, scale="")

2D analogue of `concentration(spec, fit, ...)`.

Pulls g₀ and w₀ (in the fixed-diffusivity case) from the fitted parameter
vector, then reconstructs the dynamic fractions from the tail of the vector, and
finally calls the base `surface_density(g0, w0; Ks, ics, scale)`.
"""
function surface_density(spec::FCSModelSpec{d2,S,OFF,true}, fit::FCSFitResult;
                         nd::Int = 1, scale::String = "") where {S,OFF}
    N = n_diff(spec)
    0 < nd ≤ N || throw(nd_ERROR)
    θ = coef(fit)

    g0 = θ[1];  idx = 2
    !hasoffset(spec) && (idx += 1)
    w0 = θ[idx + nd - 1]

    diff_comp = 2N - 1
    if S == globe
        diff_comp += 1
    elseif S == perpop
        diff_comp += N
    end
    idx += diff_comp

    m = _ndyn_from_len(length(θ) - (idx - 1))
    ics = isempty(spec.ics) ? ones(Int, m) : spec.ics
    sum(ics) == m || throw(DYN_COMP_ERROR)

    Ks = m == 0 ? Float64[] : collect(@view θ[dyn_start + m : dyn_start + 2m - 1])

    return surface_density(g0, w0; Ks, ics, scale)
end

"""
    surface_density(spec, fit; nd=1, w0=..., scale="")

2D, **free diffusivity**: the diffusion slots hold τᴅ, not w₀, so the user
must supply `w0=` to convert to a surface density.
"""
function surface_density(spec::FCSModelSpec{d2,S,OFF,false}, fit::FCSFitResult;
                         nd::Int = 1, w0::Union{Nothing,Real} = nothing,
                         scale::String = "") where {S,OFF}
    N = n_diff(spec)
    0 < nd ≤ n_diff(spec) || throw(nd_ERROR)
    w0 === nothing && throw(w0_REQUIRED_ERROR)
    θ = coef(fit)

    g0 = θ[1];  idx = 2
    !hasoffset(spec) && (idx += 1)

    diff_comp = 2N - 1
    if S == globe
        diff_comp += 1
    elseif S == perpop
        diff_comp += N
    end
    idx += diff_comp

    m = _ndyn_from_len(length(θ) - (idx - 1))
    ics = isempty(spec.ics) ? ones(Int, m) : spec.ics
    sum(ics) == m || throw(DYN_COMP_ERROR)

    Ks = m == 0 ? Float64[] : collect(@view θ[dyn_start + m : dyn_start + 2m - 1])

    return surface_density(g0, w0; Ks, ics, scale)
end

"""
    hydrodynamic(D; T=293.0, η=1.0016e-3, scale="")
    hydrodynamic(τD, w0; T=293.0, η=1.0016e-3, scale="")

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

hydrodynamic(τD::Real, w0::Real; kwargs...) =
    hydrodynamic(diffusivity(τD, w0); kwargs...)