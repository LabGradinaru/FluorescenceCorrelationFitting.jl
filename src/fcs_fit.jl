"""
    log_lags(n_points::Int, τmin::Int, τmax::Int)

Strictly increasing integer log-spaced lags in [τmin, τmax], zero-based.
Returns *fewer* than n_points if there aren't enough distinct integers.
"""
function log_lags(n_points::Int, τmin::Int, τmax::Int)
    @assert n_points ≥ 1
    @assert 0 ≤ τmin ≤ τmax
    # Work on [τmin+1, τmax+1] in log space, then subtract 1 to get zero-based
    r = range(log10(τmin + 1), log10(τmax + 1); length=n_points)
    lags = round.(Int, 10 .^ r .- 1)
    # Clamp and enforce strict monotonicity
    @inbounds for i in eachindex(lags)
        lags[i] = clamp(lags[i], τmin, τmax)
        if i > 1 && lags[i] ≤ lags[i-1]
            lags[i] = min(lags[i-1] + 1, τmax)
        end
    end
    # Drop duplicates if τ-range is too small
    return unique(lags)
end

"""
    infer_noscale_indices(model_name::Symbol, p0::AbstractVector;
                          n_diff::Union{Nothing,Int}=nothing,
                          offset::Union{Nothing,Real}=nothing) -> Vector{Int}

Return 1-based indices in `p0` that should NOT be scaled (kept at scale 1):
- Mixture weights (for `*_mdiff` models)
- Dynamic fractions `K_dyn`

This reflects the **parameter orderings**:
- `fcs_2d`           : [g0, offset?, τD, τ_dyn..., K_dyn...]
- `fcs_2d_mdiff`     : [g0, offset?, τD[1:n], w[1:n-1], τ_dyn..., K_dyn...]
- `fcs_2d_anom`      : [g0, offset?, τD, α, τ_dyn..., K_dyn...]
- `fcs_2d_anom_mdiff`: [g0, offset?, τD[1:n], α[1:n], w[1:n-1], τ_dyn..., K_dyn...]
- `fcs_3d`           : [g0, offset?, κ, τD, τ_dyn..., K_dyn...]
- `fcs_3d_mdiff`     : [g0, offset?, κ, τD[1:n], w[1:n-1], τ_dyn..., K_dyn...]
"""
function infer_noscale_indices(model_name::Symbol, p0::AbstractVector;
                               n_diff::Union{Nothing,Int}=nothing,
                               offset::Union{Nothing,Real}=nothing)
    L = length(p0)
    if model_name === :fcs_2d
        # base (up to τD): g0, (offset), τD
        base = isnothing(offset) ? 3 : 2
        m = _ndyn_from_len(L - base)
        return m == 0 ? Int[] : collect(base + m + 1 : base + 2m)

    elseif model_name === :fcs_2d_mdiff
        isnothing(n_diff) && throw(ArgumentError("n_diff required for fcs_2d_mdiff"))
        base0 = isnothing(offset) ? 2 : 1
        τ_end = base0 + n_diff
        w_start = τ_end + 1
        w_end = w_start + (n_diff > 1 ? (n_diff - 1) : 0) - 1
        diff_idx = max(τ_end, w_end)

        m = _ndyn_from_len(L - diff_idx)
        idx = Int[]
        n_diff > 1 && append!(idx, collect(w_start:w_end))
        m > 0 && append!(idx, collect(diff_idx + m + 1 : diff_idx + 2m))
        return idx

    elseif model_name === :fcs_2d_anom_mdiff
        isnothing(n_diff) && throw(ArgumentError("n_diff required for fcs_2d_anom_mdiff"))
        base0 = isnothing(offset) ? 2 : 1
        τ_end = base0 + n_diff
        α_end = τ_end + n_diff
        w_end = α_end + (n_diff > 1 ? (n_diff - 1) : 0)
        diff_idx = max(α_end, w_end)

        m = _ndyn_from_len(L - diff_idx)
        idx = Int[]
        n_diff > 1 && append!(idx, collect(α_end+1:w_end))
        m > 0 && append!(idx, collect(diff_idx + m + 1 : diff_idx + 2m))
        return idx
    
    elseif (model_name === :fcs_3d) || (model_name === :fcs_2d_anom)
        base = isnothing(offset) ? 4 : 3
        m = _ndyn_from_len(L - base)
        return m == 0 ? Int[] : collect(base + m + 1 : base + 2m)

    elseif model_name === :fcs_3d_mdiff
        isnothing(n_diff) && throw(ArgumentError("n_diff required for fcs_3d_mdiff"))
        base0 = isnothing(offset) ? 3 : 2
        τ_end = base0 + n_diff
        w_start = τ_end + 1
        w_end   = w_start + (n_diff > 1 ? (n_diff - 1) : 0) - 1
        diff_idx = max(τ_end, w_end)

        m = _ndyn_from_len(L - diff_idx)
        idx = Int[]
        n_diff > 1 && append!(idx, collect(w_start:w_end))
        m > 0 && append!(idx, collect(diff_idx + m + 1 : diff_idx + 2m))
        return idx

    else
        return Int[]
    end
end


"""
    build_scales_from_p0(p0; noscale_idx=Int[], zero_sub=1.0)

Construct a scale vector so that, ideally, θ0 = ones and p = scales .* θ reproduces p0.
- For indices in `noscale_idx`, scale is set to 1.0.
- For zero p0 entries, use `zero_sub` to avoid zero scale; θ0 at those indices becomes p0/zero_sub (often 0).
Returns (θ0, scales).
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
    θ0 = p0 ./ s   # equals ones except where p0==0 or noscale indices
    return θ0, s
end


"""
    fcs_fit(model::Function, lag_times, corr_data, p0; 
            wt=nothing, n_diff=nothing, scales=nothing, zero_sub=1.0, kwargs...)

Fit with LsqFit using parameter normalization.
- If `scales` is `nothing`, they are inferred from `p0` so that θ0 ≈ ones.
- For diffusion mixture models, pass `n_diff`.
- `wt` (if provided) is forwarded as `weights=wt`.
Returns the LsqFit result and also `scales` so you can recover physical params.
"""
function fcs_fit(model::Function, lag_times::AbstractVector, 
                 corr_data::AbstractVector, p0::AbstractVector;
                 σ::Union{Nothing,AbstractVector}=nothing,
                 wt::Union{Nothing,AbstractVector}=nothing,
                 n_diff::Union{Nothing,Int}=nothing,
                 scales::Union{Nothing,AbstractVector}=nothing,
                 ics::Union{Nothing,AbstractVector{Int}}=nothing,
                 diffusivity::Union{Nothing,Real}=nothing,
                 offset::Union{Nothing,Real}=nothing,
                 zero_sub::Real=1.0, kwargs...)
    # consistency checks
    length(lag_times) == length(corr_data) ||
        throw(ArgumentError("Lag times and correlation values must be of equal length."))
    !isnothing(wt) && (length(wt) == length(lag_times) ||
        throw(ArgumentError("Weights must have same size as lag times and data.")))
    !isnothing(σ) && (length(σ) == length(lag_times) ||
        throw(ArgumentError("Standard deviations must have same size as lag times and data.")))

    if isnothing(wt)
        if isnothing(σ) # empty array => equal weights
            wt = ones(eltype(corr_data), length(corr_data))
        else # std devs. are given so use 1/σ² weight
            wt = @. 1 / σ^2
        end
    end
    (isnothing(wt) && !isnothing(σ)) && ()

    # infer model name to pick non-scaled indices
    mname = nameof(model)  # Symbol if model is a named function
    model_sym = mname isa Symbol ? mname : :unknown

    # Decide which indices should not be scaled
    noscale_idx = infer_noscale_indices(model_sym, p0; n_diff, offset)

    # Build scales if not provided; get normalized θ0
    if isnothing(scales)
        θ0, scales_ = build_scales_from_p0(p0; noscale_idx=noscale_idx, zero_sub=zero_sub)
    else
        length(scales) == length(p0) || throw(ArgumentError("Provided scales length mismatch."))
        θ0 = p0 ./ scales
        scales_ = scales
    end

    # Build a two-arg model for LsqFit that maps θ → p
    model2 = isnothing(n_diff) ?
        ((x, θ) -> model(x, θ; scales=scales_, ics, diffusivity, offset)) :
        ((x, θ) -> model(x, θ; scales=scales_, ics, n_diff, offset))

    # Extract optional bounds/weights from kwargs
    lower_in = get(kwargs, :lower, nothing)
    upper_in = get(kwargs, :upper, nothing)
    filtered_kwargs = (; (k => v for (k,v) in kwargs if !(k in (:lower, :upper)))...)

    normalize_bounds(b) = b === nothing ? nothing :
        (length(b) == length(scales_) ? b ./ scales_ :
        throw(ArgumentError("lower/upper must have length $(length(scales_))")))

    lowerθ = normalize_bounds(lower_in)
    upperθ = normalize_bounds(upper_in)

    # Fitting
    x = collect(lag_times)
    fit = if (lowerθ !== nothing) && (upperθ !== nothing)
        curve_fit(model2, x, corr_data, wt, θ0; lower = lowerθ, upper = upperθ, filtered_kwargs...) 
    elseif (lowerθ !== nothing)
        curve_fit(model2, x, corr_data, wt, θ0; lower = lowerθ, filtered_kwargs...)
    elseif (upperθ !== nothing)
        curve_fit(model2, x, corr_data, wt, θ0; upper = upperθ, filtered_kwargs...)
    else
        curve_fit(model2, x, corr_data, wt, θ0; filtered_kwargs...)
    end

    return fit, scales_
end