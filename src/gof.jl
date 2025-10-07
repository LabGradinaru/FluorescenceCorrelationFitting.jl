"""
    aic(fit) -> Real

Akaike Information Criterion for a least-squares fit.

# Definition
`AIC = 2k + N*log(σ²)`, where:
- `k = length(coef(fit))` is the number of fitted parameters,
- `N = nobs(fit)` is the number of observations,
- `σ² = rss(fit)/N` is the residual variance estimate.

# Returns
- Lower is better (relative comparison across models on the same dataset).
"""
function aic(fit::LsqFit.LsqFitResult)
    k, N = length(coef(fit)), nobs(fit)
    σ2 = rss(fit) / N
    return 2k + N*log(σ2)
end

"""
    aicc(fit) -> Real

Small-sample corrected AIC.

# Definition
`AICc = AIC + (2k(k+1)) / (N - k - 1)`.

# Returns
- Recommended when `N / k` is modest.
"""
function aicc(fit::LsqFit.LsqFitResult)
    k, N = length(coef(fit)), nobs(fit)
    a = aic(fit)
    return a + (2k*(k+1)) / (N - k - 1)
end

"""
    bic(fit) -> Real

Bayesian Information Criterion for a least-squares fit.

# Definition
`BIC = k*log(N) + N*log(σ²)`, with
- `k = length(coef(fit))`,
- `N = nobs(fit)`,
- `σ² = rss(fit)/N`.

# Returns
- Lower is better; BIC penalizes model complexity more strongly than AIC.
"""
function bic(fit::LsqFit.LsqFitResult)
    k, N = length(coef(fit)), nobs(fit)
    σ2 = rss(fit) / N
    return k*log(N) + N*log(σ2)
end

"""
    bicc(fit) -> Real

Bias-corrected BIC variant.

# Definition
`BICc = N*log(σ²) + N*k*log(N)/(N - k - 2)`.

# Returns
- A more conservative penalty when `N` is not ≫ `k`.
"""
function bicc(fit::LsqFit.LsqFitResult)
    k = length(coef(fit)); N = nobs(fit)
    σ2 = rss(fit) / N
    return N * log(σ2) + N * k * log(N) / (N - k - 2)
end

"""
    chi_squared(fit; σ=nothing, dof_override=nothing)

Compute χ², reduced χ², degrees of freedom, and a p-value for a least-squares fit.

- Otherwise uses unweighted χ² = Σ rᵢ², which corresponds to σ = 1.
- dof = N - k by default (or `dof_override` if provided).
"""
function chi_squared(fit::LsqFit.LsqFitResult; σ::Union{Nothing,AbstractVector}=nothing,
                     dof_override::Union{Nothing,Int}=nothing, reduced::Bool=false)
    r = fit.resid
    N = length(r)
    k = length(coef(fit))
    dof = isnothing(dof_override) ? max(N - k, 1) : dof_override

    χ2 = if σ === nothing
        sum(abs2, r)
    else
        @assert length(σ) == N "σ length must match residuals"
        s2 = @. (r/σ)^2
        sum(s2)
    end

    chi2 = χ2
    reduced && (chi2 /= dof)
    chi2
end

chi_squared(fit::LsqFit.LsqFitResult, σ::AbstractVector) =
    chi_squared(fit; σ)

"""
    ww_test(x; drop_zeros=true)

Wald–Wolfowitz runs test on a sequence `x` (typically residuals).

- Drops zeros by default (they are uninformative for sign-runs).
- Uses the normal approximation with continuity correction.
"""
function ww_test(x::AbstractVector; drop_zeros::Bool=true)
    @assert !isempty(x)
    s = sign.(x)
    if drop_zeros
        s = s[s .!= 0]
    end
    N = length(s)
    @assert N > 1 "Need at least two non-zero-signed elements"

    n_pos = count(>(0), s)
    n_neg = N - n_pos
    @assert n_pos > 0 && n_neg > 0 "Runs test undefined if all signs are same"

    # Count runs
    R = _count_runs(s)

    # Mean/Var under H0 (random signs)
    μ = 1 + 2n_pos*n_neg / N
    σ2 = (2n_pos*n_neg * (2n_pos*n_neg - N)) / (N^2 * (N - 1))
    σ = sqrt(σ2)

    cc = R > μ ? 0.5 : -0.5

    (R - μ + cc) / σ # z-score
end

ww_test(fit::LsqFit.LsqFitResult; drop_zeros::Bool=true) =
    ww_test(fit.resid; drop_zeros)

"""
    _count_runs(s)

Count runs in a sign vector of ±1. Zeros should be removed before calling.
"""
function _count_runs(s::AbstractVector{<:Integer})
    @inbounds begin
        R = 1
        prev = s[1]
        for i in 2:length(s)
            if s[i] != prev
                R += 1
                prev = s[i]
            end
        end
        return R
    end
end

"""
    acf(x; maxlag, demean=true, unbiased=true)

Autocorrelation function up to `maxlag` (inclusive), returning ρ₀..ρ_maxlag.
"""
function acf(x::AbstractVector; maxlag::Int=clamp(length(x) - 1, 1, 1000),
             demean::Bool=true, unbiased::Bool=true)
    N = length(x)
    @assert maxlag ≥ 1 && maxlag < N "maxlag must be in [1, N-1]"
    μ = demean ? (sum(x)/N) : zero(eltype(x))
    y = x .- μ
    γ0 = sum(abs2, y) / (unbiased ? (N - 1) : N)

    ρ = similar(y, maxlag + 1)
    ρ[1] = 1
    @inbounds for k in 1:maxlag
        num = sum(@view(y[1:N-k]) .* @view(y[1+k:N]))
        denom = unbiased ? (N - k) : N
        ρ[k+1] = (num / denom) / γ0
    end
    return ρ
end

"""
    ljung_box(x; h=:auto, demean=true) -> NamedTuple

Ljung–Box Q test for residual autocorrelation up to lag `h`.

- If `h = :auto`, uses `h = clamp(round(Int, 10*log10(N)), 1, N-2)`.
- Returns `(Q, h, p, reject)` where `Q ∼ χ²_h` under H0 (no autocorrelation).
"""
function ljung_box(x::AbstractVector; h::Union{Int,Symbol}=:auto, demean::Bool=true)
    N = length(x)
    h = h === :auto ? clamp(round(Int, 10*log10(N)), 1, N-2) : h
    @assert 1 ≤ h ≤ N-2 "h must be between 1 and N-2"

    ρ = acf(x; maxlag=h, demean=demean, unbiased=false)  # ρ₀..ρ_h
    # exclude lag 0
    Q = N*(N+1) * sum(@. (ρ[2:end]^2) / (N - (1:h)))
    return Q
end

ljung_box(fit::LsqFit.LsqFitResult; h::Union{Int,Symbol}=:auto, demean::Bool=true) =
    ljung_box(fit.resid; h, demean)