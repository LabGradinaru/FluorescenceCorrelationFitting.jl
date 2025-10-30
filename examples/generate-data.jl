using FCSFitting
using DelimitedFiles
using Random

# -------------------------------
# Model parameters
# -------------------------------
n_points   = 300
start_time = 1e-9
end_time   = 1.0

diffusivity = 5e-11
w0  = 250e-9
g0  = 1.0
s   = 8
τtr = 1e-5
Ttr = 0.1

# -------------------------------
# "Estimator physics" parameters
# -------------------------------
Δt_base   = 1e-6    # correlator base sampling time (s)
T_meas    = 60.0    # nominal measurement duration (s)
η_nominal = 0.01     # global noise scale (tune to your desired level)

# Trial-to-trial randomness (CVs ~ 10-20% are modest)
cv_Tmeas  = 0.15    # CV for T_meas
cv_eta    = 0.10    # CV for global noise scale

# Optional reproducibility
Random.seed!(42)

# -------------------------------
# Helpers
# -------------------------------
# multi-tau averaging factor ~ round(τ / Δt_base), lower-bounded by 1
mtau_factor(τ, Δt) = max(1, round(Int, τ / Δt))

function sigma_fcs(lags::AbstractVector, G::AbstractVector;
                   Δt_base::Real, T_meas::Real, η::Real)
    @assert length(lags) == length(G)
    m = map(τ -> mtau_factor(τ, Δt_base), lags)                 # integer vector
    n_pairs = max.(1.0, (T_meas .- lags) ./ (Float64.(m) .* Δt_base))
    η .* (1 .+ G) ./ sqrt.(n_pairs)
end

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

# Log-normal jitter with given coefficient of variation (median=1)
lognormal_scale(cv) = exp(sqrt(log(1 + cv^2)) * randn())

# -------------------------------
# Generate lags, clean model, σ(τ), and noisy sample
# -------------------------------
lag_times = start_time .* log_lags(n_points, 1, floor(Int, end_time / start_time))
G_clean = fcs_3d(lag_times, [w0, g0, 0.0, s, τtr, Ttr]; diffusivity)

# Trial-level jitter
T_trial  = T_meas * lognormal_scale(cv_Tmeas)
η_trial  = η_nominal * lognormal_scale(cv_eta)

σ = sigma_fcs(lag_times, G_clean; Δt_base=Δt_base, T_meas=T_trial, η=η_trial)

G_noisy = G_clean .+ σ .* randn(length(lag_times))

# -------------------------------
# Save: time, data, standard deviation
# -------------------------------
out = hcat(lag_times, 
           G_clean .+ σ .* randn(length(lag_times)),
           G_clean .+ σ .* randn(length(lag_times)),
           G_clean .+ σ .* randn(length(lag_times)),
           G_clean .+ σ .* randn(length(lag_times)),
           σ, σ, σ, σ)
open("examples/fcs_sample.txt", "w") do io
    writedlm(io, out, ' ')
end