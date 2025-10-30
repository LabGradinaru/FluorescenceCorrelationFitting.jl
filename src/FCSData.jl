"""
    FCSChannel(name, τ, G, σ)

A single correlation **channel** for FCS data.

# Fields
- `name::String` — Channel label (e.g., `"G_DD"`, `"G_DA"`, or `"G[1]"`).
- `τ::Vector{Float64}` — Lag times in seconds (same length as `G`).
- `G::Vector{Float64}` — Correlation values.
- `σ::Union{Nothing,Vector{Float64}}` — Optional per-lag standard deviations (same length as `G` if present).

# Notes
All vectors are assumed to be aligned element-wise (`τ[i]` ↔ `G[i]` ↔ `σ[i]`).
"""
struct FCSChannel
    name::String                 # e.g. "G_DD" or "G_DA"
    τ::Vector{Float64}           # lag times (s)
    G::Vector{Float64}           # correlation values
    σ::Union{Nothing,AbstractArray{Float64}}  # std dev per lag (optional)
end


"""
    FCSData(channels, metadata, source)

A container for **multi-channel** FCS data plus provenance.

# Fields
- `channels::Vector{FCSChannel}` — One or more correlation channels sharing a `τ` grid.
- `metadata::Dict{String,Any}` — Arbitrary key-value info (sample, T, NA, λ, pinhole, detector, etc.).
- `source::String` — Provenance string (e.g., file path or `"in-memory"`).
"""
struct FCSData
    channels::Vector{FCSChannel}
    metadata::Dict{String,Any}   # sample, T, NA, λ, pinhole, detector, etc.
    source::String               # filepath or “in-memory”
end