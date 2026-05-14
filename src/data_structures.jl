"""
    FCSChannel{T,S}(name, τ, G, σ)

A single correlation **channel** for FCS data.

# Fields
- `name::String` — Channel label (e.g., `"G_DD"`, `"G_DA"`, or `"G[1]"`).
- `τ::Vector{T}` — Lag times in seconds (same length as `G`).
- `G::Vector{T}` — Correlation values.
- `σ::S` — Optional per-lag standard deviations (same length as `G` if present).

# Notes
All vectors are assumed to be aligned element-wise (`τ[i]` ↔ `G[i]` ↔ `σ[i]`).
"""
struct FCSChannel{T,S}
    name::String
    τ::Vector{T}
    G::Vector{T}
    σ::S

    function FCSChannel(name::String,τ::Vector{T},G::Vector{T},σ::S) where {T,S}
        @assert length(τ) == length(G) "Number of lag times must equal correlation vector length."
        if σ isa AbstractVector
            @assert length(τ) == length(σ) "Number of standard deviations must equal correlation vector length."
        end
        new{T,S}(name,τ,G,σ)
    end
end

Base.length(ch::FCSChannel) = length(ch.τ)


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
    metadata::Dict{String,Any}
    source::String
end

Base.length(data::FCSData) = length(data.channels)