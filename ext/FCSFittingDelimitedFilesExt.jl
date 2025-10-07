module FCSFittingDelimitedFilesExt

using DelimitedFiles
import FCSFitting: FCSChannel, FCSData, read_fcs

"""
    read_fcs(path; start_idx=nothing, end_idx=nothing, colspec=:auto, metadata=Dict{String,Any}())
Read FCS data from a .txt or .csv filetype.
"""
function read_fcs(path::AbstractString; start_idx::Union{Nothing,Int}=nothing,
                  end_idx::Union{Nothing,Int}=nothing,
                  colspec=:auto, metadata=Dict{String,Any}())

    raw = readdlm(path)
    r1 = isnothing(start_idx) ? 1 : start_idx
    r2 = isnothing(end_idx)   ? size(raw,1) : end_idx
    M  = raw[r1:r2, :]

    # Basic inference: 1st col = τ, next 4 = Gs, next 4 = σs (if present)
    if colspec === :auto
        ncol = size(M,2)
        τ = vec(M[:,1])
        chans = FCSChannel[]
        # try pairs (G,σ) for columns 2..n
        i = 2
        k = 1
        while i <= ncol
            G = vec(M[:,i])
            σ = (i+4 <= ncol && ncol >= 9) ? vec(M[:, i+4]) : nothing
            push!(chans, FCSChannel("G[$k]", τ, G, σ))
            i += 1
            k += 1
            if k > 4 && ncol <= 9; break; end
        end
        return FCSData(chans, metadata, String(path))
    else
        # explicit mapping
        names = String[]
        chans = FCSChannel[]
        τ = vec(M[:, first(first(colspec)) == :τ ? last(first(colspec)) : error("τ col missing")])
        # build channels
        for tup in colspec
            sym, idx = tup
            if sym === :τ; continue; end
            if sym === :G
                name = get(tup, 3, "G")
                σidx  = get(tup, 4, nothing)
                σ = isnothing(σidx) ? nothing : vec(M[:, σidx])
                push!(chans, FCSChannel(name, τ, vec(M[:,idx]), σ))
            end
        end
        return FCSData(chans, metadata, String(path))
    end
end

end # module