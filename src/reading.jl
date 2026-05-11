using Dates

# PicoQuant Unified Tag Format type codes
const TagEmpty8 = UInt32(0xFFFF0008)
const TagBool8 = UInt32(0x00000008)
const TagInt8 = UInt32(0x10000008)
const TagBitSet64 = UInt32(0x11000008)
const TagColor8 = UInt32(0x12000008)
const TagFloat8 = UInt32(0x20000008)
const TagDateTime = UInt32(0x21000008)
const TagFloat8Array = UInt32(0x2001FFFF)
const TagAnsiString = UInt32(0x4001FFFF)
const TagWideString = UInt32(0x4002FFFF)
const TagBinaryBlob = UInt32(0xFFFFFFFF)
const PQRES = b"PQRESLT\0"


"""
    read_fcs(path; kwargs...)

Read FCS data from common correlation curve file types (currently supports: .txt, .pqres).
Specific methods for handling each filetype are named `_read_ext`, where `ext` is the extension 
(e.g., see `_read_txt` for details on how text files are parsed.)
"""
function read_fcs(path::AbstractString; kwargs...)
    ext = splitext(path)[2]

    if ext == ".txt"
        return _read_txt(path; kwargs...)
    elseif ext == ".pqres"
        return _read_pqres(path; kwargs...)
    else
        return ArgumentError("Files with extension $ext are not currently able to be read to FCSData.")
    end
end


"""
    _read_txt(path; start_idx=nothing, end_idx=nothing, start_time=nothing, end_time=nothing,
              delimeter=" ", linebreak="\r\n", filling_values=0.0, colspec=nothing,
              metadata=Dict{String,Any}(), extra_kwargs...)

Read FCS correlation data from a whitespace- or delimiter-separated text file.

# Column layout
- `colspec=nothing` (default): column 1 = τ, columns 2-5 = G[1]-G[4],
  columns 6-9 = σ[1]-σ[4] (only when all four σ columns are present).
- `colspec` as a `NamedTuple`: explicit mapping, e.g.
  `(τ=1, G=[2,3,4,5], σ=[6,7,8,9], names=["G_DD","G_DA","G_AA","G_cross"])`.
  `σ` and `names` are optional fields.

# Row selection
Rows can be restricted by index (`start_idx`/`end_idx`, 1-based, applied before
any other filtering) or by lag time (`start_time`/`end_time`, inclusive, matched
against the τ column). Both filters may be combined.

Lines that are empty or begin with `#` are ignored before index slicing.
"""
function _read_txt(path::AbstractString;
    start_idx::Union{Nothing,Int}=nothing,
    end_idx::Union{Nothing,Int}=nothing,
    start_time=nothing,
    end_time=nothing,
    delimeter::String=" ",
    linebreak::String="\r\n",
    filling_values=0.0,
    colspec=nothing,
    metadata=Dict{String,Any}(),
    extra_kwargs...
)
    io = open(path, "r")
    content = read(io, String)
    close(io)

    raw = split(strip(content), linebreak)

    # drop blank lines and comment lines
    rows = [r for r in raw if !isempty(strip(r)) && !startswith(strip(r), '#')]

    r1 = isnothing(start_idx) ? 1 : start_idx
    r2 = isnothing(end_idx) ? length(rows) : end_idx
    rows = rows[r1:r2]

    isempty(rows) && return FCSData(FCSChannel[], metadata, String(path))

    # parse into a Float64 matrix, padding short rows with filling_values
    parsed = [filter(!isempty, split(strip(r), delimeter)) for r in rows]
    ncols  = maximum(length, parsed)
    M = fill(Float64(filling_values), length(parsed), ncols)
    for (i, parts) in enumerate(parsed)
        for (j, p) in enumerate(parts)
            M[i, j] = parse(Float64, p)
        end
    end

    # optional time-domain filtering on the τ column (column 1)
    if !isnothing(start_time) || !isnothing(end_time)
        τ_raw = M[:, 1]
        mask = trues(size(M, 1))
        !isnothing(start_time) && (mask .&= τ_raw .>= start_time)
        !isnothing(end_time) && (mask .&= τ_raw .<= end_time)
        M = M[mask, :]
    end

    ncol = size(M, 2)
    chans = FCSChannel[]

    if isnothing(colspec)
        # default layout: col 1 = τ, cols 2..(1+n_g) = G, cols (2+n_g)..(1+2n_g) = σ
        τ = M[:, 1]
        n_g = min(4, ncol - 1)
        has_σ = ncol >= 1 + 2 * n_g && n_g > 0
        for k in 1:n_g
            σ = has_σ ? M[:, n_g + 1 + k] : nothing
            push!(chans, FCSChannel("G[$k]", τ, M[:, k + 1], σ))
        end
    else
        # explicit colspec: NamedTuple with τ, G, and optionally σ, names
        τ = M[:, colspec.τ]
        g_cols = colspec.G
        σ_cols = hasproperty(colspec, :σ) ? colspec.σ : nothing
        chan_names = hasproperty(colspec, :names) ? colspec.names : ["G[$k]" for k in 1:length(g_cols)]
        for (k, g_col) in enumerate(g_cols)
            σ = (!isnothing(σ_cols) && k <= length(σ_cols)) ? M[:, σ_cols[k]] : nothing
            push!(chans, FCSChannel(chan_names[k], τ, M[:, g_col], σ))
        end
    end

    return FCSData(chans, metadata, String(path))
end


"""
    _read_pqres(path; metadata=Dict{String,Any}())

Read an FCS correlation result file in the PicoQuant Unified Tag Format (.pqres).
Implementation based on the Python library ptufile.

The file is parsed tag-by-tag (each entry is 48 bytes: 32-byte identifier,
4-byte index, 4-byte type code, 8-byte value). Variable-length payloads
(strings, float arrays, blobs) immediately follow their 48-byte entry.
All integers are little-endian.

All parsed tags are merged into the returned `FCSData.metadata` dict with keys
of the form `"TagName[idx]"` (indexed) or `"TagName"` (unindexed).
"""
function _read_pqres(path::AbstractString;
    metadata=Dict{String,Any}(),
    extra_kwargs...
)
    data = read(path)

    length(data) >= 16 || throw(ArgumentError("File too short to be a valid .pqres file: \"$path\""))
    data[1:8] == PQRES  || throw(ArgumentError("Not a valid .pqres file (wrong magic bytes): \"$path\""))

    io = IOBuffer(data)
    skip(io, 16) # 8-byte "magic" (i.e., file type) + 8-byte version string

    tags = Dict{String,Any}()

    while !eof(io)
        id_bytes = read(io, 32)
        length(id_bytes) == 32 || break
        tagid = rstrip(String(id_bytes), '\0')
        tagidx = ltoh(read(io, Int32))
        tagtype = ltoh(read(io, UInt32))
        tagval = ltoh(read(io, Int64))
        
        evalue = if tagtype == TagEmpty8
            nothing
        elseif tagtype == TagBool8
            tagval != 0
        elseif tagtype == TagInt8
            tagval
        elseif tagtype == TagBitSet64
            reinterpret(UInt64, tagval)
        elseif tagtype == TagColor8
            reinterpret(UInt64, tagval)
        elseif tagtype == TagFloat8
            reinterpret(Float64, tagval)
        elseif tagtype == TagDateTime
            days = reinterpret(Float64, tagval)
            DateTime(1899, 12, 30) + Millisecond(round(Int64, days * 86_400_000))
        elseif tagtype == TagAnsiString
            rstrip(String(read(io, Int(tagval))), '\0')
        elseif tagtype == TagWideString
            wbytes = read(io, Int(tagval))
            iseven(length(wbytes)) ?
                String(transcode(UInt8, reinterpret(UInt16, wbytes))) :
                String(wbytes)
        elseif tagtype == TagFloat8Array
            collect(reinterpret(Float64, read(io, Int(tagval))))
        elseif tagtype == TagBinaryBlob
            read(io, Int(tagval))
        else
            tagval  # unknown tag type ⟹ store raw bits
        end

        tagid == "Header_End" && break

        key = tagidx >= 0 ? "$(tagid)[$(tagidx)]" : tagid
        tags[key] = evalue
    end

    chans = _pqres_channels(tags)
    return FCSData(chans, merge(tags, metadata), String(path))
end

# Tag name prefixes for PicoQuant FCS correlation data.
# Full tag names embed the channel index and axis, e.g.:
#   "VarFCSCurve_0X"  → τ (lag times)  for channel 0
#   "VarFCSCurve_0Y"  → G(τ) values    for channel 0
#   "VarFCSWeight_0Y" → weights (1/σ²) for channel 0
const DataTag = "VarFCSCurve_"
const WeightTag = "VarFCSWeight_"

function _pqres_channels(tags::Dict{String,Any})::Vector{FCSChannel}
    τ_by_ch = Dict{Int, Vector{Float64}}()
    G_by_ch = Dict{Int, Vector{Float64}}()
    w_by_ch = Dict{Int, Vector{Float64}}()

    ndata = length(DataTag)
    nweight = length(WeightTag)

    for (k, v) in tags
        v isa Vector{Float64} || continue
        if startswith(k, DataTag)
            suffix = k[ndata+1:end] # e.g. "0X" or "12Y"
            isempty(suffix) && continue
            axis = suffix[end]
            idx_str = suffix[1:end-1]
            cidx = tryparse(Int, idx_str)
            if axis == 'X'
                τ_by_ch[cidx] = v
            elseif axis == 'Y'
                G_by_ch[cidx] = v
            end
        elseif startswith(k, WeightTag)
            suffix = k[nweight+1:end]
            isempty(suffix) && continue
            suffix[end] == 'Y' || continue # X-axis weights are not meaningful
            idx_str = suffix[1:end-1]
            cidx = something(tryparse(Int, idx_str), continue)
            w_by_ch[cidx] = v
        end
    end

    chans = FCSChannel[]
    for cidx in sort(collect(keys(G_by_ch)))
        haskey(τ_by_ch, cidx) || continue
        σ = if haskey(w_by_ch, cidx)
            # weights are 1/σ²; guard against non-positive values with Inf
            @. ifelse(w_by_ch[cidx] > 0, 1.0 / sqrt(w_by_ch[cidx]), Inf) # TODO: not sure if this is how SPT interprets weights. need to double check.
        else
            nothing
        end
        push!(chans, FCSChannel("G[$(cidx + 1)]", τ_by_ch[cidx], G_by_ch[cidx], σ))
    end
    return chans
end