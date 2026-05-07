
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

# TODO: need to complete implementation
"""
    _read_txt(path; start_idx=nothing, end_idx=nothing, start_time=nothing, end_time=nothing,
              delimeter=" ", linebreak="\r\n", filling_values=0.0, colspec=nothing,
              metadata=Dict{String,Any}(), extra_kwargs...)

Read FCS correlation data from a whitespace- or delimiter-separated text file.

# Column layout
- `colspec=nothing` (default): column 1 = τ, columns 2–5 = G[1]–G[4],
  columns 6–9 = σ[1]–σ[4] (only when all four σ columns are present).
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
        mask  = trues(size(M, 1))
        !isnothing(start_time) && (mask .&= τ_raw .>= start_time)
        !isnothing(end_time)   && (mask .&= τ_raw .<= end_time)
        M = M[mask, :]
    end

    ncol  = size(M, 2)
    chans = FCSChannel[]

    if isnothing(colspec)
        # default layout: col 1 = τ, cols 2..(1+n_g) = G, cols (2+n_g)..(1+2n_g) = σ
        τ   = M[:, 1]
        n_g = min(4, ncol - 1)
        has_σ = ncol >= 1 + 2 * n_g && n_g > 0
        for k in 1:n_g
            σ = has_σ ? M[:, n_g + 1 + k] : nothing
            push!(chans, FCSChannel("G[$k]", τ, M[:, k + 1], σ))
        end
    else
        # explicit colspec: NamedTuple with τ, G, and optionally σ, names
        τ          = M[:, colspec.τ]
        g_cols     = colspec.G
        σ_cols     = hasproperty(colspec, :σ)     ? colspec.σ     : nothing
        chan_names = hasproperty(colspec, :names)  ? colspec.names : ["G[$k]" for k in 1:length(g_cols)]
        for (k, g_col) in enumerate(g_cols)
            σ = (!isnothing(σ_cols) && k <= length(σ_cols)) ? M[:, σ_cols[k]] : nothing
            push!(chans, FCSChannel(chan_names[k], τ, M[:, g_col], σ))
        end
    end

    return FCSData(chans, metadata, String(path))
end


"""
    _read_pqres(path; metadata=Dict{String,Any}())
"""
function _read_pqres(path::AbstractString;
    metadata=Dict{String,Any}(),
    extra_kwargs...
)
    return FCSData(FCSChannel[], metadata, String(path)) #TODO: stub
end