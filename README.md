# FCSFitting

*Fitting fluorescence correlation spectroscopy (FCS) data in Julia*

[![CI](https://github.com/LabGradinaru/FCSFitting.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/LabGradinaru/FCSFitting.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![codecov.io](https://codecov.io/github/LabGradinaru/FCSFitting.jl/branch/main/graph/badge.svg?token=ZH9L011XZQ)](http://codecov.io/github/LabGradinaru/FCSFitting.jl/branch/main)

**FCSFitting** provides a lightweight, composable toolkit for modeling and fitting FCS autocorrelation curves. The nonlinear least‑squares backend is currently `LsqFit.jl`, with an intended move to `JuMP.jl` in the near future. Optional package extensions enable file I/O, tables, and publication‑quality plots.

> **Status:** private pre‑release repository; APIs subject to change. Target Julia ≥ **1.10**


## Features

* Generic 2D/3D diffusion models with optional anomalous diffusion (α) and multi‑component mixtures
* Additive dynamic terms (e.g., exponential kinetic terms) and optional baseline offset
* Parameter scaling and bounds handling for numerically stable fits
* Clean separation of **model specification** vs **fit configuration**
* Optional extensions for:
  * Reading delimited text files (`DelimitedFiles.jl`)
  * Pretty tabular summaries (`PrettyTables.jl`)
  * LaTeX‑style labels in plots (`LaTeXStrings.jl`)
  * Interactive/publication plots (`CairoMakie.jl`)


## Installation

Until the package is registered, install via a local checkout or a private Git URL.

### Option A — local path (recommended for development)

```julia
julia> ]
pkg> activate --shared fcs
pkg> add CairoMakie LaTeXStrings DelimitedFiles PrettyTables IJulia
pkg> dev /absolute/path/to/FCSFitting.jl
pkg> precompile
```

### Option B — private Git URL

```julia
julia> ]
pkg> activate --shared fcs
pkg> add CairoMakie LaTeXStrings DelimitedFiles PrettyTables IJulia
pkg> dev git@github.com:LabGradinaru/FCSFitting.jl.git  # or https
pkg> precompile
```

> **Tip:** use `dev` (instead of `add`) to track local changes during development.


## Environments & Jupyter kernel (VS Code/Jupyter)

If you use VS Code or Jupyter, it’s convenient to create a dedicated environment and kernel.

1. **Create/activate a shared environment**

```julia
julia> ]
pkg> activate --shared fcs
pkg> add CairoMakie LaTeXStrings DelimitedFiles PrettyTables IJulia
pkg> dev /absolute/path/to/FCSFitting.jl
pkg> precompile
```

2. **Install a Jupyter kernel that points at this env**

```julia
julia> using IJulia
julia> IJulia.installkernel("Julia (@fcs)"; env=Dict("JULIA_PROJECT" => "@fcs"))
```

3. **Select the kernel** in VS Code: `Ctrl+Shift+P` → *Notebook: Select Notebook Kernel* → *Select Another Kernel…* → *Jupyter Kernels* → **Julia (@fcs)**.


## Quick start

```julia
using FCSFitting

# Example: 3D normal diffusion with one kinetic (exponential) term and an offset.
diffusivity = 5e-11 # m^2/s
offset = 0.0
spec = FCSModelSpec(dim = d3, anom = none, offset = offset, diffusivity = diffusivity)

# Synthetic example parameters: [g0, n_exp_terms, τD, τ_dyn, K_dyn]
initial_parameters = [1.0, 5.0, 2e-7, 1e-7, 0.1]
lower_bounds = [0.9, 1.0, 1e-8,  1e-8, 0.0]
upper_bounds = [1.1, 20.0, 1e-6, 1e-4, 0.5]

# t: lag‑time vector (s); g: experimental correlation values
# Example stub (replace with real data):
t = range(1e-7, 1e-2; length=256)
g = model(spec, initial_parameters, t) .+ 0.02 .* randn(length(t))

fit, scale = fcs_fit(spec, t, g, initial_parameters; lower = lower_bounds, upper = upper_bounds)
println(fit)
```


### Reading your own data (via extension)

If your data live in a delimited file (CSV/TSV), load `DelimitedFiles` **before** `FCSFitting` to enable the extension. The files are assumed to be in the order (column-wise): lag times, data, standard deviations (optional), which is then organized into `FCSChannel` objects:

```julia
using DelimitedFiles, FCSFitting
data = read_fcs(filepath; start_idx = 20, end_idx = 300);
fit, scale = fcs_fit(spec, data.channel[1].τ, data.channel[1].G, initial_parameters; lower = lower_bounds, upper = upper_bounds)
```


### Plotting (via CairoMakie extension)

```julia
using CairoMakie, LaTeXStrings, FCSFitting

channel = FCSChannel("sample", t, g, nothing)

fig, fit, scales = fcs_plot(spec, channel, initial_parameters)
save("fit.png", fig)
```


## Models and parameters

`FCSModelSpec` declares model structure; numerical values are supplied via the parameter vector. The intended order for the parameter vector arguments can be accessed via 
```julia
using FCSFitting

coarse_parameter_order = expected_parameter_names(spec)
precise_parameter_order = expected_parameter_names(spec, initial_parameters)
```
which returns a `Vector{String}` containing the intended argument order. In general, the order should be

1. Current correlation, $G (0)$
2. (Optional) Correlation offset, $G (\infty)$
3. (Optional; if `dim=:d3`) Structure factor, $\kappa$
4. Characteristic diffusion times, $\tau_{D,i} = w_{0,i}^2 / 4D_i$ (OR the beam width $w_{0,i}$ if the diffusivity is provided)
5. (If `spec.anom != :none`) Anomalous exponents, $\alpha_i$
6. (If `spec.n_diff > 1`) Diffusion population fractions, $f_i$
7. Dynamic lifetimes, $\tau_{\mathrm{dyn}, j}$
8. Dynamic population fractions, $T_j$

The generic form of the model being fit is
$$\hat{G}(t) = G (0) \sum_{i = 1}^N f_i K (t, \tau_{D,i}, \alpha_i) \times \prod_{j = 1}^M \left[ 1 + \sum_{k = 1}^{n_j} T_{j,k} \left( e^{- t / \tau_{\mathrm{dyn}, j, k}} - 1 \right) \right] + G (\infty), $$
where $K$ is the diffusive kernel being used for the fit.
The (in)dependence of the dynamic components being fit (i.e., if they are multiplicative or additive) is dictated by the `ics` parameter of `FCSModelSpec`.
For instance, if you wish to have two **independent** dynamic processes, each with one state, one would set `ics=[1,1]`, amounting to a dynamic contribution
$$\left[ 1 + T_{1,1} \left( e^{- t / \tau_{\mathrm{dyn},1,1}} - 1 \right) \right] \left[ 1 + T_{2,1} \left( e^{- t / \tau_{\mathrm{dyn},2,1}} - 1 \right) \right]$$
to the correlation. On the other hand, if the two states are understood to be **dependent** (e.g., accounting for two triplet states, etc.) then one would set `ics = [2]` such that the contribution is now
$$1 + T_{1,1} \left( e^{- t / \tau_{\mathrm{dyn},1,1}} - 1 \right) + T_{1,2} \left( e^{- t / \tau_{\mathrm{dyn},1,2}} - 1 \right).$$


## Troubleshooting

* **Extensions aren’t active**: ensure you loaded e.g. `CairoMakie` **before** `FCSFitting` in the same session.
* **MethodError on model spec**: check that your parameter vector matches the model’s expected ordering.
* **Slow/unstable fits**: provide reasonable bounds; use `scale` output from `fcs_fit` for diagnostics; reduce parameter correlations by fixing known values if possible.
* **Pkg can’t find the repo**: double‑check the path/URL and that you have permission to the private repository.


## Contributing

Issues and PRs are welcome. Please include a minimal reproducer and specify the Julia version. For larger contributions, open an issue first to discuss design/API.