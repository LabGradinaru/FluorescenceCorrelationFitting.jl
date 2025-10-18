A package for fitting previously-correlated fluorescence correlation spectroscopy (FCS) data.

# Usage details

If you are new to Julia, read the "Environment and Jupyter kernel creation" section below first.

For most users, the primary access point should be the notebook `examples/fitting.ipynb` which illustrates the key utilities of the package.
From dataset-to-dataset, the `filepath` variable in cell 2 should be changed, as well as the `model` in cell 3 and the initial parameter estimates and their bounds.

Changes should be made to the initial parameter values which are specified by a vector `p`.
The parameter vector order depends on which model is selected and the user should refer to the docstrings for detailed references.
Briefly, for 2-dimensional FCS, `fcs_2d`,
*   `p[1]` → g0; the zero-lag autocorrelation
*   `p[2]` → offset; the offset of the correlation from 0
*   `p[3]` → τD; the diffusion time
*   `p[4:m]` → τ_dyn; the dynamic lifetimes
*   `p[m+1:N]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime.

And for 3-dimensional FCS, `fcs_3d`,
*   `p[1]` → g0; the zero-lag autocorrelation
*   `p[2]` → offset; the offset of the correlation from 0
*   `p[3]` → κ; the structure factor `κ = z0/w0`
*   `p[4]` → τD; the diffusion time
*   `p[5:m]` → τ_dyn; the dynamic lifetimes
*   `p[m+1:N]` → K_dyn; the fraction corresponding of the population corresponding to the dynamic lifetime.

Each of these parameters are scaled to $\mathcal{O}(1)$ based on their initial guess, provided they are non-zero, for the values to stabilize the fitting routine.
For calibration purposes, the diffusivity can be constrained by specifying the keyword argument `diffusivity` in `fcs_plot` or `fcs_fit`, which will result in the beam waist `w0` being the first parameter in the vector, in place of the diffusion time τD.

The value of `m` in either case is inferred from the length of the parameter vector.
The "dynamic lifetimes" and their fractions are kept rather general to allow them to encapsulate a number of phenomena including photophysical dark states (e.g., triplets and blinking), PET, and molecule dynamics which are broadly captured by an exponential kernel in the autocorrelation.
To specify which elements of the parameter vector correspond to which physical phenomena, the keyword argument `ics`, short for "independent components" is present.
For instance, if we are to have two triplet states with lifetimes $\tau_1$ and $\tau_2$, we expect that they are dependent on each other in the sense that if a fluorophore is in triplet state 1 it cannot be in triplet state 2.
The result is that the contribution to the autocorrelation is given by the sum 
$$1 + T_1 \left( e^{- t / \tau_1} - 1 \right) + T_2 \left( e^{- t / \tau_2} - 1 \right)$$
where $T_1$ and $T_2$ are the fraction of the population in the corresponding triplet state.
On the other hand, if we have one triplet state and one PET site, we expect these events to be independent, amounting to a contribution
$$\left( 1 + T e^{- t / \tau_\mathrm{tr}} - T \right) \left( 1 + Q e^{- t / \tau_\mathrm{pet}} - Q \right)$$
where $Q$ is the fraction of the observed time spent undergoing PET dynamics.
In the first case, we specify `ics = [2]` since we wish for the two components to be dependent upon each other.
In the second, one may write `ics = [1,1]`, although this is taken as the base case so such a specification is optional.

# Environment and Jupyter kernel creation

Once you have the Julia language installed (https://julialang.org/install/), since FCSFitting is not a public repository at the time of writing, the best way to interface with it is to create a new global environment.
Moreover, if you are using VSCode as an IDE, you will need to instantiate a new kernel associated with this environment.
This section is dedicated to showing users new to Julia how this can be achieved.

Open terminal or command propt and type the command `julia` then hit `Enter`.
This will launch the Julia REPL, your primary access point to the Julia language.
To interface with packages, type a right square bracket, `]`, which will bring you to the Pkg REPL.
To create a global or shared environment named `fcs`, type the command
`pkg> activate --shared fcs`.
Currently, `FCSFitting` has four "weak dependencies" which allow for one to easily plot and read data and create tables.
These can be installed in `fcs` with the command
`fcs> add CairoMakie, LaTeXStrings, DelimitedFiles, PrettyTables, IJulia`.
We will discuss the last, fifth, package shortly.
After the packages are installed, we want to add the FCSFitting repository as `fcs> add <path>/<to>/<FCSFitting>`, where the path is replaced with your local, absolute path to the FCSFitting directory.
If you intend to keep up with the most recent versions of FCSFitting, you can use `dev` instead of `add` in the previous command.
Finally, type `fcs> precompile` to precompile the environment.

Next, we need to make a kernel for the notebooks which use FCSFitting to be run in.
From the Pkg REPL, use backspace to return to the Julia REPL and type the command
`julia> using IJulia; IJulia.installkernel("Julia (@fcs)"; env=Dict("JULIA_PROJECT" => "@fcs"))`, 
amounting to a new kernel being made which uses the `fcs` environment we just created.

Finally, when you wish to run a notebook such as `examples/fitting.ipynb`, go to VSCode and, before running, enter `Ctrl+Shift+P` then type into the search bar `Notebook: Select notebook kernel`.
After hitting `Enter`, this will bring up a dropdown menu whereby you should select `Select Another Kernel... -> Jupyter Kernels -> Julia (@fcs)`.
Now the notebook will run with the kernel we just created!