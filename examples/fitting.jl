using FCSFitting
using DelimitedFiles
using PrettyTables

filepath = raw"examples/fcs_sample.txt"
data = readdlm(filepath);

start_idx = 50
end_idx = 300

times = data[start_idx:end_idx,1]
corr1 = data[start_idx:end_idx,2]
corr2 = data[start_idx:end_idx,3]
corr3 = data[start_idx:end_idx,4]
corr4 = data[start_idx:end_idx,5]
stddev1 = data[start_idx:end_idx,6]
stddev2 = data[start_idx:end_idx,7]
stddev3 = data[start_idx:end_idx,8]
stddev4 = data[start_idx:end_idx,9];

fig, fit, scale = fcs_plot(fcs_3d, times, corr3, [2e-7, 1.0, 0.0, 5, 1e-7, 0.1], wt=1 ./ stddev3.^2, lower = [1e-8, 0.9, -1e-5, 1, 1e-8, 0.0], upper = [1e-6, 1.1, 1e-5, 20, 1e-4, 0.5], diffusivity=5e-11)
fig

parameter_list = [
    "Diffusion time [s]",
    "G(0)",
    "Offset",
    "Structure factor",
    "Dynamic time 1 [s]",
    "Dynamic fraction 1"
]

column_labels = [
    latex_cell"Parameters",
    latex_cell"Values",
    latex_cell"Std. Dev.",
]

pretty_table(hcat(parameter_list, FCSFitting.parameters(fit, scale), FCSFitting.errors(fit, scale)); backend = Val(:latex))