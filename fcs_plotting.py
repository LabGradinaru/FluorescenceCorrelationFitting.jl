import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

STANDARD_YLABEL = r"$G(\tau)$"
STANDARD_XLABEL = r"$\tau$ [s]"
SCALES = {"n" : 1, "D" : 1e12, "C" : 1e9 / (1000 * 6.02214e23), "r0" : 1e9, "tauD" : 1e3, "s" : 1, "offset" : 1, "tautr1" : 1e6, "T1" : 1, "tautr2" : 1e6, "T2" : 1, "Reduced Chi-Squared" : 1, "AIC" : 1, "R Squared" : 1}
UNITS = {"n" : None, "D" : "um2/s", "C" : "nM", "r0" : "nm", "tauD" : "ms", "s" : None, "offset" : None, "tautr1" : "us", "T1" : None, "tautr2" : "us", "T2" : None, "Reduced Chi-Squared" : None, "AIC" : None, "R Squared" : None}

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def model_plot(lag_times, data, fit, 
               label = "", color = 'b',
               xlabel=STANDARD_XLABEL, 
               ylabel=STANDARD_YLABEL):
    
    
    fig = plt.figure(1)
    frame1 = fig.add_axes((.1,.3,.8,.6))

    plt.scatter(lag_times, data,label=label,facecolors = 'none',color=color)
    plt.plot(lag_times, fit.best_fit,  label = label + " Fit", color=lighten_color(color,1.3))
    plt.ylabel(ylabel)
    plt.semilogx()
    plt.legend()

    frame1.set_xticklabels([])
    plt.grid()

    frame2 = fig.add_axes((.1,.1,.8,.175))
    plt.plot(lag_times, fit.residual, color=lighten_color(color,1.3))
    plt.ylabel('Residuals')
    plt.xlabel(xlabel)
    plt.semilogx()
    plt.grid()
    plt.show()

def cc_model_plot(lag_times, data1,fit1,data2,fit2,data3,fit3,OVCF_g,OVCF_r, 
               label = "", color = 'b',
               xlabel=STANDARD_XLABEL, 
               ylabel=STANDARD_YLABEL):
    
    fig = plt.figure(1)
    frame1 = fig.add_axes((.1,.3,.8,.6))

    plt.scatter(lag_times, data1/OVCF_g,label="CF1",facecolors = 'none',color="#8DBF2E")
    plt.plot(lag_times, fit1.best_fit/OVCF_g,  label = "CF1 Fit", color=lighten_color("#8DBF2E"))
    plt.scatter(lag_times, data2/OVCF_r,label="CF2",facecolors = 'none',color="#DC4633")
    plt.plot(lag_times, fit2.best_fit/OVCF_r,  label = "CF2 Fit", color=lighten_color("#DC4633"))
    plt.scatter(lag_times, data3,label="CF3",facecolors = 'none',color="#36454F")
    plt.plot(lag_times, fit3.best_fit,  label = "CF3 Fit", color=lighten_color("#36454F"))
    plt.ylabel(ylabel)
    plt.semilogx()
    plt.legend()

    frame1.set_xticklabels([])
    plt.grid()

    frame2 = fig.add_axes((.1,.1,.8,.175))
    plt.plot(lag_times, fit1.residual, color=lighten_color("#8DBF2E"))
    plt.plot(lag_times, fit2.residual, color=lighten_color("#DC4633"))
    plt.plot(lag_times, fit3.residual, color=lighten_color("#36454F"))
    plt.ylabel('Residuals')
    plt.xlabel(xlabel)
    plt.semilogx()
    plt.grid()
    plt.show()

class DictTable(dict):
    # Writes two dictionaries to an IPython table
    def _repr_html_(self):
        html = ["<table width=15%>"]
        html.append("<tr><td>Metric</td><td>Score</td><\tr>")
        for key, value in iter(self.items()):
            html.append("<tr>")
            html.append(f"<td>{key}</td>")
            scaled_value = SCALES[key] * value if key in SCALES.keys() else value
            html.append(f"<td>{scaled_value:.4f}</td>")
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

def fit_evaluation(fit):
    scores_dict = {}

    scores_dict['Reduced Chi-Squared'] = fit.redchi
    scores_dict['AIC'] = fit.aic
    scores_dict['R Squared'] = fit.rsquared
    d = DictTable(scores_dict)
    return d

def autocorr(ts, max_lag, normalize):
    N = ts.size
    avg = np.average(ts)
    
    fluct = ts - avg
    num_elements = max_lag if max_lag <= N else N
    results = np.ones(num_elements, dtype=ts.dtype)
    if normalize:
        results *= 1/np.var(fluct)

    for i in range(num_elements):
        results[i] *= np.sum(fluct[i:] * fluct[:(N-i)])/N
    return results

def residual_autocorrelation(fit, max_lag = 50, normalize=True, color='b'):
    residuals = fit.residual
    N = residuals.size
    fig = plt.figure(1)

    plt.stem(autocorr(residuals, max_lag, normalize), linefmt=color)
    plt.axhline(2 / np.sqrt(N), linestyle='--', color='red')
    plt.axhline(-2 / np.sqrt(N), linestyle='--', color='red')
    plt.ylabel(r'$\hat{\rho} (h)$')
    plt.xlabel(r'$h$')
    plt.grid()
    plt.show()

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

def residuals_histogram(fit, num_bins = 20):
    residuals = fit.residual
    freqs, edges , _ = plt.hist(residuals, num_bins, label='Residuals')
    bin_centres = (edges[:-1] + edges[1:])/2

    model = Model(gaussian)
    result = model.fit(freqs, x=bin_centres, amp=residuals.size/2, cen=0, wid=1)
    best_model_values = tuple(result.values.values())

    smoothed_domain = np.linspace(np.min(residuals), np.max(residuals), 1000)
    plt.plot(smoothed_domain, gaussian(smoothed_domain, *best_model_values), '-', color='red', label='Fitted Gaussian')

    plt.ylabel('Frequency')
    plt.xlabel('Residual Value')
    plt.legend()
    plt.show()