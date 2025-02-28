import numpy as np
import matplotlib.pyplot as plt

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

    plt.plot(lag_times, data,label=label, color=color)
    plt.plot(lag_times, fit.best_fit, '--', label = label + " Fit", color=lighten_color(color,1.3))
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

class DictTable(dict):
    # Overridden dict class which takes a dict in the form {'a': 2, 'b': 3},
    # and renders an HTML Table in IPython Notebook.
    def _repr_html_(self):
        html = ["<table width=40%>"]
        for key, value in iter(self.items()):
            html.append("<tr>")
            html.append(f"<td>{key}</td>")
            scaled_value = SCALES[key] * value if key in SCALES.keys() else value
            html.append(f"<td>{scaled_value:.4f}</td>")
            html.append(f"<td>{UNITS[key]}</td>")
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

def interpret_fit(fit, r0 = None):
    params_dict = fit.best_values
    if r0 is not None or 'r0' in params_dict.keys():
        r0 = r0 if r0 is not None else params_dict['r0']
        if 'D' in params_dict.keys() and 'tauD' not in params_dict.keys():
            params_dict['tauD'] = r0**2 / (4 * params_dict['D'])
        if 'tauD' in params_dict.keys() and 'D' not in params_dict.keys():
            params_dict['D'] = r0**2 / (4 * params_dict['tauD'])
        if 'n' and 's' in params_dict.keys() and 'C' not in params_dict.keys():
            params_dict['C'] = params_dict['n'] / (np.pi**1.5 * params_dict['s'] * r0**3)

    params_dict['Reduced Chi-Squared'] = fit.redchi
    params_dict['AIC'] = fit.aic
    params_dict['R Squared'] = fit.rsquared
    d = DictTable(params_dict)
    return d