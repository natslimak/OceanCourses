import pylab as plt
import numpy as np
import scipy.signal as ss

def makeplots(wind, waves, structure, response, timeInfo, color, **kwargs):

    # Check if the axis has been passed otherwise initialize plot
    ax = kwargs.get("ax", np.zeros(2))
    if not ax.any():
        _, ax = plt.subplots(4,2, sharex='col')
    
    Filter = wind["t"] > timeInfo["TTrans"]
    FilterResponse = response["t"] > timeInfo["TTrans"]

    ax[0,0].plot(wind["t"], wind["V_hub"], color=color, linewidth=2.5)
    ax[0,0].set_ylabel(r"V [$m/s$]", fontsize=10)
    ax[0,0].tick_params(labelsize=10)
    
    f, _, S = freqSpectrum(wind["t"], wind["V_hub"])
    ax[0,1].plot(f, S, color=color, linewidth=2.5)
    ax[0,1].set_ylabel(r"PSD [$m^2/s^2 / Hz$]", fontsize=10)
    ax[0,1].tick_params(labelsize=10)
    
    ax[1,0].plot(waves["t"], waves["eta"], color=color)
    ax[1,0].set_ylabel(r"$\eta$ [$m$]", fontsize=10)
    
    f, _, S = freqSpectrum(waves["t"],  waves["eta"])
    ax[1,1].plot(f, S, color=color)
    ax[1,1].set_ylabel(r"PSD [$m^2 / Hz$]", fontsize=10)    

    ax[2,0].plot(response["t"], response["x1"], color=color)
    ax[2,0].set_ylabel(r"Surge [$m$]", fontsize=10)
    
    f, _, S = freqSpectrum(response["t"][FilterResponse], response["x1"][FilterResponse])
    ax[2,1].plot(f, S, color=color)
    ax[2,1].set_ylabel(r"PSD [$m^2 / Hz$]", fontsize=10)    
    ax[2,1].axvline(structure["fnat"][0], 0., np.max(S), color='k', linestyle='--',label='_nolegend_')

    ax[3,0].plot(response["t"], np.rad2deg(response["x5"]), color=color)    
    ax[3,0].set_ylabel(r"Pitch [rad]", fontsize=10)
    
    f, _, S = freqSpectrum(response["t"][FilterResponse], response["x5"][FilterResponse])
    ax[3,1].plot(f, S, color=color)
    ax[3,1].set_ylabel(r"PSD [rad^2 / Hz]", fontsize=10)
    ax[3,1].axvline(structure["fnat"][1], 0., np.max(S), color='k', linestyle='--',label='_nolegend_')
    
    ax[3,1].set_xlim([0., timeInfo["fHighCut"]])
    [ax_.grid(True) for ax_ in ax.ravel()]
    
    plt.tight_layout()
    
    return ax
    

def makeplotsMonopile(wind, waves, loads, response, timeInfo, color, **kwargs):

    # Check if the axis has been passed otherwise initialize plot
    ax = kwargs.pop("ax", np.zeros(2))
    if not ax.any():
        _, ax = plt.subplots(4,2, sharex='col')
    
    Filter = response["t"] > timeInfo["TTrans"]
    
    ax[0,0].plot(wind["t"], wind["V_hub"], color=color, **kwargs)
    ax[0,0].set_ylabel("V[m/s]")
    
    f, _, S = freqSpectrum(wind["t"], wind["V_hub"])
    ax[0,1].plot(f, S, color=color, **kwargs)
    ax[0,1].set_ylabel("PSD [m^2/s^2 / Hz]")
    
    ax[1,0].plot(waves["t"], waves["eta"], color=color, **kwargs)
    ax[1,0].set_ylabel("eta[m]")
    
    f, _, S = freqSpectrum(waves["t"],  waves["eta"])
    ax[1,1].plot(f, S, color=color, **kwargs)
    ax[1,1].set_ylabel("PSD [m^2 / Hz]")

    ax[2,0].plot(response["t"], response["alpha"], color=color, **kwargs)
    ax[2,0].set_ylabel("alpha[m]")
    
    f, _, S = freqSpectrum(response["t"], response["alpha"])
    ax[2,1].plot(f, S, color=color, **kwargs)
    ax[2,1].set_ylabel("PSD [m^2 / Hz]")    

    ax[3,0].plot(loads["t"], np.rad2deg(loads["M"]), color=color, **kwargs)
    ax[3,0].set_ylabel("moment[Nm]")
    
    f, _, S = freqSpectrum(loads["t"], loads["M"])
    ax[3,1].plot(f, S, color=color, **kwargs)
    ax[3,1].set_ylabel("PSD [(Nm)^2 / Hz]")
        
    ax[3,1].set_xlim([0., timeInfo["fHighCut"]])
    [ax_.grid(True) for ax_ in ax.ravel()]
    
    plt.tight_layout()
    
    return ax

def freqSpectrum(t, x, flagMean=False):
    
    # Takes a time vector and a signal and returns the one-sided Fourier 
    # coefficients obtained with FFT, as well as the PSD function

    # flagmean=False: f[0]=df, a[0]=mean(x) is removed
    # flagmean=True: f[0]=0,  a[0]=mean(x) is kept
    
    if len(t) == 1:
        raise ValueError("The signal has only one element.")
    
    else:
        Tmax = t[-1] - t[0]
        df = Tmax**-1
        
        if flagMean:
            f = np.arange(len(t))*df
        else:
            f = (np.arange(1,len(t)+1)*df)
            
        a = np.fft.fft(x) / len(t)
        
        dt = t[1] - t[0]        
        f_nyq = 1./2./dt        
        a[f > f_nyq] = 0.
        
        if flagMean:
            a[1:] = 2*a[1:]
        else:
            a = 2*a[1:len(f)]
            a = np.append(a, 0.)
            
        # PSD function
        
        S = np.abs(a)**2 / (2*df)
        
        return f, a, S
        
        
def recolor_lines_by_time(fig, t_split=600.0, color_before='red', color_after='blue'):
    """
    Recolor plotted lines in the left column of a figure grid based on a time split.
    
    Parameters:
    -----------
    fig : numpy array of axes
        Figure axes array (typically 4x2)
    t_split : float
        Time value at which to split colors (default: 600.0 seconds)
    color_before : str
        Color for data before t_split (default: 'red')
    color_after : str
        Color for data after t_split (default: 'blue')
    """
    axs = np.array(fig)
    left_col = axs[:, 0].ravel()
    
    for ax in left_col:
        orig_lines = list(ax.get_lines())
        for line in orig_lines:
            x = np.asarray(line.get_xdata())
            y = np.asarray(line.get_ydata())
            if x.size == 0:
                continue

            mask1 = x <= t_split
            mask2 = x > t_split

            lw = line.get_linewidth()
            ls = line.get_linestyle()
            mk = line.get_marker()
            msz = line.get_markersize()
            alpha = line.get_alpha()
            orig_label = line.get_label()

            if mask1.any():
                label1 = orig_label if (mask1.any() and not mask2.any()) else None
                ax.plot(x[mask1], y[mask1], color=color_before, linewidth=lw, linestyle=ls, 
                       marker=mk, markersize=msz, alpha=alpha, label=label1)

            if mask2.any():
                label2 = orig_label if (mask2.any() and not mask1.any()) else None
                ax.plot(x[mask2], y[mask2], color=color_after, linewidth=lw, linestyle=ls, 
                       marker=mk, markersize=msz, alpha=alpha, label=label2)

            try:
                line.remove()
            except Exception:
                pass