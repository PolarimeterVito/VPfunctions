import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sp

from .manipulators import calc_phaseNoise, calc_freqNoise, calc_fractFreqNoise, calc_timingJitter, calc_Linewidth, calc_rmsRin, normalize_Lf_frep, scale_Lf_opt
from .readers import read_RSRSWP_data




def plot_Lf(measData, fileName):
    """
    Plots the L(f) data on a semilogarithmic plot and saves the plot as a PNG file.

    Parameters:
    measData (numpy.ndarray): A 2D array where the first row contains the Fourier frequencies (Hz) 
                              and the second row contains the corresponding L(f) values (dBc/Hz).
    fileName (str): The base name of the file to save the plot as. The plot will be saved in a 
                    directory named 'plots' with the filename format '<fileName>_Lf.png'.

    Returns:
    None
    """
    fig1, ax1 = plt.subplots()

    ax1.semilogx(measData[0],measData[1])
    ax1.set_xlabel(r'Fourier Frequency (Hz)', fontsize = 20)
    ax1.set_ylabel(r'L(f) (dBc/Hz)', fontsize = 20)
    ax1.set_xticks(10**np.arange(0,8,1))
    ax1.set_yticks(np.arange(round(np.amin(measData[1]),-1),round(np.amax(measData[1]),-1),20))
    ax1.set_xlim([measData[0,0], measData[0,len(measData[0]) - 1]])
    ax1.set_ylim([np.amin(measData[1]), np.amax(measData[1]) + 5])
    ax1.grid(which = 'major', color = 'gray', linestyle = '-', linewidth = 0.75)
    ax1.grid(which = 'minor', color = 'lightgray', linestyle = '--', linewidth = 0.5)
    [x.set_linewidth(1.5) for x in ax1.spines.values()]
    fig1.tight_layout()

    os.makedirs('plots', exist_ok=True)
    save_name = os.path.join('plots', fileName + '_Lf.png')
    fig1.savefig(save_name)
    print('Succesfully plotted '+ save_name)
    return

def plot_S_phi(measData, fileName):
    """
    Plots the phase noise data on a semilogarithmic plot and saves the plot as a PNG file.

    Parameters:
    measData (numpy.ndarray): A 2D array where the first row contains the Fourier frequencies (Hz) 
                              and the second row contains the corresponding L(f) values (dBc/Hz).
    fileName (str): The base name of the file to save the plot as. The plot will be saved in a 
                    directory named 'plots' with the filename format '<fileName>_S_phi.png'.

    Returns:
    None
    """
    PN = calc_phaseNoise(measData)
    fig1, ax1 = plt.subplots()

    ax1.loglog(PN[0],PN[1])
    ax1.set_xlabel(r'Fourier Frequency (Hz)', fontsize = 20)
    ax1.set_ylabel(r'$PN-PSD\,(rad^2/Hz)$', fontsize = 20)
    ax1.set_xticks(10**np.arange(0,8,1))
    ax1.set_xlim([PN[0,0], PN[0,len(PN[0]) - 1]])
    ax1.set_ylim([np.amin(PN[1]) * 0.8, np.amax(PN[1]) * 1.5])
    ax1.grid(which = 'major', color = 'gray', linestyle = '-', linewidth = 0.75)
    ax1.grid(which = 'minor', color = 'lightgray', linestyle = '--', linewidth = 0.5)
    [x.set_linewidth(1.5) for x in ax1.spines.values()]
    fig1.tight_layout()

    os.makedirs('plots', exist_ok=True)
    save_name = os.path.join('plots', fileName + '_S_phi.png')
    fig1.savefig(save_name)
    print('Succesfully plotted '+ save_name)
    return

def plot_S_nu(measData, fileName):
    """
    Plots the frequency noise data on a semilogarithmic plot and saves the plot as a PNG file.

    Parameters:
    measData (numpy.ndarray): A 2D array where the first row contains the Fourier frequencies (Hz) 
                              and the second row contains the corresponding L(f) values (dBc/Hz).
    fileName (str): The base name of the file to save the plot as. The plot will be saved in a 
                    directory named 'plots' with the filename format '<fileName>_S_nu.png'.

    Returns:
    None
    """
    FN = calc_freqNoise(measData)
    fig1, ax1 = plt.subplots()

    ax1.loglog(FN[0],FN[1])
    ax1.set_xlabel(r'Fourier Frequency (Hz)', fontsize = 20)
    ax1.set_ylabel(r'$FN-PSD\,(Hz^2/Hz)$', fontsize = 20)
    ax1.set_xticks(10**np.arange(0,8,1))
    ax1.set_xlim([FN[0,0], FN[0,len(FN[0]) - 1]])
    ax1.set_ylim([np.amin(FN[1]) * 0.8, np.amax(FN[1])*1.5])
    ax1.grid(which = 'major', color = 'gray', linestyle = '-', linewidth = 0.75)
    ax1.grid(which = 'minor', color = 'lightgray', linestyle = '--', linewidth = 0.5)
    [x.set_linewidth(1.5) for x in ax1.spines.values()]
    fig1.tight_layout()
        
    os.makedirs('plots', exist_ok=True)
    save_name = os.path.join('plots', fileName + '_S_nu.png')
    fig1.savefig(save_name)
    print('Succesfully plotted '+ save_name)
    return

def plot_S_y(measData, fcarrier, fileName):
    """
    Plots the fractional frequency noise data on a semilogarithmic plot and saves the plot as a PNG file.

    Parameters:
    measData (numpy.ndarray): A 2D array where the first row contains the Fourier frequencies (Hz) 
                              and the second row contains the corresponding L(f) values (dBc/Hz).
    fileName (str): The base name of the file to save the plot as. The plot will be saved in a 
                    directory named 'plots' with the filename format '<fileName>_S_y.png'.

    Returns:
    None
    """
    FFN = calc_fractFreqNoise(measData, fcarrier)
    fig1, ax1 = plt.subplots()

    ax1.loglog(FFN[0],FFN[1])
    ax1.set_xlabel(r'Fourier Frequency (Hz)', fontsize = 20)
    ax1.set_ylabel(r'FFN-PSD (1/Hz)', fontsize = 20)
    ax1.set_xticks(10**np.arange(0,8,1))
    ax1.set_xlim([FFN[0,0], FFN[0,len(FFN[0]) - 1]])
    ax1.set_ylim([np.amin(FFN[1]) * 0.8, np.amax(FFN[1])*1.5])
    ax1.grid(which = 'major', color = 'gray', linestyle = '-', linewidth = 0.75)
    ax1.grid(which = 'minor', color = 'lightgray', linestyle = '--', linewidth = 0.5)
    [x.set_linewidth(1.5) for x in ax1.spines.values()]
    fig1.tight_layout()
        
    os.makedirs('plots', exist_ok=True)
    save_name = os.path.join('plots', fileName + '_S_y.png')
    fig1.savefig(save_name)
    print('Succesfully plotted '+ save_name)
    return

def plot_S_x(measData, fcarrier, fileName):
    """
    Plots the timing jitter data on a semilogarithmic plot and saves the plot as a PNG file.

    Parameters:
    measData (numpy.ndarray): A 2D array where the first row contains the Fourier frequencies (Hz) 
                              and the second row contains the corresponding L(f) values (dBc/Hz).
    fileName (str): The base name of the file to save the plot as. The plot will be saved in a 
                    directory named 'plots' with the filename format '<fileName>_S_x.png'.

    Returns:
    None
    """
    TJ = calc_timingJitter(measData, fcarrier)
    fig1, ax1 = plt.subplots()

    ax1.loglog(TJ[0],TJ[1])
    ax1.set_xlabel(r'Fourier Frequency (Hz)', fontsize = 20)
    ax1.set_ylabel(r'$TJ-PSD/,(s^2/Hz)$', fontsize = 20)
    ax1.set_xticks(10**np.arange(0,8,1))
    ax1.set_xlim([TJ[0,0], TJ[0,len(TJ[0]) - 1]])
    ax1.set_ylim([np.amin(TJ[1]) * 0.8, np.amax(TJ[1])*1.5])
    ax1.grid(which = 'major', color = 'gray', linestyle = '-', linewidth = 0.75)
    ax1.grid(which = 'minor', color = 'lightgray', linestyle = '--', linewidth = 0.5)
    [x.set_linewidth(1.5) for x in ax1.spines.values()]
    fig1.tight_layout()
        
    os.makedirs('plots', exist_ok=True)
    save_name = os.path.join('plots', fileName + '_S_x.png')
    fig1.savefig(save_name)
    print('Succesfully plotted '+ save_name)
    return

def plot_Linewidth(measData, fileName):
    """
    Plots the frequency noise data and the corresponding linewidth estimated with the beta-separation line approach on a semilogarithmic plot and saves the plot as a PNG file.

    Parameters:
    measData (numpy.ndarray): A 2D array where the first row contains the Fourier frequencies (Hz) 
                              and the second row contains the corresponding L(f) values (dBc/Hz).
    fileName (str): The base name of the file to save the plot as. The plot will be saved in a 
                    directory named 'plots' with the filename format '<fileName>_LW.png'.

    Returns:
    None
    """
    FN = calc_freqNoise(measData)
    betaSepLine = 8*np.log(2)/(np.pi**2)*FN[0]
    LW = calc_Linewidth(measData)
    fig1, ax1 = plt.subplots(2, 1, sharex = True)

    ax1[0].loglog(FN[0],FN[1],FN[0],betaSepLine)
    ax1[0].set_ylabel(r'$FN-PSD\,(Hz^2/Hz)$', fontsize = 15)
    ax1[0].set_xticks(10**np.arange(0,8,1))
    ax1[0].set_xlim([FN[0,0], FN[0,len(FN[0]) - 1]])
    ax1[0].set_ylim([np.amin(FN[1]) * 0.8, np.amax(FN[1])*1.5])
    ax1[0].grid(which = 'major', color = 'gray', linestyle = '-', linewidth = 0.75)
    ax1[0].grid(which = 'minor', color = 'lightgray', linestyle = '--', linewidth = 0.5)
    [x.set_linewidth(1.5) for x in ax1[0].spines.values()]
        
    ax1[1].semilogx(LW[0],LW[1])
    ax1[1].set_xlabel(r'Fourier Frequency (Hz)', fontsize = 15)
    ax1[1].set_ylabel(r'Linewidth (Hz)', fontsize = 15)
    ax1[1].set_xticks(10**np.arange(0,8,1))
    ax1[1].set_xlim([LW[0,0], LW[0,len(FN[0]) - 1]])
    ax1[1].set_ylim([0, np.amax(LW[1])*1.2])
    ax1[1].grid(which = 'major', color = 'gray', linestyle = '-', linewidth = 0.75)
    ax1[1].grid(which = 'minor', color = 'lightgray', linestyle = '--', linewidth = 0.5)
    [x.set_linewidth(1.5) for x in ax1[1].spines.values()]
    fig1.tight_layout()
        
    os.makedirs('plots', exist_ok=True)
    save_name = os.path.join('plots', fileName + '_LW.png')
    fig1.savefig(save_name)
    print('Succesfully plotted '+ save_name)
    return

def plot_Lf_AM(measData, fileName):
    """
    Plots the L(f) of an AM noise measurement on a semilogarithmic plot and saves the plot as a PNG file.

    Parameters:
    measData (numpy.ndarray): A 2D array where the first row contains the Fourier frequencies (Hz) 
                              and the second row contains the corresponding L(f) values (dBc/Hz).
    fileName (str): The base name of the file to save the plot as. The plot will be saved in a 
                    directory named 'plots' with the filename format '<fileName>_Lf_AM.png'.

    Returns:
    None
    """
    fig1, ax1 = plt.subplots()

    ax1.semilogx(measData[0],measData[1])
    ax1.set_xlabel(r'Fourier Frequency (Hz)', fontsize = 20)
    ax1.set_ylabel(r'SSB AM-PSD (dBc/Hz)', fontsize = 20)
    ax1.set_xticks(10**np.arange(0,8,1))
    ax1.set_xlim([measData[0,0], measData[0,len(measData[0]) - 1]])
    ax1.set_ylim([np.amin(measData[1]), np.amax(measData[1]) + 5])
    ax1.grid(which = 'major', color = 'gray', linestyle = '-', linewidth = 0.75)
    ax1.grid(which = 'minor', color = 'lightgray', linestyle = '--', linewidth = 0.5)
    [x.set_linewidth(1.5) for x in ax1.spines.values()]
    fig1.tight_layout()

    os.makedirs('plots', exist_ok=True)
    save_name = os.path.join('plots', fileName + 'Lf_AM.png')
    fig1.savefig(save_name)
    print('Succesfully plotted '+ save_name)
    return

def plot_integratedRmsRin(measData, fileName):
    """
    Plots the L(f) of an AM noise measurement and the corresponding cumulative RMS RIN on a semilogarithmic plot and saves the plot as a PNG file.

    Parameters:
    measData (numpy.ndarray): A 2D array where the first row contains the Fourier frequencies (Hz) 
                              and the second row contains the corresponding L(f) values (dBc/Hz).
    fileName (str): The base name of the file to save the plot as. The plot will be saved in a 
                    directory named 'plots' with the filename format '<fileName>_intRmsRin.png'.

    Returns:
    None
    """
    integratedRmsRin = calc_rmsRin(measData)
    fig1, ax1 = plt.subplots(2, 1, sharex = True)

    ax1[0].semilogx(measData[0],measData[1])
    ax1[0].set_ylabel(r'SSB AM-PSD (dBc/Hz)', fontsize = 15)
    ax1[0].set_xticks(10**np.arange(0,8,1))
    ax1[0].set_xlim([measData[0,0], measData[0,len(measData[0]) - 1]])
    ax1[0].set_ylim([np.amin(measData[1]), np.amax(measData[1]) + 5])
    ax1[0].grid(which = 'major', color = 'gray', linestyle = '-', linewidth = 0.75)
    ax1[0].grid(which = 'minor', color = 'lightgray', linestyle = '--', linewidth = 0.5)
    [x.set_linewidth(1.5) for x in ax1[0].spines.values()]
    
    ax1[1].semilogx(integratedRmsRin[0],integratedRmsRin[1])
    ax1[1].set_xlabel(r'Fourier Frequency (Hz)', fontsize = 15)
    ax1[1].set_ylabel(r'INTEGRATED RMS RIN (%)', fontsize = 15)
    ax1[1].set_xticks(10**np.arange(0,8,1))
    ax1[1].set_xlim([integratedRmsRin[0,0], integratedRmsRin[0,len(integratedRmsRin[0]) - 1]])
    ax1[1].set_ylim([0, np.amax(integratedRmsRin[1])*1.2])
    ax1[1].grid(which = 'major', color = 'gray', linestyle = '-', linewidth = 0.75)
    ax1[1].grid(which = 'minor', color = 'lightgray', linestyle = '--', linewidth = 0.5)
    [x.set_linewidth(1.5) for x in ax1[1].spines.values()]
    fig1.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    save_name = os.path.join('plots', fileName + '_intRmsRin.png')
    fig1.savefig(save_name)
    print('Succesfully plotted '+ save_name)
    return

def plot_all_noise_files(all_fn, frep, wl, normalize_to_frep=True, do_linewidth_estimation=False):
    """
    Plots various noise characteristics from a list of measurement files.

    This function reads measurement data from each file in `all_fn`, determines the type of measurement (PN or AM),
    and plots various noise characteristics. If the measurement type is 'PN' and `normalize_to_frep` is True, the data
    is normalized to the repetition rate of the laser. If `do_linewidth_estimation` is True, the linewidth is estimated
    using the beta-separation-line approach. The function generates and saves plots for phase noise, frequency noise,
    and other related characteristics. If the measurement type is 'AM', it plots amplitude noise characteristics.

    Parameters:
    all_fn (list of str): List of file paths to the measurement data files.
    frep (float): Repetition rate of the laser in Hz.
    wl (float): Wavelength of the laser in meters.
    normalize_to_frep (bool, optional): If True, normalize the data to the repetition rate of the laser. Default is True.
    do_linewidth_estimation (bool, optional): If True, perform linewidth estimation using the beta-separation-line approach. Default is False.

    Returns:
    None
   
    """
    for x in all_fn:
        measType, fcarrier, Pcarrier, measData = read_RSRSWP_data(x)
        currFileName = os.path.splitext(os.path.basename(x))[0]
        print('File: ' + x)
        print('Type of measurement: ' + measType)
        print('Carrier power: {:1f} dBm'.format(Pcarrier))
            
        if measType == 'PN':
            if normalize_to_frep:
                plotData, harmonic = normalize_Lf_frep(measData, fcarrier, frep)
                RFfreq = frep
                print('Normalizing dataset to the repetition rate of the laser.')
                print('Repetition rate of the laser: {:1f} MHz'.format(frep*1e-6))
                print('Carrier frequency of the measurement: {:2f} MHz'.format(fcarrier*1e-6))
                print('Harmonic = {:d}'.format(harmonic))
            else:
                plotData = measData
                RFfreq = fcarrier
                print('Dataset not normalized to laser repetiton rate.')
                print('Carrier frequency of the measurement: {:2f} MHz'.format(fcarrier*1e-6))

            plot_Lf(plotData, currFileName)
            plot_S_phi(plotData, currFileName)
            plot_S_nu(plotData, currFileName)
            plot_S_x(plotData, RFfreq, currFileName)
            plot_S_y(plotData, RFfreq, currFileName)

            if do_linewidth_estimation:
                LWdata = scale_Lf_opt(measData, wl, fcarrier)
                print('Linewidth estimated with the beta-separation-line approach.')
                print('Carrier frequency of the measurement: {:2f} MHz'.format(fcarrier*1e-6))
                print('Optical frequency: {:2f} THz'.format(sp.constants.c/wl*1e-12))
                print('Scaling factor: {:e}'.format(sp.constants.c/wl/fcarrier))
                print('Estimated linewidth: {:e} Hz'.format(calc_Linewidth(LWdata)[1,0]))
                plot_Linewidth(LWdata, currFileName)
        elif measType == 'AM':
            print('File: ' + x)
            plot_Lf_AM(measData, currFileName)
            plot_integratedRmsRin(measData, currFileName)
        else:
            print('Unknown measurement type: ' + measType)
            print('No plotting done!')
        
        print('\n')  
        plt.close('all')
    print('Plotting finished!')
    return

def set_plot_params(style: object = 'presentation', dpi: object = 600) -> object:
    """
    Set the parameters for plotting with matplotlib based on the specified style.

    Parameters:
    style (str): The style of the plot. Options are 'optica_two_column', 'optica_one_column', 
                 'presentation'. Defaults to 'presentation'.
    dpi (int): The resolution of the plot in dots per inch. Defaults to 600.

    Returns:
    None
    """
    if style == 'optica_two_column':
        # Settings for two column optica journal:
        figW = 5.75
        figH = figW/2
    elif style == 'optica_one_column':
        # Settings for single column optica journal:
        figW = 3.25
        figH = figW/2
    elif style == 'presentation':
        # Settings for presentation style:
        figW = 40/2.54
        figH = 20/2.54
    else:
        print('Unknown style, using presentation style!')
        figW = 40/2.54
        figH = 20/2.54
    plt.rcParams.update({       'axes.grid': True,
                                'axes.grid.axis': 'both',
                                'axes.grid.which': 'both',
                                'axes.linewidth': 1.0,
                                'lines.linewidth': 1.5,
                                'lines.markersize': 5,
                                'lines.markeredgewidth': 0.75,
                                'lines.solid_capstyle': 'round',
                                'lines.solid_joinstyle': 'round',
                                'lines.dash_capstyle': 'round',
                                'lines.dash_joinstyle': 'round',
                                'axes.linewidth': 0.75,
                                'axes.labelpad': 1,
                                'font.size': 18,
                                'axes.labelsize': 15,  # standard 7
                                'font.sans-serif': 'Arial',
                                'axes.unicode_minus': False,
                                'mathtext.default': 'regular',
                                'mathtext.fontset': 'custom',
                                'mathtext.rm': 'Bitstream Vera Sans',        # $\mathrm{Text}$
                                'mathtext.it': 'Bitstream Vera Sans:italic', # $\mathit{Text}$
                                'mathtext.bf': 'Bitstream Vera Sans:bold',   # $\mathbf{Text}$
                                'legend.fontsize': 15, #standard 7
                                'legend.numpoints': 1,
                                'legend.columnspacing': 0.3,
                                'legend.fancybox': True,
                                'legend.labelspacing': 0.2,
                                'legend.handlelength': 1, # 1 standard
                                'legend.handletextpad': 0.5,
                                'legend.markerscale':0.9,
                                'legend.frameon': True,
                                'legend.labelspacing':0.02,
                                'savefig.dpi': dpi,
                                #'text.latex.preamble': r'\usepackage{mathrsfs}\usepackage{amsmath}',
                                #'text.latex.preamble': r'\usepackage{unicode-math}\setmathfont{XITS Math}',
                                'text.usetex': False,
                                'xtick.labelsize': 15,
                                'xtick.major.size': 5,        # major tick size in points
                                'xtick.minor.size': 2.5,      # minor tick size in points
                                'xtick.major.width': 1.25,    # major tick width in points
                                'xtick.minor.width': 1.25,    # minor tick width in points
                                'ytick.labelsize': 15,
                                'ytick.major.size': 5,      # major tick size in points
                                'ytick.minor.size': 2.5,      # minor tick size in points
                                'ytick.major.width': 1.25,    # major tick width in points
                                'ytick.minor.width': 1.25,    # minor tick width in points
                                'figure.figsize': (figW, figH)})
    return