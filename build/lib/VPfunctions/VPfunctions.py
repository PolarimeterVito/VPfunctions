import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
from datetime import datetime as dt

# Function: get_all_filenames
# Description: This function reads all filenames in the script-folder and subfolders with a given file extension
# Parameters:
#   - file_ext: file extension (default value = '.CSV')
# Returns:
#   - all_fn: list with all paths and filenames with the respective file extension
def get_all_filenames(file_ext='.CSV'):
    all_fn = []
    for dirpath, dirnames, filenames in os.walk("."):
        all_fn.extend([os.path.join(dirpath, filename) for filename in filenames if filename.endswith(file_ext)])
    return all_fn

# Function: read_RSRSWP_data
# Description: This function reads the data and relevant header info from a noise measurement data file from the R&S FSWP
# Parameters:
#   - filename: name of the file to be imported
#   - trace: number of the trace to be imported (default value = 1)
#   - sep: separator used in the file (default value = ',')
# Returns:
#   - measType: measurement type of the dataset, either 'PN' or 'AM'
#   - fcarrier: carrier frequency of the measurement (in Hz)
#   - Pcarrier: carrier power of the measurement (in dBm)
#   - measData: np.array with [0] being the Fourier frequency axis and [1] being the PSD values
def read_RSRSWP_data(filename, trace=1, sep=','):
    # Initialize variables
    readlines = 0
    ii = 1
    ind = 0
    start = False
    skiplines = np.Inf
    freq = []
    PSD = []
    measType = 0.
    fcarrier = 0.
    Pcarrier = 0.
    
    # Open the file
    with open(filename, 'r') as f:
        # Read each line in the file
        for line in f:
            lineData = line.strip().split(sep)
            # Extract relevant header info
            if lineData[0] == 'Trace Result':
                measType = lineData[1]
            elif lineData[0] == 'Signal Frequency':
                fcarrier = float(lineData[1])
            elif lineData[0] == 'Signal Level':
                Pcarrier = float(lineData[1])
            # Start reading data when the desired trace is found
            elif lineData[0] == 'Trace' and int(lineData[1]) == trace:
                start = True
            # Get the number of lines to skip and the number of lines to read
            elif lineData[0] == 'Values' and start:
                readlines = int(lineData[1])
                skiplines = ii
            # Read the data
            elif ii > skiplines and ind < readlines:
                freq.append(float(lineData[0]))
                PSD.append(float(lineData[1]))
                ind += 1
            ii += 1
        # Create the np.array with frequency and PSD values
        measData = np.array([freq, PSD])
    
    return measType, fcarrier, Pcarrier, measData

# Function: calc_phaseNoise
# Description: This function calculates the phase noise from L(f)
# Parameters:
#   - measData: data input, format as returned from function 'read_RSRSWP_data'
# Returns:
#   - phaseNoise: np.array with [0] being the Fourier frequency axis and [1] being the PSD values
def calc_phaseNoise(measData):
    phaseNoise = np.array([measData[0], 2 * 10 ** (measData[1] / 10)])
    return phaseNoise

# Function: calc_freqNoise
# Description: This function calculates the frequency noise from L(f)
# Parameters:
#   - measData: data input, format as returned from function 'read_RSRSWP_data'
# Returns:
#   - freqNoise: np.array with [0] being the Fourier frequency axis and [1] being the PSD values
def calc_freqNoise(measData):
    freqNoise = np.array([measData[0], 2 * 10 ** (measData[1] / 10) * measData[0] ** 2])
    return freqNoise

# Function: calc_timingJitter
# Description: This function calculates the timing jitter from L(f)
# Parameters:
#   - measData: data input, format as returned from function 'read_RSRSWP_data'
#   - fcarrier: carrier frequency of the measData (in Hz)
# Returns:
#   - timingJitter: np.array with [0] being the Fourier frequency axis and [1] being the PSD values
def calc_timingJitter(measData, fcarrier):
    timingJitter = np.array([measData[0], 2 * 10 ** (measData[1] / 10) / (2 * np.pi ** 2 * fcarrier ** 2)])
    return timingJitter

# Function: calc_fractFreqNoise
# Description: This function calculates the fractional frequency noise from L(f)
# Parameters:
#   - measData: data input, format as returned from function 'read_RSRSWP_data'
#   - fcarrier: carrier frequency of the measData (in Hz)
# Returns:
#   - fractFreqNoise: np.array with [0] being the Fourier frequency axis and [1] being the PSD values
def calc_fractFreqNoise(measData, fcarrier):
    fractFreqNoise = np.array([measData[0], 2 * 10 ** (measData[1] / 10) * measData[0] ** 2 / fcarrier ** 2])
    return fractFreqNoise

# function calc_Linewidth
# estimates the linewidth from L(f) using the beta-separation line approach
# requirements: import numpy as np
#               function calc_freqNoise
# input:  measData  --> data input, format as returned from function 'read_RSRSWP_data'
# output: intLinewidth --> np.array with [0] being the Fourier frequency axis and [1] being the linewidth values
def calc_Linewidth(measData):
    freqNoise = calc_freqNoise(measData)
    intLinewidth = np.array([measData[0], np.zeros(len(measData[1]))])
    toIntegrate = np.where(freqNoise[1] > 8 * np.log(2) / (np.pi ** 2) * measData[0], freqNoise[1], 0)
    integrated = -np.flip(integrate.cumtrapz(np.flip(toIntegrate), np.flip(measData[0]), initial=0))
    intLinewidth[1] = np.sqrt(8 * np.log(2) * integrated)
    return intLinewidth

# Function: scale_Lf_opt
# Description: This function scales a phase noise measurement in the RF domain up to the optical frequency
# Parameters:
#   - measData: data input, format as returned from function 'read_RSRSWP_data'
#   - wl: wavelength of the investigated laser
#   - fcarrier: carrier frequency of the noise measurement
# Returns:
#   - scaledMeasData: np.array with [0] being the Fourier frequency axis and [1] being the PSD values
def scale_Lf_opt(measData, wl, fcarrier):
    scaledMeasData = np.array([measData[0], measData[1] + 20 * np.log10(sp.constants.c / wl / fcarrier)])
    return scaledMeasData

# Function: normalize_Lf_frep
# Description: This function normalizes a phase noise measurement at a higher harmonic to the laser repetition rate phase noise
# Parameters:
#   - measData: data input, format as returned from function 'read_RSRSWP_data'
#   - fcarrier: carrier frequency of the phase noise measurement
#   - frep: repetition rate of the laser
# Returns:
#   - normMeasData: np.array with [0] being the Fourier frequency axis and [1] being the PSD values
#   - harmonic: integer value of the harmonic
def normalize_Lf_frep(measData, fcarrier, frep):
    harmonic = np.rint(fcarrier / frep)
    normMeasData = np.array([measData[0], measData[1] - 20 * np.log10(harmonic)])
    return normMeasData, int(harmonic)

# Function: calc_rmsRin
# Description: This function calculates the rms RIN of the input signal from the PSD of the input signal
# Parameters:
#   - measData: data input, format as returned from function 'read_RSRSWP_data'
# Returns:
#   - integratedRmsRin: np.array with [0] being the Fourier frequency axis and [1] being the rms RIN values
def calc_rmsRin(measData):
    integratedRmsRin = np.array([measData[0], np.zeros(len(measData[1]))])
    linearized = 10 ** (measData[1] / 10)
    integrated = -np.flip(np.cumtrapz(np.flip(linearized), np.flip(measData[0]), initial=0))
    integratedRmsRin[1] = np.sqrt(2 * integrated) * 100
    return integratedRmsRin

# Function: calc_rmsRin
# Description: This function calculates the rms RIN of the input signal from the PSD of the input signal
# Parameters:
#   - measData: data input, format as returned from function 'read_RSRSWP_data'
# Returns:
#   - integratedRmsRin: np.array with [0] being the Fourier frequency axis and [1] being the rms RIN values
def calc_rmsRin(measData):
    integratedRmsRin = np.array([measData[0], np.zeros(len(measData[1]))])
    linearized = 10 ** (measData[1] / 10)
    integrated = -np.flip(np.cumtrapz(np.flip(linearized), np.flip(measData[0]), initial=0))
    integratedRmsRin[1] = np.sqrt(2 * integrated) * 100
    return integratedRmsRin

# Function: plot_Lf
# Description: This function plots L(f) and saves the plot
# Parameters:
#   - measData: data input, format as returned from function 'read_RSRSWP_data'
#   - fileName: name of the .png file to be saved
def plot_Lf(measData, fileName):
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
# function plot_S_phi
# Description: This function plots S_phi(f) and saves the plot
# Parameters:
#   - measData: data input, format as returned from function 'read_RSRSWP_data'
#   - fileName: name of the .png file to be saved
def plot_S_phi(measData, fileName):
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

# Function: plot_S_nu
# Description: This function plots S_nu(f) and saves the plot
# Parameters:
#   - measData: data input, format as returned from function 'read_RSRSWP_data'
#   - fileName: name of the .png file to be saved
def plot_S_nu(measData, fileName):
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

# function plot_S_y
# Description: This function plots S_y(f) and saves the plot
# Parameters:
#   - measData: data input, format as returned from function 'read_RSRSWP_data'
#   - fcarrier: carrier frequency of the measData (in Hz)
#   - fileName: name of the .png file to be saved
def plot_S_y(measData, fcarrier, fileName):
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

# Function: plot_S_x
# Description: This function plots S_x(f) and saves the plot
# Parameters:
#   - measData: data input, format as returned from function 'read_RSRSWP_data'
#   - fcarrier: carrier frequency of the measData (in Hz)
#   - fileName: name of the .png file to be saved
def plot_S_x(measData, fcarrier, fileName):
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

# Function: plot_Linewidth
# Description: This function plots the frequency noise and linewidth and saves the plot
# Parameters:
#   - measData: data input, format as returned from function 'read_RSRSWP_data'
#   - fileName: name of the .png file to be saved
def plot_Linewidth(measData, fileName):
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

# Function: plot_Lf_AM
# Description: This function plots L(f) and saves the plot for SSB AM noise measurement
# Parameters:
#   - measData: data input, format as returned from function 'read_RSRSWP_data'
#   - fileName: name of the .png file to be saved
def plot_Lf_AM(measData, fileName):
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

# Function: plot_integratedRmsRin
# Description: This function plots S_nu(f) and saves the plot
# Parameters:
#   - measData: data input, format as returned from function 'read_RSRSWP_data'
#   - fileName: name of the .png file to be saved
def plot_integratedRmsRin(measData, fileName):
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

# Function: plot_all_noise_files
# Description: This function plots all noise data in all variants and saves the plots
# Parameters:
#   - all_fn: list of filenames to process
#   - frep: repetition frequency of the measurement (in Hz)
#   - wl: wavelength of the measurement (in nm)
#   - normalize_to_frep: boolean flag indicating whether to normalize to frep
#   - do_linewidth_estimation: boolean flag indicating whether to perform linewidth estimation
def plot_all_noise_files(all_fn, frep, wl, normalize_to_frep=True, do_linewidth_estimation=False):
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

# Function: read_RPFP_plot_data
# Description: This function reads data from a RPFP plot file and extracts the specified trace
# Parameters:
#   - filename: name of the file to read
#   - trace: the trace number to extract
def read_RPFP_plot_data(filename, trace):
    # Initialize variables
    extract_data = False
    x_data = []
    y_data = []
    
    # Open the file with UTF-8 encoding
    with open(filename, 'r',encoding='utf-8') as f:
        # Read each line in the file
        for line in f:
            line_data = re.split(r'[,\s]\s*', line)
            # Start data extraction when the correct trace is found
            if line_data[1] == 'plot' and line_data[2] == str(trace):
                extract_data = True
            elif (line_data[1] == 'plot' and line_data[2] != str(trace)) or line_data[0] == '':
                extract_data = False
            elif extract_data:
                x_data.append(float(line_data[0]))
                y_data.append(float(line_data[1]))
    return np.array([x_data, y_data])

# Function: read_TLPM_data
# Description: This function reads the data from a Thorlabs power meter data file with n saved power traces
# Parameters:
#   - filename: name of the file to be imported
# Output:
#   - measData: np.array with [smpl,timestamp,pwr_1,pwr_2,pwr_3, ..., pwr_n]
def read_TLPM_data(filename):
    data_start = False
    smpl = []
    timestamp = []
    with open(filename, 'r') as f:
        for line in f:
            lineData = re.split(r'\s*,\s*', line)
            if lineData[0] == 'Samples':
                data_start = True
                # Initialize power data arrays
                pwr_entries = len(lineData)-4
                pwr_data = np.empty(pwr_entries,dtype=object)
                for i in range(pwr_entries):
                    pwr_data[i] = []
            elif data_start:
                smpl.append(int(lineData[0])) 
                timestamp.append(dt.combine(dt.strptime(lineData[1], r'%m/%d/%Y').date(),dt.strptime(lineData[2], '%H:%M:%S.%f').time()))
                for i in range(pwr_entries):
                    pwr_data[i].append(float(lineData[i+3]))

    # Calculate relative time in minutes
    rel_time_min = [(x-timestamp[0]).total_seconds()/60 for x in timestamp]
    measData = np.array([smpl,timestamp,rel_time_min,*pwr_data])

    return measData

# Function: read_AQ6374_data
# Description: This function reads the data and relevant header info from a noise measurement data file from the R&S FSWP
# Parameters:
#   - filename: name of the file to be imported
# Output:
#   - measData: np.array with [0] being the Wavelength axis and [1] being the PSD values
#   - res: resolution of the measurement in nm as a float value
def read_AQ6374_data(filename):
    smpl_size = 0
    data_start = False
    wl = []
    pwr_data = []
    res = 0.0
    with open(filename, 'r') as f:
        for line in f:
            lineData = re.split(',|\n', line)
            if lineData[0] == '\"RESLN\"':
                res = float(lineData[1])
            elif lineData[0] == '\"SMPL\"':
                smpl_size = int(lineData[1])
            elif lineData[0] == '\"[TRACE DATA]\"':
                data_start = True
            elif data_start and len(wl) <= smpl_size:
                wl.append(float(lineData[0]))
                pwr_data.append(float(lineData[1]))
        measData = np.array([wl,pwr_data])
    return measData, res

# Function: read_Redstone_data
# Description: This function reads the data and relevant header info from a noise measurement data file from the R&S FSWP
# Parameters:
#   - filename: name of the file to be imported
# Output:
#   - measData: np.array with [0] being the Wavelength axis and [1] being the PSD values
def read_Redstone_data(filename):
    data_start = False
    wl = []
    pwr_data = []
    with open(filename, 'r') as f:
        for line in f:
            lineData = re.split(r'\s*[;\n]\s*', line)
            if lineData[0] == r'[EndOfFile]':
                break
            elif data_start:
                wl.append(float(lineData[0]))
                pwr_data.append(float(lineData[1]))
            elif lineData[0] == r'[Data]':
                data_start = True

        measData = np.array([wl,pwr_data])
    return measData

# Function: norm_spectrum
# Description: This function normalizes the input dataset to its maximum value
# Parameters:
#   - OSA_data: data input, format as returned from function 'read_AQ6374_data'
# Output:
#   - normalized_data: dataset normalized to its maximum value
def norm_spectrum(OSA_data):
    return np.array([OSA_data[0], OSA_data[1] / max(OSA_data[1])])

# Function: calc_FWHM
# Description: This function calculates the full width at half maximum (FWHM) of the input spectrum
# Parameters:
#   - OSA_data: data input, format as returned from function 'read_AQ6374_data'
# Output:
#   - FWHM: FWHM of the input spectrum
def calc_FWHM(OSA_data):
    leftIndex = np.where(OSA_data[1,:] >= 0.5*max(OSA_data[1]))[0][0]
    rightIndex = np.where(OSA_data[1,:] >= 0.5*max(OSA_data[1]))[0][-1]
    FWHM = OSA_data[0,rightIndex] - OSA_data[0,leftIndex]
    print('FWHM of the measured spectrum is: {:2.2f} nm'.format(FWHM))
    return FWHM

# Function: calc_CWL
# Description: This function calculates the central wavelength (CWL) of the input spectrum
# Parameters:
#   - OSA_data: data input, format as returned from function 'read_AQ6374_data'
# Output:
#   - CWL: central wavelength of the input spectrum
def calc_CWL(OSA_data):
    CWL = OSA_data[0,OSA_data.argmax(axis = 1)[1]];
    print('CWL of the measured spectrum is: {:4.2f} nm'.format(CWL))
    return CWL

# Function: read_RSRSWP_RF_data
# Description: This function reads the data and relevant header info from an RF data file from the R&S FSWP
# Parameters:
#   - filename: name of the file to be imported
#   - sep: separator used in the file (default value = ',')
# Output:
#   - measData: np.array with [0] being the Fourier frequency axis and [1] being the PSD values
def read_RSRSWP_RF_data(filename, sep=';'):
    readlines = 0
    ii = 1
    ind = 0
    skiplines = np.Inf
    freq = []
    PSD = []
    with open(filename, 'r') as f:
        for line in f:
            lineData = re.split(';|\n', line)
            if lineData[0] == 'Values':
                readlines = int(lineData[1])
                skiplines = ii
            if ii > skiplines and ind < readlines:
                freq.append(float(lineData[0]))
                PSD.append(float(lineData[1]))
                ind += 1
            ii += 1
        measData = np.array([freq, PSD])
    return measData

# Function to correct the BB measurements for the detector BGD and the normalize to the carrier power
# Parameters:
# - data_signal: numpy.ndarray, signal data array
# - data_bgd: numpy.ndarray, background data array
# - voltage: float, voltage value in volts
# - resistance: float, resistance value in ohms (default is 50 ohms)
# Returns:
# - data_dBc: numpy.ndarray, corrected data in dBc
def correct_BB_meas(data_signal, data_bgd, voltage, resistance=50):
    if data_signal.shape != data_bgd.shape:
        raise ValueError("Signal and background data must have the same size (RBW)!")
    
    data_signal_lin = 10**(data_signal[1] / 10)
    data_bgd_lin = 10**(data_bgd[1] / 10)
    carrier_pwr = (voltage**2) / resistance * 1e3  # in mW, to match the units of the data
    corrected_data_lin = data_signal_lin - data_bgd_lin
    normalized_data_lin = corrected_data_lin / carrier_pwr
    normalized_data_dBc = 10 * np.log10(normalized_data_lin)
    data_dBc = np.vstack((data_signal[0], normalized_data_dBc))
    
    return data_dBc