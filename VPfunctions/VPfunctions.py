import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
from datetime import datetime as dt

def get_all_filenames(file_ext='.CSV'):
    """
    Retrieve all filenames with a specific extension from the current directory and its subdirectories.

    Parameters:
        file_ext (str): The file extension to filter by. Default is '.CSV'.

    Returns:
        list: A list of full file paths that match the specified file extension.
    """
    all_fn = []
    for dirpath, dirnames, filenames in os.walk("."):
        all_fn.extend([os.path.join(dirpath, filename) for filename in filenames if filename.endswith(file_ext)])
    return all_fn

def read_RSRSWP_data(filename, trace=1, sep=','):
    """
    Reads RSRSWP data from a specified file and returns measurement type, 
    carrier frequency, carrier power, and measurement data.

    Parameters:
    filename (str): The path to the file containing the RSRSWP data.
    trace (int): The trace number to read data from. Default is 1.
    sep (str): The delimiter used in the file. Default is ','.

    Returns:
    tuple: A tuple containing:
        - measType (str): The type of measurement.
        - fcarrier (float): The carrier frequency in Hz.
        - Pcarrier (float): The carrier power in dBm.
        - measData (np.array): A 2D numpy array with Fourier frequency and PSD values.
    """
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

def calc_phaseNoise(measData):
    """
    Calculate phase noise from L(f) measurement data.

    Parameters:
    measData (list or array-like): A list or array containing measurement data. 
                                   The first element is the Fourier frequency in Hz, 
                                   and the second element is L(f) in dBc/Hz.

    Returns:
    numpy.ndarray: An array where the first element is the Fourier frequency in Hz and 
                   the second element is the phase noise PSD in rad^2/Hz.
    """
    phaseNoise = np.array([measData[0], 2 * 10 ** (measData[1] / 10)])
    return phaseNoise

def calc_freqNoise(measData):
    """
    Calculate frequency noise from L(f) measurement data.

    Parameters:
    measData (list or array-like): A list or array containing measurement data. 
                                   The first element is the Fourier frequency in Hz, 
                                   and the second element is L(f) in dBc/Hz.

    Returns:
    numpy.ndarray: An array where the first element is the Fourier frequency in Hz and 
                   the second element is the frequency noise PSD in Hz^2/Hz.
    """
    freqNoise = np.array([measData[0], 2 * 10 ** (measData[1] / 10) * measData[0] ** 2])
    return freqNoise

def calc_timingJitter(measData, fcarrier):
    """
    Calculate the timing jitter from L(f) measurement data.

    Parameters:
    measData (list or array-like): A list or array containing measurement data. 
                                   The first element is the Fourier frequency in Hz, 
                                   and the second element is L(f) in dBc/Hz.

    Returns:
    numpy.ndarray: An array where the first element is the Fourier frequency in Hz and 
                   the second element is the timing jitter PSD in s^2/Hz.
    """
    timingJitter = np.array([measData[0], 2 * 10 ** (measData[1] / 10) / (2 * np.pi ** 2 * fcarrier ** 2)])
    return timingJitter

def calc_fractFreqNoise(measData, fcarrier):
    """
    Calculate the fractional frequency noise from L(f) measurement data.

    Parameters:
    measData (list or array-like): A list or array containing measurement data. 
                                   The first element is the Fourier frequency in Hz, 
                                   and the second element is L(f) in dBc/Hz.

    Returns:
    numpy.ndarray: An array where the first element is the Fourier frequency in Hz and 
                   the second element is the fractional frequency noise PSD in 1/Hz.
    """
    fractFreqNoise = np.array([measData[0], 2 * 10 ** (measData[1] / 10) * measData[0] ** 2 / fcarrier ** 2])
    return fractFreqNoise

def calc_Linewidth(measData):
    """
    Estimate the linwidth from L(f) using the beta-separation line approach.

    Parameters:
    measData (array-like): A 2D array where the first element is the Fourier frequency
                           data in Hz and the second element is the corresponding
                           PSD data in dBc/Hz.

    Returns:
    numpy.ndarray: A 2D array where the first element is the Fourier frequency in Hz
                   and the second element is the accumulated linewidth.
    """
    freqNoise = calc_freqNoise(measData)
    intLinewidth = np.array([measData[0], np.zeros(len(measData[1]))])
    toIntegrate = np.where(freqNoise[1] > 8 * np.log(2) / (np.pi ** 2) * measData[0], freqNoise[1], 0)
    integrated = -np.flip(integrate.cumtrapz(np.flip(toIntegrate), np.flip(measData[0]), initial=0))
    intLinewidth[1] = np.sqrt(8 * np.log(2) * integrated)
    return intLinewidth

def scale_Lf_opt(measData, wl, fcarrier):
    """
    Scales a measured L(f) input to the optical frequency.

    Parameters:
    measData (array-like): A 2D array where the first element is the Fourier frequency
                           in Hz and the second element is the corresponding
                           PSD data in dBc/Hz.
    wl (float): The optical wavelength to which is scaled in m.
    fcarrier (float): The carrier frequency of the L(f) data in Hz.

    Returns:
    numpy.ndarray: A 2D array where the first element is the Fourier frequency in Hz
                   and the second element is the accumulated linewidth.
    """
    scaledMeasData = np.array([measData[0], measData[1] + 20 * np.log10(sp.constants.c / wl / fcarrier)])
    return scaledMeasData

def normalize_Lf_frep(measData, fcarrier, frep):
    """
    Scales a measured L(f) input to the fundmental repetition rate.

    Parameters:
    measData (array-like): A 2D array where the first element is the Fourier frequency
                           in Hz and the second element is the corresponding
                           PSD data in dBc/Hz.
    fcarrier (float): The carrier frequency of the L(f) data in Hz.
    frep (float): The repetition rate of the laser in Hz.

    Returns:
    numpy.ndarray: A 2D array where the first element is the Fourier frequency in Hz
                   and the second element is the accumulated linewidth.
    """
    harmonic = np.rint(fcarrier / frep)
    normMeasData = np.array([measData[0], measData[1] - 20 * np.log10(harmonic)])
    return normMeasData, int(harmonic)

def calc_rmsRin(measData):
    """
    Calculate the root mean square (RMS) relative intensity noise (RIN) from L(f) data.

    Parameters:
    measData (array-like): A 2D array where the first element is the Fourier frequency
                           in Hz and the second element is the corresponding
                           PSD data in dBc/Hz.

    Returns:
    numpy.ndarray: A 2D array where the first row is the Fourier frequency in Hz and the second row is the
                   cumulative RMS RIN values as a percentage.
    """
    integratedRmsRin = np.array([measData[0], np.zeros(len(measData[1]))])
    linearized = 10 ** (measData[1] / 10)
    integrated = -np.flip(np.cumtrapz(np.flip(linearized), np.flip(measData[0]), initial=0))
    integratedRmsRin[1] = np.sqrt(2 * integrated) * 100
    return integratedRmsRin

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

def read_RPFP_plot_data(filename, trace):
    """
    This functions reads the data from a RP fiber power plot file and extracts the specified trace.
    Parameters:
        filename (str): The path to the file containing the plot data.
        trace (int): The trace number to extract data for.
    Returns:
        np.ndarray: A 2D NumPy array where the first row contains the x data and
                    the second row contains the y data.
    """
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

def read_TLPM_data(filename):
    """
    Reads the data from a Thorlabs Power Meter data file with n saved power traces.

    Parameters:
        filename (str): The path to the CSV file containing the TLPM data.

    Returns:
        numpy.ndarray: A numpy array containing the following data:
            - Sample numbers (list of int)
            - Timestamps (list of datetime objects)
            - Relative time in minutes (list of float)
            - Power data (list of lists of float)
    """
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

def read_AQ6374_data(filename):
    """
    Reads data from an AQ6374 optical spectrum analyzer file.

    Parameters:
    filename (str): The path to the file containing the AQ6374 data.

    Returns:
    tuple: A tuple containing:
        - measData (numpy.ndarray): A 2D array where the first row contains the wavelength data and the second row contains the power data.
        - res (float): The resolution of the measurement.
    """
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

def read_Redstone_data(filename):
    """
    Reads data from a Redstone optical spectrum analyzer file.

    Parameters:
    filename (str): The path to the file containing the Redstone data.

    Returns:
    numpy.ndarray: A 2D NumPy array where the first row contains wavelengths
                   and the second row contains power data.
    """
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

def norm_spectrum(OSA_data):
    """
    Normalize the spectrum data.

    This function takes in a 2D array where the first element is the wavelength data and the second element is the intensity data. 
    It normalizes the intensity data by dividing it by the maximum intensity value.

    Parameters:
    OSA_data (list or numpy.ndarray): A 2D array with wavelength data as the first element and intensity data as the second element.

    Returns:
    numpy.ndarray: A 2D array with the original wavelength data and the normalized intensity data.
    """
    return np.array([OSA_data[0], OSA_data[1] / max(OSA_data[1])])

def calc_FWHM(OSA_data, printFWHM = True):
    """
    Calculate the Full Width at Half Maximum (FWHM) of the measured spectrum.

    Parameters:
    OSA_data (numpy.ndarray): A 2D array where the first row contains the wavelength data (in nm) and the second row contains the intensity data.
    printFWHM (bool, optional): If True, prints the FWHM value. Default is True.

    Returns:
    float: The FWHM of the measured spectrum in nm.
    """
    leftIndex = np.where(OSA_data[1,:] >= 0.5*max(OSA_data[1]))[0][0]
    rightIndex = np.where(OSA_data[1,:] >= 0.5*max(OSA_data[1]))[0][-1]
    FWHM = OSA_data[0,rightIndex] - OSA_data[0,leftIndex]
    if printFWHM:
        print('FWHM of the measured spectrum is: {:2.2f} nm'.format(FWHM))
    return FWHM

def calc_CWL(OSA_data, printCWL =True):
    """
    Calculate the center wavelength (CWL) of the measured spectrum. This is a simple function that returns the wavelength at the maximum intensity value.

    Parameters:
    OSA_data (numpy.ndarray): A 2D array where the first row contains the wavelength data (in nm) and the second row contains the intensity data.
    printCWL (bool, optional): If True, prints the FWHM value. Default is True.

    Returns:
    float: The CWL of the measured spectrum in nm.
    """
    CWL = OSA_data[0,OSA_data.argmax(axis = 1)[1]]
    if printCWL:
        print('CWL of the measured spectrum is: {:4.2f} nm'.format(CWL))
    return CWL

def read_RSRSWP_RF_data(filename, sep=';'):
    """
    Reads Rohde&Schwarz RSWP RF data from a file and returns it as a NumPy array.

    The function reads a file containing frequency and PSD (Power Spectral Density) values separated by a specified delimiter.

    Parameters:
    filename (str): The path to the file containing the data.
    sep (str, optional): The delimiter used in the file to separate values. Default is ';'.

    Returns:
    numpy.ndarray: A 2D array where the first row contains frequency values and the second row contains PSD values.
    """
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

def correct_BB_meas(data_signal, data_bgd, voltage, resistance=50):
    """
    Corrects a baseband (BB) measurement by subtracting the detector background noise and normalizing the result to the carrier power.

    Parameters:
    data_signal (numpy.ndarray): 2D array where the first row is Fourier frequency in Hz and the second row is the signal power in dBm.
    data_bgd (numpy.ndarray): 2D array where the first row is Fourier frequency in Hz and the second row is the background noise power in dBm.
    voltage (float): The voltage of the carrier signal in volts.
    resistance (float, optional): The resistance in ohms. Default is 50 ohms.

    Returns:
    numpy.ndarray: 2D array where the first row is Fourier frequency and the second row is the corrected and normalized signal power in dBc.

    Raises:
    ValueError: If the shape of data_signal and data_bgd do not match.
    """
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