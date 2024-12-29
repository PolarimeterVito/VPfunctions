import numpy as np
import scipy.constants as sp
import scipy.integrate as integrate

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