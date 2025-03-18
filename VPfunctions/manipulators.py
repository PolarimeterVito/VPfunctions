import numpy as np
import scipy as sp
from numpy.typing import NDArray
from typing import Union, Sequence, Tuple

def calc_phaseNoise(
        measData: Union[NDArray[np.float64], Sequence[Sequence[float]]]
    ) -> NDArray[np.float64]:
    """
    Calculate phase noise from L(f) measurement data.

    Parameters:
        measData (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - The first row (measData[0]): Fourier frequency in Hz.
            - The second row (measData[1]): L(f) in dBc/Hz.

    Returns:
        NDArray[np.float64]: A 2D NumPy array with dtype float64 and shape (2, n).
            - Row 0: Fourier frequency in Hz.
            - Row 1: Phase noise PSD in rad^2/Hz.

    Raises:
        ValueError: If the input data does not have shape (2, n).
    """
    # Convert input to a NumPy array with dtype float64
    if not (isinstance(measData, np.ndarray) and measData.dtype == np.float64):
        measData = np.asarray(measData, dtype=np.float64)

    # Validate the structure of the input
    if measData.shape[0] != 2:
        raise ValueError("Input must have shape (2, n)")

    # Perform calculation
    phaseNoise = np.array(
        [measData[0], 
        2 * np.power(10, measData[1] / 10)], 
        dtype=np.float64
    ) 
    return phaseNoise


def calc_freqNoise(
        measData: Union[NDArray[np.float64], Sequence[Sequence[float]]]
    ) -> NDArray[np.float64]:
    """
    Calculate frequency noise from L(f) measurement data.

    Parameters:
        measData (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - The first row (measData[0]): Fourier frequency in Hz.
            - The second row (measData[1]): L(f) in dBc/Hz.

    Returns:
        NDArray[np.float64]: A 2D NumPy array with dtype float64 and shape (2, n), where:
            - Row 0: Fourier frequency in Hz.
            - Row 1: Frequency noise PSD in Hz^2/Hz.

    Raises:
        ValueError: If the input data does not have shape (2, n).
    """
    # Convert input to a NumPy array with dtype float64
    if not (isinstance(measData, np.ndarray) and measData.dtype == np.float64):
        measData = np.asarray(measData, dtype=np.float64)

    # Validate the structure of the input
    if measData.shape[0] != 2:
        raise ValueError("Input must have shape (2, n)")

    # Perform calculation
    freqNoise = np.array(
        [measData[0], 
        2 * np.power(10, measData[1] / 10) * np.power(measData[0], 2)],
        dtype=np.float64
    )
    return freqNoise

def calc_timingJitter(
        measData: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
        fcarrier: Union[int, float, np.float64]
    ) -> NDArray[np.float64]:
    """
    Calculate the timing jitter from L(f) measurement data.

    Parameters:
        measData (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - The first row (measData[0]): Fourier frequency in Hz.
            - The second row (measData[1]): L(f) in dBc/Hz.
        fcarrier (Union[int, float, np.float64]): The carrier frequency in Hz. Must be a positive number.

    Returns:
        NDArray[np.float64]: A 2D NumPy array with dtype float64 and shape (2, n), where:
            - Row 0: Fourier frequency in Hz.
            - Row 1: Timing jitter PSD in s^2/Hz.

    Raises:
        ValueError: If fcarrier is not a positive number or if the input data does not have shape (2, n).
    """
    # Validate fcarrier
    if fcarrier <= 0:
        raise ValueError("fcarrier must be a positive number")

    # Convert input to a NumPy array with dtype float64
    if not (isinstance(measData, np.ndarray) and measData.dtype == np.float64):
        measData = np.asarray(measData, dtype=np.float64)

    # Validate the structure of the input
    if measData.shape[0] != 2:
        raise ValueError("Input must have shape (2, n)")

    # Perform calculation
    SCALE_FACTOR = 1 / (2 * np.pi**2 * fcarrier**2)
    timingJitter = np.array(
        [measData[0], 
        SCALE_FACTOR * np.power(10, measData[1] / 10)],
        dtype=np.float64
    )
    return timingJitter

def calc_fractFreqNoise(
        measData: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
        fcarrier: Union[int, float, np.float64]
    ) -> NDArray[np.float64]:
    """
    Calculate the fractional frequency noise from L(f) measurement data.

    Parameters:
        measData (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - The first row (measData[0]): Fourier frequency in Hz.
            - The second row (measData[1]): L(f) in dBc/Hz.
        fcarrier (Union[int, float, np.float64]): The carrier frequency in Hz. Must be a positive number.

    Returns:
        NDArray[np.float64]: A 2D NumPy array with dtype float64 and shape (2, n), where:
            - Row 0: Fourier frequency in Hz.
            - Row 1: Fractional frequency noise PSD in 1/Hz.

    Raises:
        ValueError: If fcarrier is not a positive number or if the input data does not have shape (2, n).
    """
    # Validate fcarrier
    if fcarrier <= 0:
        raise ValueError("fcarrier must be a positive number")

    # Convert input to a NumPy array with dtype float64
    if not (isinstance(measData, np.ndarray) and measData.dtype == np.float64):
        measData = np.asarray(measData, dtype=np.float64)

    # Validate the structure of the input
    if measData.shape[0] != 2:
        raise ValueError("Input must have shape (2, n)")

    # Perform calculation
    SCALE_FACTOR = 2 / fcarrier**2
    fractFreqNoise = np.vstack([
        measData[0],
        SCALE_FACTOR * np.power(10, measData[1] / 10) * np.power(measData[0], 2)],
        dtype=np.float64
    )
    return fractFreqNoise


def calc_Linewidth(
        measData: Union[NDArray[np.float64], Sequence[Sequence[float]]]
    ) -> NDArray[np.float64]:
    """
    Estimate the linewidth from L(f) using the beta-separation line approach.

    Parameters:
        measData (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - The first row (measData[0]): Fourier frequency in Hz.
            - The second row (measData[1]): PSD data in dBc/Hz.

    Returns:
        NDArray[np.float64]: A 2D NumPy array with dtype float64 and shape (2, n), where:
            - Row 0: Fourier frequency in Hz.
            - Row 1: Accumulated linewidth.

    Raises:
        ValueError: If the input data does not have shape (2, n).
    """
    # Convert input to a NumPy array with dtype float64
    if not (isinstance(measData, np.ndarray) and measData.dtype == np.float64):
        measData = np.asarray(measData, dtype=np.float64)

    # Validate the structure of the input
    if measData.shape[0] != 2:
        raise ValueError("Input must have shape (2, n)")

    # Calculate frequency noise
    freqNoise = calc_freqNoise(measData)

    # Precompute the beta-separation line threshold
    beta_sep_line = 8 * np.log(2) / (np.pi ** 2)

    # Determine the integrand based on the beta-separation line
    toIntegrate = np.where(freqNoise[1] > beta_sep_line * measData[0], freqNoise[1], 0)

    # Perform integration; the arrays are flipped to integrate from high to low frequency, then flipped back to match the original order, sqrt applied to get linewidth
    integrated = np.sqrt(-np.flip(sp.integrate.cumulative_trapezoid(np.flip(toIntegrate), np.flip(measData[0]), initial=0)))

    # Calculate and return the accumulated linewidth
    SCALE_FACTOR = np.sqrt(8 * np.log(2))
    intLinewidth = np.vstack(
        [measData[0], 
         SCALE_FACTOR * integrated],
         dtype=np.float64
    )
    return intLinewidth

def scale_Lf_opt(
        measData: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
        wl: Union[int, float, np.float64], 
        fcarrier: Union[int, float, np.float64]
    ) -> NDArray[np.float64]:
    """
    Scale a measured L(f) input to the optical frequency.

    Parameters:
        measData (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - Row 0: Fourier frequency in Hz.
            - Row 1: PSD data in dBc/Hz.
        wl (Union[int, float, np.float64]): The optical wavelength in meters. Must be a positive number.
        fcarrier (Union[int, float, np.float64]): The carrier frequency of the L(f) data in Hz. Must be a positive number.

    Returns:
        NDArray[np.float64]: A 2D NumPy array with dtype float64 and shape (2, n), where:
            - Row 0: Fourier frequency in Hz.
            - Row 1: Scaled PSD data in dBc/Hz.

    Raises:
        ValueError: If wl or fcarrier are not a positive number or if the input data does not have shape (2, n).
    """
    # Validate wl
    if wl <= 0:
        raise ValueError("wl must be a positive number")

    # Validate fcarrier
    if fcarrier <= 0:
        raise ValueError("fcarrier must be a positive number")

    # Convert input to a NumPy array with dtype float64
    if not (isinstance(measData, np.ndarray) and measData.dtype == np.float64):
        measData = np.asarray(measData, dtype=np.float64)

    # Validate the structure of the input
    if measData.shape[0] != 2:
        raise ValueError("Input must have shape (2, n)")

    # Perform scaling
    SCALE_FACTOR = 20 * np.log10(sp.constants.c / wl / fcarrier)
    scaledMeasData = np.array(
        [measData[0], 
         measData[1] + SCALE_FACTOR],
        dtype=np.float64
    )
    return scaledMeasData

def normalize_Lf_frep(
        measData: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
        fcarrier: Union[int, float, np.float64], 
        frep: Union[int, float, np.float64]
    ) -> Tuple[NDArray[np.float64], int]:
    """
    Normalize a measured L(f) input to the fundamental repetition rate.

    Parameters:
        measData (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - Row 0: Fourier frequency in Hz.
            - Row 1: PSD data in dBc/Hz.
        fcarrier (Union[int, float, np.float64]): The carrier frequency of the L(f) data in Hz. Must be a positive number.
        frep (Union[int, float, np.float64]): The repetition rate of the laser in Hz. Must be a positive number.

    Returns:
        Tuple[NDArray[np.float64], int]: 
            - A 2D NumPy array with dtype float64 and shape (2, n), where:
                - Row 0: Fourier frequency in Hz.
                - Row 1: Normalized PSD data in dBc/Hz.
            - The harmonic number as an integer.

    Raises:
        ValueError: If fcarrier or frep are not a positive number or if the input data does not have shape (2, n).
    """
    # Validate fcarrier
    if fcarrier <= 0:
        raise ValueError("fcarrier must be a positive number")

    # Validate frep
    if frep <= 0:
        raise ValueError("frep must be a positive number")

    # Convert input to a NumPy array with dtype float64
    if not (isinstance(measData, np.ndarray) and measData.dtype == np.float64):
        measData = np.asarray(measData, dtype=np.float64)

    # Validate the structure of the input
    if measData.shape[0] != 2:
        raise ValueError("Input must have shape (2, n)")

    # Calculate the harmonic number
    harmonic = int(np.rint(fcarrier / frep))
    log_harmonic = 20 * np.log10(harmonic) # Scaling happens with the log of the harmonic number sqaured, therefore 20*log(harmonic) is used

    # Normalize the PSD data
    normMeasData = np.array(
        [measData[0], 
         measData[1] - log_harmonic],
        dtype=np.float64
    )
    return normMeasData, harmonic

def calc_rmsRin(
        measData: Union[NDArray[np.float64], Sequence[Sequence[float]]]
    ) -> NDArray[np.float64]:
    """
    Calculate the root mean square (RMS) relative intensity noise (RIN) from L(f) data.

    Parameters:
        measData (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - Row 0: Fourier frequency in Hz.
            - Row 1: PSD data in dBc/Hz.

    Returns:
        NDArray[np.float64]: A 2D NumPy array with dtype float64 and shape (2, n), where:
            - Row 0: Fourier frequency in Hz.
            - Row 1: Cumulative RMS RIN values as a percentage.

    Raises:
        ValueError: If the input data does not have shape (2, n).
    """
    # Convert input to a NumPy array with dtype float64
    if not (isinstance(measData, np.ndarray) and measData.dtype == np.float64):
        measData = np.asarray(measData, dtype=np.float64)

    # Validate the structure of the input
    if measData.shape[0] != 2:
        raise ValueError("Input must have shape (2, n)")

    # Linearize the PSD values
    linearized = np.power(10, measData[1] / 10)

    # Perform integration
    integrated = sp.integrate.cumulative_trapezoid(linearized[::-1], measData[0][::-1], initial=0)[::-1]

    # Calculate RMS RIN as percentage
    SCALE_FACTOR = np.sqrt(2) * 100
    rmsRin = np.vstack([
        measData[0],
        SCALE_FACTOR * np.sqrt(integrated)],
        dtype=np.float64
    )

    return rmsRin

def norm_spectrum(
        OSA_data: Union[NDArray[np.float64], Sequence[Sequence[float]]]
    ) -> NDArray[np.float64]:
    """
    Normalize the spectrum data.

    This function takes in a 2D array where the first row contains the wavelength data 
    and the second row contains the intensity data. It normalizes the intensity data 
    by dividing it by the maximum intensity value.

    Parameters:
        OSA_data (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - Row 0: Wavelength data.
            - Row 1: Intensity data.

    Returns:
        NDArray[np.float64]: A 2D NumPy array with dtype float64 and shape (2, n), where:
            - Row 0: Wavelength data.
            - Row 1: Normalized intensity data.

    Raises:
        ValueError: If the input data does not have shape (2, n).
    """
    # Convert input to a NumPy array with dtype float64
    if not (isinstance(OSA_data, np.ndarray) and OSA_data.dtype == np.float64):
        OSA_data = np.asarray(OSA_data, dtype=np.float64)

    # Validate the structure of the input
    if OSA_data.shape[0] != 2:
        raise ValueError("Input must have shape (2, n)")

    # Normalize the intensity data
    max_intensity = np.max(OSA_data[1])
    normalized_spectrum = np.array(
        [OSA_data[0], 
         OSA_data[1] / max_intensity], 
        dtype=np.float64
    )
    return normalized_spectrum

def calc_FWHM(
        OSA_data: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
        printFWHM: bool = True
    ) -> float:
    """
    Calculate the Full Width at Half Maximum (FWHM) of the measured spectrum.

    Parameters:
        OSA_data (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - Row 0: Wavelength data (in nm).
            - Row 1: Intensity data.
        printFWHM (bool, optional): If True, prints the FWHM value. Default is True.

    Returns:
        float: The FWHM of the measured spectrum in nm.

    Raises:
        ValueError: If the input data does not have shape (2, n).
    """
    # Convert input to a NumPy array with dtype float64
    if not (isinstance(OSA_data, np.ndarray) and OSA_data.dtype == np.float64):
        OSA_data = np.asarray(OSA_data, dtype=np.float64)

    # Validate the structure of the input
    if OSA_data.shape[0] != 2:
        raise ValueError("Input must have shape (2, n)")

    # Find indices where intensity is greater than or equal to half maximum
    half_max = 0.5 * np.max(OSA_data[1])
    indices = np.where(OSA_data[1] >= half_max)[0]

    # Compute FWHM
    FWHM = OSA_data[0, indices[-1]] - OSA_data[0, indices[0]]

    # Optionally print the FWHM
    if printFWHM:
        print(f'FWHM of the measured spectrum is: {FWHM:.2f} nm')

    return FWHM

def calc_CWL(
        OSA_data: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
        printCWL: bool = True
    ) -> float:
    """
    Calculate the center wavelength (CWL) of the measured spectrum.

    This function returns the wavelength corresponding to the maximum intensity value.

    Parameters:
        OSA_data (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - Row 0: Wavelength data (in nm).
            - Row 1: Intensity data.
        printCWL (bool, optional): If True, prints the CWL value. Default is True.

    Returns:
        float: The CWL of the measured spectrum in nm.

    Raises:
        ValueError: If the input data does not have shape (2, n).
    """
    # Convert input to a NumPy array with dtype float64
    if not (isinstance(OSA_data, np.ndarray) and OSA_data.dtype == np.float64):
        OSA_data = np.asarray(OSA_data, dtype=np.float64)

    # Validate the structure of the input
    if OSA_data.shape[0] != 2:
        raise ValueError("Input must have shape (2, n)")

    # Find the index of the maximum intensity
    max_index = np.argmax(OSA_data[1])

    # Get the center wavelength
    CWL = OSA_data[0, max_index]

    # Optionally print the CWL
    if printCWL:
        print(f'CWL of the measured spectrum is: {CWL:.2f} nm')

    return CWL

def correct_BB_meas(
        data_signal: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        data_bgd: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        voltage: float,
        resistance: float = 50
    ) -> NDArray[np.float64]:
    """
    Correct a baseband (BB) measurement by subtracting the detector background noise 
    and normalizing the result to the carrier power.

    Parameters:
        data_signal (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A 2D input array-like structure with shape (2, n).
            - Row 0: Fourier frequency in Hz.
            - Row 1: Signal power in dBm.
        data_bgd (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A 2D input array-like structure with shape (2, n).
            - Row 0: Fourier frequency in Hz.
            - Row 1: Background noise power in dBm.
        voltage (float): The voltage of the carrier signal in volts.
        resistance (float, optional): The resistance in ohms. Default is 50 ohms.

    Returns:
        NDArray[np.float64]: A 2D NumPy array with dtype float64 and shape (2, n), where:
            - Row 0: Fourier frequency in Hz.
            - Row 1: Corrected and normalized signal power in dBc.

    Raises:
        ValueError: If the shape of `data_signal` and `data_bgd` do not match or are not of shape (2, n).
    """
    # Convert inputs to NumPy arrays with dtype float64
    if not (isinstance(data_signal, np.ndarray) and data_signal.dtype == np.float64):
        data_signal = np.asarray(data_signal, dtype=np.float64)
    if not (isinstance(data_bgd, np.ndarray) and data_bgd.dtype == np.float64):
        data_bgd = np.asarray(data_bgd, dtype=np.float64)

    # Validate the structure of the inputs
    if data_signal.shape != data_bgd.shape:
        raise ValueError("Signal and background data must have the same shape.")

    if data_signal.shape[0] != 2:
        raise ValueError("Input data must have shape (2, n).")

    # Linearize signal and background data
    signal_lin = np.power(10, data_signal[1] / 10)
    bgd_lin = np.power(10, data_bgd[1] / 10)

    # Calculate carrier power in mW
    carrier_pwr = (voltage**2) / resistance * 1e3  # Convert to mW to match linearized data units

    # Correct and normalize data
    corrected_data_lin = signal_lin - bgd_lin
    normalized_data_lin = corrected_data_lin / carrier_pwr

    # Convert back to dBc
    normalized_data_dBc = 10 * np.log10(normalized_data_lin)

    # Construct the output array
    data_dBc = np.vstack([
        data_signal[0], 
        normalized_data_dBc
    ])

    return data_dBc