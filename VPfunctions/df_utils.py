import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import List, Dict, Any

from .readers import (
    read_AQ6374_data,
    read_Redstone_data,
    read_RS_FSWP_noise_data,
)

from .manipulators import (
    calc_phaseNoise, 
    calc_freqNoise, 
    calc_fractFreqNoise, 
    calc_timingJitter,
    norm_spectrum,
    calc_FWHM,
    calc_CWL,
)

def AQ6374_to_df (
        file_names: List[str]
    ) -> pd.DataFrame:
    """
    Processes spectral data files from a Yokogawa AQ6374 optical spectrum analyzer and returns the results in a pandas DataFrame.

    Parameters
    ----------
    file_names : List[str]
        List of file paths to process.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing:
        - 'Filename': Names of the processed files.
        - 'Raw Data': Raw spectral data as numpy arrays (2, n).
        - 'Normalized Data': Normalized spectral data as numpy arrays (2, n).
        - 'Resolution (nm)': Spectral resolution for each file.
        - 'CWL (nm)': Calculated center wavelength (CWL) values.
        - 'FWHM (nm)': Calculated full-width at half-maximum (FWHM) values.

    Raises
    ------
    ValueError
        If `file_names` is empty or None.
    Exception
        If an error occurs while processing any file. The error details are printed, and default values (None or NaN) are assigned for that file.

    Notes
    -----
    - The raw data is expected to have a shape of (2, n) for each file, where `n` can vary between files.
    - Internally, the function uses the `VPfunctions` module to read, normalize, and calculate the CWL and FWHM values.
    - If a file fails to process, its associated values are set to None or NaN in the DataFrame, and an error message is printed.
    - The function is robust and continues processing other files even if some fail.
    """
    # Validate input
    if not file_names or len(file_names) == 0:
        raise ValueError("The file_names list is empty. Provide at least one valid file path.")
    
    # Initialize arrays
    num_files: int = len(file_names)
    raw_data_array: NDArray[np.object_] = np.empty(num_files, dtype=np.object_)
    norm_data_array: NDArray[np.object_] = np.empty(num_files, dtype=np.object_)
    res_array: NDArray[np.float64] = np.empty(num_files, dtype=np.float64)
    cwl_array: NDArray[np.float64] = np.empty(num_files, dtype=np.float64)
    fwhm_array: NDArray[np.float64] = np.empty(num_files, dtype=np.float64)

    # Load the data from the files, normalize, calculate the CWL and FWHM
    for i, file in enumerate(file_names):
        try:
            raw_data_array[i], res_array[i] = read_AQ6374_data(file)
            norm_data_array[i] = norm_spectrum(raw_data_array[i])
            cwl_array[i] = calc_CWL(raw_data_array[i], False)
            fwhm_array[i] = calc_FWHM(raw_data_array[i], False)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            raw_data_array[i] = None
            norm_data_array[i] = None
            res_array[i] = np.nan
            cwl_array[i] = np.nan
            fwhm_array[i] = np.nan
    
    # Create a pandas DataFrame
    data: Dict[str, Any] = {
        'Filename': file_names,                  # List[str]
        'Raw Data': raw_data_array,              # NDArray[np.object_]
        'Normalized Data': norm_data_array,      # NDArray[np.object_]
        'Resolution (nm)': res_array,            # NDArray[np.float64]
        'CWL (nm)': cwl_array,                   # NDArray[np.float64]
        'FWHM (nm)': fwhm_array                  # NDArray[np.float64]
    }
    df = pd.DataFrame(data)

    return df

def Redstone_to_df (
        file_names: List[str]
    ) -> pd.DataFrame:
    """
    Processes spectral data files from a Thorlabs Redstone optical spectrum analyzer and returns the results in a pandas DataFrame.

    Parameters
    ----------
    file_names : List[str]
        List of file paths to process.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing:
        - 'Filename': Names of the processed files.
        - 'Raw Data': Raw spectral data as numpy arrays (2, n).
        - 'Normalized Data': Normalized spectral data as numpy arrays (2, n).
        - 'CWL (nm)': Calculated center wavelength (CWL) values.
        - 'FWHM (nm)': Calculated full-width at half-maximum (FWHM) values.

    Raises
    ------
    ValueError
        If `file_names` is empty or None.
    Exception
        If an error occurs while processing any file. The error details are printed, and default values (None or NaN) are assigned for that file.

    Notes
    -----
    - The raw data is expected to have a shape of (2, n) for each file, where `n` can vary between files.
    - Internally, the function uses the `VPfunctions` module to read, normalize, and calculate the CWL and FWHM values.
    - If a file fails to process, its associated values are set to None or NaN in the DataFrame, and an error message is printed.
    - The function is robust and continues processing other files even if some fail.
    """
    # Validate input
    if not file_names or len(file_names) == 0:
        raise ValueError("The file_names list is empty. Provide at least one valid file path.")
    
    # Initialize arrays
    num_files: int = len(file_names)
    raw_data_array: NDArray[np.object_] = np.empty(num_files, dtype=np.object_)
    norm_data_array: NDArray[np.object_] = np.empty(num_files, dtype=np.object_)
    cwl_array: NDArray[np.float64] = np.empty(num_files, dtype=np.float64)
    fwhm_array: NDArray[np.float64] = np.empty(num_files, dtype=np.float64)

    # Load the data from the files, normalize, calculate the CWL and FWHM
    for i, file in enumerate(file_names):
        try:
            raw_data_array[i] = read_Redstone_data(file)
            norm_data_array[i] = norm_spectrum(raw_data_array[i])
            cwl_array[i] = calc_CWL(raw_data_array[i], False)
            fwhm_array[i] = calc_FWHM(raw_data_array[i], False)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            raw_data_array[i] = None
            norm_data_array[i] = None
            cwl_array[i] = np.nan
            fwhm_array[i] = np.nan
    
    # Create a pandas DataFrame
    data: Dict[str, Any] = {
        'Filename': file_names,                  # List[str]
        'Raw Data': raw_data_array,              # NDArray[np.object_]
        'Normalized Data': norm_data_array,      # NDArray[np.object_]
        'CWL (nm)': cwl_array,                   # NDArray[np.float64]
        'FWHM (nm)': fwhm_array                  # NDArray[np.float64]
    }
    df = pd.DataFrame(data)

    return df

def FSWP_PN_to_df (
        file_names: List[str]
    ) -> pd.DataFrame:
    """
    Processes phase noise measurement files from a Rohde & Schwarz FSWP and returns the results in a pandas DataFrame.

    Parameters
    ----------
    file_names : List[str]
        List of file paths to process.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing:
        - 'Filename': Names of the processed files.
        - 'Carrier Frequency (Hz)': Carrier frequency for each file.
        - 'Carrier Power (dBm)': Carrier power for each file.
        - 'L(f) (dBc/Hz)': Phase noise power spectral density L(f).
        - 'Phase Noise (rad^2/Hz)': Phase noise spectral density S_phy(f).
        - 'Frequency Noise (Hz^2/Hz)': Frequency noise spectral density S_nu(f).
        - 'Fractional Frequency Noise (1/Hz)': Fractional frequency noise spectral density S_y(f).
        - 'Timing Jitter (s^2/Hz)': Timing jitter spectral density S_x(f).

    Raises
    ------
    ValueError
        If `file_names` is empty or None.
    Exception
        If an error occurs while processing any file. The error details are printed, and default values (None or NaN) are assigned for that file.

    Notes
    -----
    - The function assumes that the measurement type is 'PN' (Phase Noise). Files with other measurement types (e.g., BB or AM) will have variations of phase noise set to None.
    - The raw data arrays are expected to have varying sizes depending on the file content.
    - If a file fails to process, its associated values are set to None or NaN in the DataFrame, and an error message is printed.
    - The function is robust and continues processing other files even if some fail.
    """
    # Validate input
    if not file_names or len(file_names) == 0:
        raise ValueError("The file_names list is empty. Provide at least one valid file path.")
    
    # Initialize arrays
    num_files: int = len(file_names)
    Lf_array: NDArray[np.object_] = np.empty(num_files, dtype=np.object_)           # L(f), as received directly from the FSWP
    S_phy_array: NDArray[np.object_] = np.empty(num_files, dtype=np.object_)        # S_phy(f), phase noise
    S_nu_array: NDArray[np.object_] = np.empty(num_files, dtype=np.object_)         # S_nu(f), frequency noise
    S_y_array: NDArray[np.object_] = np.empty(num_files, dtype=np.object_)          # S_y(f), fractional frequency noise
    S_x_array: NDArray[np.object_] = np.empty(num_files, dtype=np.object_)          # S_x(f), timing jitter
    f_carrier_array: NDArray[np.float64] = np.empty(num_files, dtype=np.float64)    # carrier frequency
    P_carrier_array: NDArray[np.float64] = np.empty(num_files, dtype=np.float64)    # carrier power
    meas_type: str = ''                                                             # measurement type, only for checking that no BB or AM measurements are included

    # Load the data from the files, normalize, calculate the CWL and FWHM
    for i, file in enumerate(file_names):
        try:
            meas_type, f_carrier_array[i], P_carrier_array[i], Lf_array[i] = read_RS_FSWP_noise_data(file)
            if meas_type == 'PN':
                S_phy_array[i] = calc_phaseNoise(Lf_array[i])
                S_nu_array[i] = calc_freqNoise(Lf_array[i])
                S_y_array[i] = calc_fractFreqNoise(Lf_array[i],f_carrier_array[i])
                S_x_array[i] = calc_timingJitter(Lf_array[i])
            else:
                print(f"For file {file} measurement type is not PN, but {meas_type}. Variations of phase noise are set to None.")
                S_phy_array[i] = None
                S_nu_array[i] = None
                S_y_array[i] = None
                S_x_array[i] = None

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            S_phy_array[i] = None
            S_nu_array[i] = None
            S_y_array[i] = None
            S_x_array[i] = None
            f_carrier_array[i] = np.nan
            P_carrier_array[i] = np.nan
    
    # Create a pandas DataFrame
    data: Dict[str, Any] = {
        'Filename': file_names,                     	# List[str]
        'Carrier Frequency (Hz)': f_carrier_array,      # NDArray[np.float64]
        'Carrier Power (dBm)': P_carrier_array,         # NDArray[np.float64]
        'L(f) (dBc/Hz)': Lf_array,                      # NDArray[np.object_]
        'Phase Noise (rad^2/Hz)': S_phy_array,          # NDArray[np.object_]
        'Frequency Noise (Hz^2/Hz)': S_nu_array,        # NDArray[np.object_]
        'Fractional Frequency Noise (1/Hz)': S_y_array, # NDArray[np.object_]
        'Timing Jitter (s^2/Hz)': S_x_array,            # NDArray[np.object_]
    }
    df = pd.DataFrame(data)

    return df