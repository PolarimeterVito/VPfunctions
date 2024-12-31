import os
import re
import numpy as np
from datetime import datetime as dt
from typing import List, Tuple
from numpy.typing import NDArray
import sys

def get_all_filenames(
        file_ext: str = '.csv', 
        root_dir: str = '.'
    ) -> List[str]:
    """
    Retrieve all filenames with a specific extension from a given directory and its subdirectories.

    Parameters:
        file_ext (str): The file extension to filter by. Default is '.csv'.
        root_dir (str): The root directory to start searching from. Default is the current directory.

    Returns:
        List[str]: A list of file paths that match the specified file extension.
    """
    # Validate inputs
    if not file_ext.startswith("."):
        raise ValueError("file_ext must be a non-empty string starting with '.'")

    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"The directory '{root_dir}' does not exist or is not a directory.")

    # Initialize list to store all matching file paths
    all_fn: List[str] = []

    # Find and collect matching file paths
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(file_ext):  # Case-sensitive matching
                all_fn.append(os.path.join(dirpath, filename))
    return all_fn

def read_RSRSWP_data(
        filename: str, 
        trace: int=1, 
        sep: str=','
    ) -> Tuple[str, float, float, NDArray[np.float64]]:
    """
    Reads RSRSWP data from a specified file and returns measurement type, 
    carrier frequency, carrier power, and measurement data.

    Parameters:
        filename (str): The path to the file containing the RSRSWP data.
        trace (int): The trace number to read data from. Must be between 1 and 8. Default is 1.
        sep (str): The delimiter used in the file. Default is ','.

    Returns:
        Tuple[str, float, float, NDArray[np.float64]]: A tuple containing:
            - measType (str): The type of measurement.
            - fcarrier (float): The carrier frequency in Hz.
            - Pcarrier (float): The carrier power in dBm.
            - measData (NDArray[np.float64]): A 2D numpy array of shape (2, n) 
              with Fourier frequency in Hz and PSD values in dBc/Hz (phase noise)
              or dBm/Hz (baseband noise).
    """

    # Ensure the file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    
    # Validate the 'trace' parameter
    if not (1 <= trace <= 8):
        raise ValueError("The 'trace' parameter must be an integer between 1 and 8.")
    
    # Validate the 'sep' parameter
    if len(sep) != 1:
        raise ValueError("The 'sep' parameter must be a single character.")
    
    # Initialize variables
    readlines: int = 0
    ind: int = 0
    start: bool = False
    skiplines: int = sys.maxsize # Initialize skiplines to an "infinite" value
    freq: List[float] = []
    psd: List[float] = []
    measType: str = ""
    fcarrier: float = 0.
    Pcarrier: float = 0.
    
    # Open the file
    with open(filename, 'r') as f:
        # Read each line in the file
        for ii, line in enumerate(f, start=1):
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
                psd.append(float(lineData[1]))
                ind += 1
        # Create the np.array with frequency and PSD values
        measData = np.array([freq, psd], dtype=np.float64)
    
    return measType, fcarrier, Pcarrier, measData

def read_RPFP_plot_data(
        filename: str, 
        trace: int = 1
    ) -> NDArray[np.float64]:
    """
    Reads data from an RP fiber power plot file and extracts the specified trace.

    Parameters:
        filename (str): The path to the file containing the plot data.
        trace (int): The trace number to extract data for. Must be a positive integer.

    Returns:
        np.ndarray: A 2D NumPy array of shape (2, n) where:
            - The first row contains the x data.
            - The second row contains the y data.
    """

    # Validate inputs
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    if trace < 1:
        raise ValueError("The 'trace' parameter must be a positive integer.")
    
    # Initialize variables
    extract_data: bool = False
    x_data: List[float] = []
    y_data: List[float] = []
    
    # Open the file
    with open(filename, 'r') as f:
        # Read each line in the file
        for line in f:
            line_data = re.split(r'[,\s]+', line)
            # Start data extraction when the correct trace is found
            if line_data[1] == 'plot' and line_data[2] == str(trace):
                extract_data = True
            # Stop data extraction when an empty line is found    
            elif line_data[0] == '':
                extract_data = False
            elif extract_data:
                x_data.append(float(line_data[0]))
                y_data.append(float(line_data[1]))
    return np.array([x_data, y_data], dtype=np.float64)

def read_TLPM_data(
        filename: str
    ) -> Tuple[NDArray[np.int32], NDArray[np.datetime64], NDArray[np.float64], *Tuple[NDArray[np.float64], ...]]:
    """
    Reads the data from a Thorlabs Power Meter data file with n saved power traces.

    Parameters:
        filename (str): The path to the CSV file containing the TLPM data.

    Returns:
        Tuple[NDArray[np.int32], NDArray[object], NDArray[np.float64], Tuple[NDArray[np.float64], ...]]:
            - A 1D NumPy array of sample numbers (int).
            - A 1D NumPy array of datetime objects (timestamps).
            - A 1D NumPy array of relative times in minutes (float).
            - n independent 1D NumPy arrays (float), each representing a power measurement trace in watts.


    """
    # Validate input
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")

    # Initialize variables
    data_start: bool = False
    smpl: List[int] = []
    timestamp: List[dt] = []
    pwr_entries: int = 0
    pwr_data: List[List[float]] = []

    with open(filename, 'r') as f:
        for line in f:
            lineData = re.split(r'\s*,\s*', line)
            if lineData[0] == 'Samples':
                data_start = True
                # Initialize power data arrays
                pwr_entries = len(lineData)-4
                pwr_data = [[] for _ in range(pwr_entries)]  # Initialize power data as a list of lists
            elif data_start:
                smpl.append(int(lineData[0])) 
                timestamp.append(dt.strptime(f"{lineData[1]} {lineData[2]}", r"%m/%d/%Y %H:%M:%S.%f"))
                for i in range(pwr_entries):
                    pwr_data[i].append(float(lineData[i+3]))

    # Convert lists to NumPy arrays
    smpl_array = np.array(smpl, dtype=np.int32)
    timestamp_array = np.array(timestamp, dtype=np.datetime64)
    rel_time_min = np.array([(ts - timestamp[0]).total_seconds() / 60 for ts in timestamp], dtype=np.float64)
    pwr_data_arrays = tuple(np.array(trace, dtype=np.float64) for trace in pwr_data)


    return (smpl_array, timestamp_array, rel_time_min, *pwr_data_arrays)

def read_AQ6374_data(
        filename: str
    ) -> Tuple[NDArray[np.float64], float]:
    """
    Reads data from a Yokogawa AQ6374 optical spectrum analyzer file.

    Parameters:
        filename (str): The path to the file containing the AQ6374 data.

    Returns:
        Tuple[NDArray[np.float64], float]: A tuple containing:
            - measData: A 2D NumPy array where the first row contains wavelength data in nm (float)
              and the second row contains power data in a.u. (float).
            - res: The resolution of the measurement in nm (float).
    """

    # Validate input
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    
    # Initialize variables
    smpl_size: int = 0
    data_start: bool = False
    wl: List[float] = []
    pwr_data: List[float] = []
    res: float = 0.0
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
    
    # Convert lists to NumPy arrays
    measData = np.array([wl,pwr_data], dtype=np.float64)

    return measData, res

def read_Redstone_data(
        filename: str
    ) -> NDArray[np.float64]:
    """
    Reads data from a Thorlabs Redstone optical spectrum analyzer file.

    Parameters:
        filename (str): The path to the file containing the Redstone data.

    Returns:
        NDArray[np.float64]: A 2D NumPy array of shape (2, n), where:
            - The first row contains wavelengths in nm (float).
            - The second row contains power data in a.u. (float).
    """

    # Validate input
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    
    # Initialize variables
    data_start: bool = False
    wl: List[float] = []
    pwr_data: List[float] = []

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

        measData = np.array([wl,pwr_data], dtype=np.float64)
    return measData

def read_RSRSWP_RF_data(
        filename: str, 
        sep: str = ';'
    )-> NDArray[np.float64]:
    """
    Reads Rohde&Schwarz RSWP RF data from a file and returns it as a NumPy array.

    The function reads a file containing frequency and PSD (Power Spectral Density) values separated by a specified delimiter.

    Parameters:
        filename (str): The path to the file containing the data.
        sep (str, optional): The delimiter used in the file to separate values. Default is ';'.

    Returns:
        NDArray[np.float64]: A 2D NumPy array of shape (2, n), where:
            - The first row contains frequencies in Hz (float).
            - The second row contains power spectral density data in dBm/Hz (float).
    """
    # Ensure the file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    
    # Validate the 'sep' parameter
    if len(sep) != 1:
        raise ValueError("The 'sep' parameter must be a single character.")

    readlines: int = 0
    ind: int = 0
    skiplines: int = sys.maxsize # Initialize skiplines to an "infinite" value
    freq: List[float] = []
    psd: List[float] = []
    with open(filename, 'r') as f:
        for ii, line in enumerate(f, start=1):
            lineData = re.split(';|\n', line)
            if lineData[0] == 'Values':
                readlines = int(lineData[1])
                skiplines = ii
            if ii > skiplines and ind < readlines:
                freq.append(float(lineData[0]))
                psd.append(float(lineData[1]))
                ind += 1
        
    measData = np.array([freq, psd], dtype=np.float64)
    return measData