import os
import re
import numpy as np
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