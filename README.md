# Readme

**VPfunctions** is a Python library designed for extracting, processing, and visualizing data from measurement devices and software tools, which are commonly used in the Optical Metrology Research Group located at the Faculty of Physics, University of Vienna, Austria.

## Features

- **File Handling**: Extract filenames with specific extensions from directories, now supporting a variable root directory.
- **Data Parsing**: Parse and analyze measurement data from custom CSV formats, including files from specific devices and software tools:
  - **Rohde & Schwarz FSWP Phase Noise Analyzer**:
    - Noise files (AM Noise, Phase Noise, and Baseband Noise).
    - RF spectrum analyzer files.
  - **Yokogawa AQ6374 Optical Spectrum Analyzer** files.
  - **Thorlabs Redstone Optical Spectrum Analyzer** files.
  - **Thorlabs Powermeter Software** files.
  - **RP Fiber Power** simulation output files.
- **Noise Analysis**: Compute phase noise, frequency noise, timing jitter, fractional frequency noise, integrated RMS relative intensity noise (RIN), and more.
- **Optical Spectrum Analysis**: Compute the Full Width at Half Maximum (FWHM), the center wavelength (CWL) and normalize optical spectra.
- **Visualization**: Generate consistent and customizable plots for various noise characteristics and save them automatically.
- **Optimized Functions**: All functions now feature:
  - Enhanced speed using NumPy operations.
  - Support for flexible input types.
  - Comprehensive type annotations and error handling.
- **Specialized Tools**:
  - Scaling phase noise to optical frequencies.
  - Normalizing phase noise to laser repetition rates.
  - Estimating linewidth using the beta-separation line approach.
- **Integration of Pandas Dataframes**
  - A DataFrame containing all variations of phase noise can be easily generated from a list of input files.
  - A DataFrame containing the normalized PSD, as well as CWL and FWHM of an optical spectrum is returned from a list of input files.


## Installation

To use VPfunctions, first clone the repository and install the library using pip:

```bash
pip install VPfunctions
```

You can also directly install the library from GitHub:

```bash
pip install git+https://github.com/PolarimeterVito/VPfunctions.git
```

For updates, use:
```bash
pip install --upgrade git+https://github.com/PolarimeterVito/VPfunctions.git
```

## Dependencies

The library requires the following Python packages:

- `numpy`
- `matplotlib`
- `scipy`

These dependencies will be automatically installed if you follow the installation instructions.

## Usage

Here’s a simple example of how to use VPfunctions:

```python
import VPfunctions as vp

# Extract all filenames with the .csv extension
all_files = vp.get_all_filenames(".csv")
print(all_files)

# Read and process noise measurement data from a CSV file
measType, fcarrier, Pcarrier, measData = vp.read_RSRSWP_data("example.csv")
print("Measurement Type:", measType)
print("Carrier Frequency (Hz):", fcarrier)
print("Carrier Power (dBm):", Pcarrier)

# Plot L(f)
vp.plot_Lf(measData, "example_output")
```

## Library Structure

The library contains the following key functions:

### File and Data Handling

- `get_all_filenames(file_ext='.CSV', root_dir='.')`: Extract filenames with a specific extension from directories.
- `read_RS_FSWP_noise_data(filename, trace=1, sep=',')`: Parse CSV data for noise measurements from a Rohde&Schwarz FSWP file.
- `read_RS_FSWP_RF_data(filename, sep=';')`: Read RF data from from a Rohde&Schwarz FSWP file.
- `read_RPFP_plot_data(filename, trace)`: Read RP fiber power plot data for a specific trace.
- `read_TLPM_data(filename)`: Read data from Thorlabs power meter files with multiple power traces.
- `read_AQ6374_data(filename)`: Read measurement data from a Yokogawa AQ6374 and extract resolution info.
- `read_Redstone_data(filename)`: Read measurement data from Thorlabs Redstone files.


### Noise Calculations and Analysis

- `calc_phaseNoise(measData)`: Compute phase noise.
- `calc_freqNoise(measData)`: Compute frequency noise.
- `calc_timingJitter(measData, fcarrier)`: Compute timing jitter.
- `calc_fractFreqNoise(measData, fcarrier)`: Compute fractional frequency noise.
- `calc_Linewidth(measData)`: Calculate the optical linewidth using the beta-sep approach.
- `scale_Lf_opt(measData, wl, fcarrier)`: Scale phase noise to optical frequencies.
- `normalize_Lf_frep(measData, fcarrier, frep)`: Normalize phase noise to laser repetion rate.
- `calc_rmsRin(measData)`: Calculate integrated RMS RIN.
- `correct_BB_meas(data_signal, data_bgd, voltage, resistance=50)`: Correct baseband measurements for the detector background and normalize to the signal power.

### Optical Spectra Handling and Analysis

- `norm_spectrum(OSA_data)`: Normalize an optical spectrum to its maximum value.
- `calc_FWHM(OSA_data, printFWHM = True)`: Calculate the full width at half maximum (FWHM) of a spectrum. Printing the result my be switched off.
- `calc_CWL(OSA_data, printCWL = True)`: Calculate the central wavelength (CWL) of a spectrum. Printing the result my be switched off.

### Pandas Dataframe Integration

 - `AQ6374_to_df(file_names)`: Processes spectral data files from a Yokogawa AQ6374 and returns a DataFrame containg the raw data, normalized data, resolution, CWL and FWHM for each file.
 - `Redstone_to_df(file_names)`: Processes spectral data files from a Thorlabs Redstone OSA and returns a DataFrame containg the raw data, normalized data, resolution, CWL and FWHM for each file.
 - `FSWP_PN_to_df(file_names)`: Processes Rohde&Schwarz FSWP phase noise files, returning a dataframe containing all 5 common variations of phase noise as well as carrier frequency and power for each file.

### Visualization

- `plot_Lf(measData, fileName)`: Plot phase noise.
- `plot_S_phi(measData, fileName)`: Plot phase noise power spectral density.
- `plot_S_nu(measData, fileName)`: Plot frequency noise power spectral density.
- `plot_S_y(measData, fcarrier, fileName)`: Plot fractional frequency noise.
- `plot_S_x(measData, fcarrier, fileName)`: Plot timing jitter.
- `plot_Linewidth(measData, fileName)`: Plot frequency noise and linewidth.
- `plot_Lf_AM(measData, fileName)`: Plot SSB AM noise measurements.
- `plot_integratedRmsRin(measData, fileName)`: Plot integrated RMS RIN.
- `plot_all_noise_files(all_fn, frep, wl, normalize_to_frep=True, do_linewidth_estimation=False)`: Plot all noise data in all variants.

### Configuration

- `set_plot_params(style='presentation', dpi=600)`: Set plot parameters for different styles (e.g., presentation or journal).

## Contribution

Contributions are welcome! If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/yourusername/VPfunctions).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.