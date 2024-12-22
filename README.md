**VPfunctions** is a Python library designed for extracting, processing, and visualizing data from CSV files, specifically tailored for noise measurements and related analyses.

## Features

- **File Handling**: Extract filenames with specific extensions from directories.
- **Data Parsing**: Parse and analyze measurement data from custom CSV formats.
- **Noise Analysis**: Compute phase noise, frequency noise, timing jitter, fractional frequency noise, and more.
- **Visualization**: Generate plots for various noise characteristics and save them automatically.
- **Specialized Functions**:
  - Scaling phase noise to optical frequencies.
  - Normalizing phase noise to laser repetition rates.
  - Estimating linewidth using the beta-separation line approach.
  - Calculating integrated RMS relative intensity noise (RIN).
  - Extracting and normalizing optical spectra.

## Installation

To use VPfunctions, first clone the repository and install the library using pip:

```bash
pip install VPfunctions
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
from VPfunctions import VPfunctions

# Extract all filenames with the .csv extension
all_files = VPfunctions.get_all_filenames(".csv")
print(all_files)

# Read and process noise measurement data from a CSV file
measType, fcarrier, Pcarrier, measData = VPfunctions.read_RSRSWP_data("example.csv")
print("Measurement Type:", measType)
print("Carrier Frequency (Hz):", fcarrier)
print("Carrier Power (dBm):", Pcarrier)

# Plot the phase noise
VPfunctions.plot_Lf(measData, "example_output")
```

## Library Structure

The library contains the following key functions:

### File and Data Handling

- `get_all_filenames(file_ext='.CSV')`: Extract filenames with a specific extension from directories.
- `read_RSRSWP_data(filename, trace=1, sep=',')`: Parse CSV data for noise measurements from R&S FSWP.
- `read_RPFP_plot_data(filename, trace)`: Read RPFP plot data for a specific trace.
- `read_TLPM_data(filename)`: Read data from Thorlabs power meter files with multiple power traces.
- `read_AQ6374_data(filename)`: Read measurement data from AQ6374 and extract resolution info.
- `read_Redstone_data(filename)`: Read measurement data from Redstone files.
- `read_RSRSWP_RF_data(filename, sep=';')`: Read RF data from R&S FSWP files.

### Noise Calculations

- `calc_phaseNoise(measData)`: Compute phase noise.
- `calc_freqNoise(measData)`: Compute frequency noise.
- `calc_timingJitter(measData, fcarrier)`: Compute timing jitter.
- `calc_fractFreqNoise(measData, fcarrier)`: Compute fractional frequency noise.
- `calc_Linewidth(measData)`: Calculate the optical linewidth using the beta-sep approach.
- `scale_Lf_opt(measData, wl, fcarrier)`: Scale phase noise to optical frequencies.
- `normalize_Lf_frep(measData, fcarrier, frep)`: Normalize phase noise to laser repetion rate.
- `calc_rmsRin(measData)`: Calculate integrated RMS RIN.
- `correct_BB_meas(data_signal, data_bgd, voltage, resistance=50)`: Correct basebandmeasurements.
- `norm_spectrum(OSA_data)`: Normalize an optical spectrum to its maximum value.
- `calc_FWHM(OSA_data)`: Calculate the full width at half maximum (FWHM) of a spectrum.
- `calc_CWL(OSA_data)`: Calculate the central wavelength (CWL) of a spectrum.

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

## Acknowledgments

Thanks to all contributors and users for their feedback and support.


