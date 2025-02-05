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

#### NALM_calc features

- **Wavelengths**: Calculations are possible for many wavelengths simultaneously
- **Class structure**: Material, component and laser environments are provided as classes making them easy to use while providing all necessary functionality
- **Material class**: Material properties (refractive index and GVD) can be provided in three different modes:
	- **Provide Sellmeier**: Coefficients of the three-part Sellmeier equation 
	- **Provide n**: Calculations via provided data for the refractive index, the GVD is calculated from it
	- **Provide n and GVD**: Both GVD and n are directly provided by the user
- **Component class**: With provided length, material, and position of the component in the laser the component is fully defined and automatically added to a provided laser instance
- **Laser definitions**: All properties of the laser are calculated from its attached components:
	- **Total material GDD**: The dispersion of all components added together
	- **Grating separation**: The grating separation needed to achieve 0 net GDD
	- **Fiber length**: Provided all other components including the gain fiber, the necessary fiber length can calculated to achieve the target repetition rate
	- **Repetition rate**: If an arbitrary fiber length is used, the resulting repetition rate can be calculated
	- **Net GDD**: Plotting is available to visualize the material GDD, grating GDD and net GDD


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
- `pandas`

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

Here’s a simple example of how to use define the materials using the MaterialClass using NALM_calc:

```python
import VPfunctions.NALM_calc as nalm

#Example working wavelength
lam = np.linspace(1.01, 1.08, 301) #wavelengths in micrometers

#Define example data of the refractive index
A=3.73895
B=0.0516156
C=-0.00115321
D=-0.00590524
def test_n(x):
    return np.sqrt(A + B/(x**2 - C) + D*x**2)
n_TGG_data = test_n(lam)

#Material definition via provide Sellmeier mode
ex1 = nalm.MaterialClass(wavelength=lam, mode='provide Sellmeier', coefficients=[0.6961663, 0.4079426, 0.8974794, 0.0684043**2, 0.116241**2, 9.896161**2])

#Material definition via provide n mode
ex2 = nalm.MaterialClass(wavelength=lam, mode='provide n', n=n_TGG_data)

#Material definition via provide n and GVD mode
ex3 = nalm.MaterialClass(wavelength=lam, mode='provide n and GVD', n=n_TGG_data, gvd=ex2.gvd)
```

Here’s a simple example of how to use to calculate a simple laser design of 20MHz repetition rate using NALM_calc:

```python
#Initiate the laser instance
Laser_20MHz = nalm.LaserClass(wavelength=lam, center_wavelength=1.03, free_space_length=250, target_f_rep=20)

# All components in the loop
comp_gainfiber = nalm.ComponentClass(material=ex1, length=500, position="loop", laser_instance=Laser_20MHz)
comp_WDM = nalm.ComponentClass(material=ex1, length=55, position="loop", laser_instance=Laser_20MHz)
comp_PBCC = nalm.ComponentClass(material=ex1, length=25, position="loop", laser_instance=Laser_20MHz)

# All components in the free space section
comp_FR = nalm.ComponentClass(material=ex2, length=20, position="free_space", laser_instance=Laser_20MHz)
comp_l4 = nalm.ComponentClass(material=ex1, length=1, position="free_space", laser_instance=Laser_20MHz)
comp_l2_1 = nalm.ComponentClass(material=ex1, length=1, position="free_space", laser_instance=Laser_20MHz)
comp_PBC = nalm.ComponentClass(material=ex1, length=10, position="free_space", laser_instance=Laser_20MHz)
comp_l2_2 = nalm.ComponentClass(material=ex1, length=1, position="free_space", laser_instance=Laser_20MHz)
comp_grating1 = nalm.ComponentClass(material=ex1, length=0.95, position="free_space", laser_instance=Laser_20MHz)
comp_grating2 = nalm.ComponentClass(material=ex1, length=0.95, position="free_space", laser_instance=Laser_20MHz)

#Calculate the fiber length needed and add a fiber with that length
Laser_20MHz.calculate_laser(fiber_material=ex1, plotting=False)
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

### Module Structure of NALM_calc

The library contains the following classes with the attributes, parameters and methods as following:

#### `MaterialClass(wavelength, mode, **kwargs)`

##### Attributes:
- `instances (list)`: Class variable storing all instances of MaterialClass.
- `B1, B2, B3, C1, C2, C3 (np.float64)`: Sellmeier coefficients
- `gvd (NDArray[np.float64])`: Group velocity dispersion in fs²/mm.
- `n (NDArray[np.float64])`: Refractive index.
##### Parameters
- `wavelength`: The wavelengths for which the properties are calculated
- `mode`: The mode of initialization. Options are: 'provide Sellmeier', 'provide n', 'provide n and GVD'
- `**kwargs`: Mode specific parameters required:
	-  `coefficients`: A sequence of six Sellmeier coefficients (B1, B2, B3, C1, C2, C3) when mode is 'provide Sellmeier'
	- `n`: Refractive index values when mode is 'provide n' or 'provide n and GVD'.
	- `gvd`: GVD values when mode is 'provide n and GVD'.
##### Methods
- `GVD_calculator(wavelength, coefficients)`: Computes group velocity dispersion (GVD) at a given wavelength using Sellmeier coefficients.
- `get_sellmeier_coefficients(wavelength, n, initial_guess)`: Fits the Sellmeier equation to refractive index data and returns optimized coefficients. Providing an initial guess for the coefficients is optional, default is `[1, 1, 1, 0.01, 0.1, 100]` for (B1, B2, B3, C1, C2, C3)
- `sellmeier(coefficients, wavelength)`: Computes refractive index and its derivatives using the Sellmeier equation.


#### `ComponentClass(material, length, position)`

##### Attributes:
- `material`: The material of the component, which defines its optical properties.
- `length`: The physical length of the component in millimeters.
- `position`: The position of the component within the laser system. Options are:
	- 'loop': Positioned within the fiber loop.
	- 'free_space': Positioned in free space.
- `gdd`: The group delay dispersion (GDD) of the component in fs².
- `laser: The laser instance this component is attached to, if any.
##### Parameters
- `material`: The material of the component, which defines its optical properties.
- `length`: The physical length of the component in millimeters.
- `position`: The position of the component within the laser system. Options are:
	- 'loop': Positioned within the fiber loop.
	- 'free_space': Positioned in free space.
- `laser_instance`: The laser instance this component is attached to, if any.
##### Methods
- `attach_laser(laser)`: Attaches the component to a laser instance, updating both the component and the laser.
- `gdd_calculator()`: Computes the Group Delay Dispersion (GDD) of the component based on its material and length.

#### LaserClass(lam, wavelength, free_space_length, target_f_rep)`

##### Attributes:
- `instances`: Class variable to store all instances of LaserClass.
- `components`: List to store components added to the laser system.
- `center_wavelength`: Center wavelength of the laser in micrometers.
- `free_space_length`: Length of the free space in the laser system in millimeters.
- `wavelength`: List of wavelengths.
- `total_material_gdd`: Total group delay dispersion (GDD) of the material in ps².
- `ideal_grating_separation`: Ideal grating separation.
- `target_f_rep`: Target repetition frequency in MHz.
- `index`: Index of the center wavelength in `wavelength` to the given `wavelength`.
- `real_f_rep`: Real repetition frequency in MHz.
- `free_space_minus_components`: Free space length minus the length of components in the laser system.
##### Parameters
- `lam`: List of wavelengths.
- `wavelength`: Wavelength of the laser in micrometers.
- `free_space_length`: Length of the free space in the laser system in millimeters.
- `target_f_rep`: Target repetition frequency in Hz.
##### Methods
- `add_component(component)`: Adds a component to the laser system. The component must have attributes: `gdd`, `position`, `length`, and `material`.
- `material_gdd()`: Calculates the total material group delay dispersion (GDD) and stores it in `total_material_gdd`.
- `ideal_grating_sep(grating_GDD)`: Computes the ideal grating separation based on a given grating GDD.
- `calculated_fiber_length(material)`: Determines the fiber length required for a given material to achieve the target repetition frequency.
- `calculate_real_f_rep()`: Computes the real repetition frequency based on the current system configuration.
- `calculate_free_space_length_without_components()`: Computes the free space length without the components and updates `free_space_minus_components`.
- `calculate_laser(fiber_material, plotting=True, provided_fiber_length=None)`: Runs a full calculation of the laser system, including GDD, grating separation, and fiber length. Executes all methods above in the right order. Optionally generates a plot of dispersion curves and lets you use a custom fiber length instead of the optimal one if `provided_fiber_length` is provided.
- `gdd_grating(wavelength, grating_distance, grating_period=1.0, alpha=31.3)`: Calculate the Group Delay Dispersion (GDD) for a grating.

## Contribution

Contributions are welcome! If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/yourusername/VPfunctions).
The NALM_calc module was developed by Yannick Hein. 

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.