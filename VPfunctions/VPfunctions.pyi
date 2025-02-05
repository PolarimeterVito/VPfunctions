from typing import List, Tuple, Union, Sequence
import numpy as np
from numpy.typing import NDArray
import pandas as pd

# File and Data Handling
def get_all_filenames(
    file_ext: str = ".csv", 
    root_dir: str = "."
) -> List[str]: ...

def read_RSRSWP_data(
    filename: str, 
    trace: int = 1, 
    sep: str = ","
) -> Tuple[str, float, float, NDArray[np.float64]]: ...

def read_RPFP_plot_data(
    filename: str, 
    trace: int
) -> NDArray[np.float64]: ...

def read_TLPM_data(
    filename: str
) -> Tuple[NDArray[np.int32], NDArray[np.datetime64], NDArray[np.float64], Tuple[NDArray[np.float64], ...]]: ...

def read_AQ6374_data(
    filename: str
) -> Tuple[NDArray[np.float64], float]: ...

def read_Redstone_data(
    filename: str
) -> NDArray[np.float64]: ...

def read_RSRSWP_RF_data(
    filename: str, 
    sep: str = ";"
) -> NDArray[np.float64]: ...

def read_RS_FSWP_noise_data(
    filename: str, 
    trace: int = 1, 
    sep: str = ","
) -> Tuple[str, float, float, NDArray[np.float64]]: ...

def read_RS_FSWP_RF_data(
    filename: str, 
    sep: str = ";"
) -> NDArray[np.float64]: ...

# Noise Calculations
def calc_phaseNoise(
    measData: Union[NDArray[np.float64], Sequence[Sequence[float]]]
) -> NDArray[np.float64]: ...

def calc_freqNoise(
    measData: Union[NDArray[np.float64], Sequence[Sequence[float]]]
) -> NDArray[np.float64]: ...

def calc_timingJitter(
    measData: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
    fcarrier: Union[int, float, np.float64]
) -> NDArray[np.float64]: ...

def calc_fractFreqNoise(
    measData: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
    fcarrier: Union[int, float, np.float64]
) -> NDArray[np.float64]: ...

def calc_Linewidth(
    measData: Union[NDArray[np.float64], Sequence[Sequence[float]]]
) -> NDArray[np.float64]: ...

def scale_Lf_opt(
    measData: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
    wl: Union[int, float, np.float64], 
    fcarrier: Union[int, float, np.float64]
) -> NDArray[np.float64]: ...

def normalize_Lf_frep(
    measData: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
    fcarrier: Union[int, float, np.float64], 
    frep: Union[int, float, np.float64]
) -> Tuple[NDArray[np.float64], int]: ...

def calc_rmsRin(
    measData: Union[NDArray[np.float64], Sequence[Sequence[float]]]
) -> NDArray[np.float64]: ...

def correct_BB_meas(
    data_signal: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
    data_bgd: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
    voltage: Union[int, float, np.float64], 
    resistance: Union[int, float, np.float64] = 50
) -> NDArray[np.float64]: ...

def norm_spectrum(
    OSA_data: Union[NDArray[np.float64], Sequence[Sequence[float]]]
) -> NDArray[np.float64]: ...

def calc_FWHM(
    OSA_data: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
    printFWHM: bool = True
) -> float: ...

def calc_CWL(
    OSA_data: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
    printCWL: bool = True
) -> float: ...

# Visualization
def plot_Lf(
    measData: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
    fileName: str
) -> None: ...

def plot_S_phi(
    measData: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
    fileName: str
) -> None: ...

def plot_S_nu(
    measData: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
    fileName: str
) -> None: ...

def plot_S_y(
    measData: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
    fcarrier: Union[int, float, np.float64], 
    fileName: str
) -> None: ...

def plot_S_x(
    measData: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
    fcarrier: Union[int, float, np.float64], 
    fileName: str
) -> None: ...

def plot_Linewidth(
    measData: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
    fileName: str
) -> None: ...

def plot_Lf_AM(
    measData: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
    fileName: str
) -> None: ...

def plot_integratedRmsRin(
    measData: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
    fileName: str
) -> None: ...

def plot_all_noise_files(
    all_fn: List[str], 
    frep: Union[int, float, np.float64], 
    wl: Union[int, float, np.float64], 
    normalize_to_frep: bool = True, 
    do_linewidth_estimation: bool = False
) -> None: ...

# Configuration
def set_plot_params(
    style: str = "presentation", 
    dpi: int = 600
) -> None: ...

# Dataframe Utilities
def AQ6374_to_df (
        file_names: List[str]
    ) -> pd.DataFrame: ...

def Redstone_to_df (
        file_names: List[str]
    ) -> pd.DataFrame: ...

def FSWP_PN_to_df (
        file_names: List[str]
    ) -> pd.DataFrame: ...

class MaterialClass:
    def __init__(
        self,
        wavelength: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        mode: str = 'provide Sellmeier',
        **kwargs
    ) -> None: ...

    def gvd_calculator(
        self,
        wavelength: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        coefficients: Union[NDArray[np.float64], Sequence[Sequence[float]]]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    @staticmethod
    def get_sellmeier_coefficients(
        wavelength: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        n_data: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        initial_guess: Union[NDArray[np.float64], Sequence[Sequence[float]], None] = None
    ) -> NDArray[np.float64]: ...

    @staticmethod
    def sellmeier(
        coefficients: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        wavelength: Union[NDArray[np.float64], Sequence[Sequence[float]]]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

class LaserClass:
    def __init__(
        self,
        wavelength: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
        center_wavelength: Union[int, float, np.float64],
        free_space_length: Union[int, float, np.float64],
        target_f_rep: Union[int, float, np.float64]
    ) -> None: ...

    def add_component(
        self,
        component: "ComponentClass"
    ) -> None: ...

    def material_gdd(
        self
    ) -> None: ...

    def ideal_grating_sep(
        self,
        grating_GDD: Union[NDArray[np.float64], Sequence[Sequence[float]]]
    ) -> None: ...

    def calculated_fiber_length(
        self,
        material: "MaterialClass"
    ) -> np.float64: ...

    def calculate_real_f_rep(
        self
    ) -> None: ...
    
    def calculate_free_space_length_without_components(
        self
    ) -> None: ...

    def calculate_laser(
        self,
        fiber_material: "MaterialClass",
        plotting: bool=True,
        provided_fiber_length: Union[None, int, float, np.float64]=None
    ) -> None: ...

    @staticmethod
    def gdd_grating(
        wavelength: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        grating_distance: Union[int, float, np.float64],
        grating_period: Union[int, float, np.float64]=1.0,
        alpha: Union[int, float, np.float64]=31.3
    ) -> NDArray[np.float64]: ...
    
class ComponentClass:
    def __init__(
        self,
        material: "MaterialClass",
        length: Union[int, float, np.float64],
        position: str,
        laser_instance: Union["LaserClass", None]=None
    ) -> None: ...

    def attach_laser(
        self,
        laser: "LaserClass"
    ) -> None: ...

    def gdd_calculator(
        self
    ) -> np.float64: ...