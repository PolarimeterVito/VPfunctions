from typing import List, Tuple, Union, Sequence
import numpy as np
from numpy.typing import NDArray

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
