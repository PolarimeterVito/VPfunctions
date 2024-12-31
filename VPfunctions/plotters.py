import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from numpy.typing import NDArray
from typing import Union, Sequence, List

from .manipulators import (
    calc_phaseNoise, 
    calc_freqNoise, 
    calc_fractFreqNoise, 
    calc_timingJitter, 
    calc_Linewidth, 
    calc_rmsRin, 
    normalize_Lf_frep, 
    scale_Lf_opt
)

from .readers import read_RSRSWP_data

def plot_Lf(
        measData: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        fileName: str
    ) -> None:
    """
    Plot the L(f) data on a semilogarithmic plot and save the plot as a PNG file.

    Parameters:
        measData (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - Row 0: Fourier frequencies (Hz).
            - Row 1: Corresponding L(f) values (dBc/Hz).
        fileName (str): The base name of the file to save the plot as. The plot will be saved in a 
                        directory named 'plots' with the filename format '<fileName>_Lf.png'.

    Returns:
        None

    Raises:
        ValueError: If `measData` does not have shape (2, n).
        ValueError: If `fileName` is not a non-empty string.

    Notes:
        - Existing files with the same name will be overwritten.
    """
    # Validate fileName
    if not fileName.strip():
        raise ValueError(f"fileName must be a non-empty string. Provided: '{fileName}'")

    # Convert measData to a NumPy array with dtype float64
    if not (isinstance(measData, np.ndarray) and measData.dtype == np.float64):
        measData = np.asarray(measData, dtype=np.float64)

    # Validate the structure of measData
    if measData.shape[0] != 2:
        raise ValueError("measData must have shape (2, n).")

    # Create the plot
    try:
        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)  # type: ignore

        ax.semilogx(measData[0], measData[1])  # type: ignore
        ax.set_xlabel(r'Fourier Frequency (Hz)', fontsize=20)  # type: ignore
        ax.set_ylabel(r'L(f) (dBc/Hz)', fontsize=20)  # type: ignore

        # Set axis ticks, limits, and grid
        ax.set_xticks(np.power(10, np.arange(0, 8, 1)))  # type: ignore
        ax.set_yticks(np.arange(np.round(np.amin(measData[1]), -1), np.round(np.amax(measData[1]), -1) + 20, 20))  # type: ignore
        ax.set_xlim([measData[0, 0], measData[0, -1]])  # type: ignore
        ax.set_ylim([np.amin(measData[1]), np.amax(measData[1]) + 5])  # type: ignore
        ax.grid(which='major', color='gray', linestyle='-', linewidth=0.75)  # type: ignore
        ax.grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)  # type: ignore

        # Style the plot spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        fig.tight_layout()

        # Ensure output directory exists and save the plot
        os.makedirs('plots', exist_ok=True)
        save_name = os.path.join('plots', f"{fileName}_Lf.png")
        fig.savefig(save_name)  # type: ignore
        print(f'Successfully plotted {save_name}')
    finally:
        # Ensure the plot is always closed
        plt.close(fig) # type: ignore
    return

def plot_S_phi(
        measData: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        fileName: str
    ) -> None:
    """
    Plot the phase noise data on a semilogarithmic plot and save the plot as a PNG file.

    Parameters:
        measData (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - Row 0: Fourier frequencies (Hz).
            - Row 1: Corresponding L(f) values (dBc/Hz).
        fileName (str): The base name of the file to save the plot as. The plot will be saved in a 
                        directory named 'plots' with the filename format '<fileName>_S_phi.png'.

    Returns:
        None

    Raises:
        ValueError: If `measData` does not have shape (2, n).
        ValueError: If `fileName` is not a non-empty string.

    Notes:
        - Existing files with the same name will be overwritten.
    """
    # Validate fileName
    if not fileName.strip():
        raise ValueError(f"fileName must be a non-empty string. Provided: '{fileName}'")

    # Convert measData to a NumPy array with dtype float64
    if not (isinstance(measData, np.ndarray) and measData.dtype == np.float64):
        measData = np.asarray(measData, dtype=np.float64)

    # Validate the structure of measData
    if measData.shape[0] != 2:
        raise ValueError("measData must have shape (2, n).")

    # Calculate phase noise
    pn = calc_phaseNoise(measData)

    # Create the plot
    try:
        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)  # type: ignore

        ax.loglog(pn[0], pn[1])  # type: ignore
        ax.set_xlabel(r'Fourier Frequency (Hz)', fontsize=20)  # type: ignore
        ax.set_ylabel(r'$PN-PSD\,(rad^2/Hz)$', fontsize=20)  # type: ignore

        # Set axis ticks, limits, and grid
        ax.set_xticks(np.power(10, np.arange(0, 8, 1)))  # type: ignore
        ax.set_xlim([pn[0, 0], pn[0, -1]])  # type: ignore
        ax.set_ylim([np.amin(pn[1]) * 0.8, np.amax(pn[1]) * 1.5])  # type: ignore
        ax.grid(which='major', color='gray', linestyle='-', linewidth=0.75)  # type: ignore
        ax.grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)  # type: ignore

        # Style the plot spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        fig.tight_layout()

        # Ensure output directory exists and save the plot
        os.makedirs('plots', exist_ok=True)
        save_name = os.path.join('plots', f"{fileName}_S_phi.png")
        fig.savefig(save_name)  # type: ignore
        print(f"Successfully plotted {save_name}")
    finally:
        # Ensure the plot is always closed
        plt.close(fig)  # type: ignore
    return

def plot_S_nu(
        measData: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        fileName: str
    ) -> None:
    """
    Plot the frequency noise data on a semilogarithmic plot and save the plot as a PNG file.

    Parameters:
        measData (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - Row 0: Fourier frequencies (Hz).
            - Row 1: Corresponding L(f) values (dBc/Hz).
        fileName (str): The base name of the file to save the plot as. The plot will be saved in a 
                        directory named 'plots' with the filename format '<fileName>_S_nu.png'.

    Returns:
        None

    Raises:
        ValueError: If `measData` does not have shape (2, n).
        ValueError: If `fileName` is not a non-empty string.

    Notes:
        - Existing files with the same name will be overwritten.
    """
    # Validate fileName
    if not fileName.strip():
        raise ValueError(f"fileName must be a non-empty string. Provided: '{fileName}'")

    # Convert measData to a NumPy array with dtype float64
    if not (isinstance(measData, np.ndarray) and measData.dtype == np.float64):
        measData = np.asarray(measData, dtype=np.float64)

    # Validate the structure of measData
    if measData.shape[0] != 2:
        raise ValueError("measData must have shape (2, n).")

    # Calculate frequency noise
    fn = calc_freqNoise(measData)

    # Create the plot
    try:
        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)  # type: ignore

        ax.loglog(fn[0], fn[1])  # type: ignore
        ax.set_xlabel(r'Fourier Frequency (Hz)', fontsize=20)  # type: ignore
        ax.set_ylabel(r'$FN-PSD\,(Hz^2/Hz)$', fontsize=20)  # type: ignore

        # Set axis ticks, limits, and grid
        ax.set_xticks(np.power(10, np.arange(0, 8, 1)))  # type: ignore
        ax.set_xlim([fn[0, 0], fn[0, -1]])  # type: ignore
        ax.set_ylim([np.amin(fn[1]) * 0.8, np.amax(fn[1]) * 1.5])  # type: ignore
        ax.grid(which='major', color='gray', linestyle='-', linewidth=0.75)  # type: ignore
        ax.grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)  # type: ignore

        # Style the plot spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        fig.tight_layout()

        # Ensure output directory exists and save the plot
        os.makedirs('plots', exist_ok=True)
        save_name = os.path.join('plots', f"{fileName}_S_nu.png")
        fig.savefig(save_name)  # type: ignore
        print(f"Successfully plotted {save_name}")
    finally:
        # Ensure the plot is always closed
        plt.close(fig)  # type: ignore
    return

def plot_S_y(
        measData: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        fcarrier: Union[int, float, np.float64],
        fileName: str
    ) -> None:
    """
    Plot the fractional frequency noise data on a semilogarithmic plot and save the plot as a PNG file.

    Parameters:
        measData (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - Row 0: Fourier frequencies (Hz).
            - Row 1: Corresponding L(f) values (dBc/Hz).
        fcarrier (Union[int, float, np.float64]): The carrier frequency of the L(f) data in Hz. Must be a positive number.
        fileName (str): The base name of the file to save the plot as. The plot will be saved in a 
                        directory named 'plots' with the filename format '<fileName>_S_y.png'.

    Returns:
        None

    Raises:
        ValueError: If `measData` does not have shape (2, n).
        ValueError: If `fcarrier` is not a positive number.
        ValueError: If `fileName` is not a non-empty string.

    Notes:
        - Existing files with the same name will be overwritten.
    """
    # Validate fileName
    if not fileName.strip():
        raise ValueError(f"fileName must be a non-empty string. Provided: '{fileName}'")

    # Validate fcarrier
    if fcarrier <= 0:
        raise ValueError(f"fcarrier must be a positive number. Provided: {fcarrier}")

    # Convert measData to a NumPy array with dtype float64
    if not (isinstance(measData, np.ndarray) and measData.dtype == np.float64):
        measData = np.asarray(measData, dtype=np.float64)

    # Validate the structure of measData
    if measData.shape[0] != 2:
        raise ValueError("measData must have shape (2, n).")

    # Calculate fractional frequency noise
    ffn = calc_fractFreqNoise(measData, fcarrier)

    # Create the plot
    try:
        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)  # type: ignore

        ax.loglog(ffn[0], ffn[1])  # type: ignore
        ax.set_xlabel(r'Fourier Frequency (Hz)', fontsize=20)  # type: ignore
        ax.set_ylabel(r'FFN-PSD (1/Hz)', fontsize=20)  # type: ignore

        # Set axis ticks, limits, and grid
        ax.set_xticks(np.power(10, np.arange(0, 8, 1)))  # type: ignore
        ax.set_xlim([ffn[0, 0], ffn[0, -1]])  # type: ignore
        ax.set_ylim([np.amin(ffn[1]) * 0.8, np.amax(ffn[1]) * 1.5])  # type: ignore
        ax.grid(which='major', color='gray', linestyle='-', linewidth=0.75)  # type: ignore
        ax.grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)  # type: ignore

        # Style the plot spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        fig.tight_layout()

        # Ensure output directory exists and save the plot
        os.makedirs('plots', exist_ok=True)
        save_name = os.path.join('plots', f"{fileName}_S_y.png")
        fig.savefig(save_name)  # type: ignore
        print(f"Successfully plotted {save_name}")
    finally:
        # Ensure the plot is always closed
        plt.close(fig)  # type: ignore
    return

def plot_S_x(
        measData: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        fcarrier: Union[int, float, np.float64],
        fileName: str
    ) -> None:
    """
    Plot the timing jitter data on a semilogarithmic plot and save the plot as a PNG file.

    Parameters:
        measData (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - Row 0: Fourier frequencies (Hz).
            - Row 1: Corresponding L(f) values (dBc/Hz).
        fcarrier (Union[int, float, np.float64]): The carrier frequency of the L(f) data in Hz. Must be a positive number.
        fileName (str): The base name of the file to save the plot as. The plot will be saved in a 
                        directory named 'plots' with the filename format '<fileName>_S_x.png'.

    Returns:
        None

    Raises:
        ValueError: If `measData` does not have shape (2, n).
        ValueError: If `fcarrier` is not a positive number.
        ValueError: If `fileName` is not a non-empty string.

    Notes:
        - Existing files with the same name will be overwritten.
    """
    # Validate fileName
    if not fileName.strip():
        raise ValueError(f"fileName must be a non-empty string. Provided: '{fileName}'")

    # Validate fcarrier
    if fcarrier <= 0:
        raise ValueError(f"fcarrier must be a positive number. Provided: {fcarrier}")

    # Convert measData to a NumPy array with dtype float64
    if not (isinstance(measData, np.ndarray) and measData.dtype == np.float64):
        measData = np.asarray(measData, dtype=np.float64)

    # Validate the structure of measData
    if measData.shape[0] != 2:
        raise ValueError("measData must have shape (2, n).")

    # Calculate timing jitter
    tj = calc_timingJitter(measData, fcarrier)

    # Create the plot
    try:
        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)  # type: ignore

        ax.loglog(tj[0], tj[1])  # type: ignore
        ax.set_xlabel(r'Fourier Frequency (Hz)', fontsize=20)  # type: ignore
        ax.set_ylabel(r'$TJ-PSD\,(s^2/Hz)$', fontsize=20)  # type: ignore

        # Set axis ticks, limits, and grid
        ax.set_xticks(np.power(10, np.arange(0, 8, 1)))  # type: ignore
        ax.set_xlim([tj[0, 0], tj[0, -1]])  # type: ignore
        ax.set_ylim([np.amin(tj[1]) * 0.8, np.amax(tj[1]) * 1.5])  # type: ignore
        ax.grid(which='major', color='gray', linestyle='-', linewidth=0.75)  # type: ignore
        ax.grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)  # type: ignore

        # Style the plot spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        fig.tight_layout()

        # Ensure output directory exists and save the plot
        os.makedirs('plots', exist_ok=True)
        save_name = os.path.join('plots', f"{fileName}_S_x.png")
        fig.savefig(save_name)  # type: ignore
        print(f"Successfully plotted {save_name}")
    finally:
        # Ensure the plot is always closed
        plt.close(fig)  # type: ignore
    return

def plot_Linewidth(
        measData: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        fileName: str
    ) -> None:
    """
    Plot the frequency noise data and the corresponding linewidth estimated with the beta-separation line approach on a semilogarithmic plot, and save the plot as a PNG file.

    Parameters:
        measData (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - Row 0: Fourier frequencies (Hz).
            - Row 1: Corresponding L(f) values (dBc/Hz).
        fileName (str): The base name of the file to save the plot as. The plot will be saved in a 
                        directory named 'plots' with the filename format '<fileName>_LW.png'.

    Returns:
        None

    Raises:
        ValueError: If `measData` does not have shape (2, n).
        ValueError: If `fileName` is not a non-empty string.

    Notes:
        - Existing files with the same name will be overwritten.
    """
    # Validate fileName
    if not fileName.strip():
        raise ValueError(f"fileName must be a non-empty string. Provided: '{fileName}'")

    # Convert measData to a NumPy array with dtype float64
    if not (isinstance(measData, np.ndarray) and measData.dtype == np.float64):
        measData = np.asarray(measData, dtype=np.float64)

    # Validate the structure of measData
    if measData.shape[0] != 2:
        raise ValueError("measData must have shape (2, n).")

    # Calculate frequency noise, beta-separation line, and linewidth
    fn = calc_freqNoise(measData)
    beta_sep_line = 8 * np.log(2) / (np.pi**2) * fn[0]
    lw = calc_Linewidth(measData)

    # Create the plot
    try:
        fig, ax = plt.subplots(2, 1, sharex=True, squeeze=True)  # type: ignore

        # Top subplot: Frequency noise
        ax[0].loglog(fn[0], fn[1], label="Frequency Noise")  # type: ignore
        ax[0].loglog(fn[0], beta_sep_line, label="Beta Separation Line", linestyle='--')  # type: ignore
        ax[0].set_ylabel(r'$FN-PSD\,(Hz^2/Hz)$', fontsize=15)  # type: ignore
        ax[0].set_xlim([fn[0, 0], fn[0, -1]])  # type: ignore
        ax[0].set_ylim([np.amin(fn[1]) * 0.8, np.amax(fn[1]) * 1.5])  # type: ignore
        ax[0].grid(which='major', color='gray', linestyle='-', linewidth=0.75)  # type: ignore
        ax[0].grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)  # type: ignore
        for spine in ax[0].spines.values():
            spine.set_linewidth(1.5)
        ax[0].legend(fontsize=12)

        # Bottom subplot: Linewidth
        ax[1].semilogx(lw[0], lw[1], label="Linewidth")  # type: ignore
        ax[1].set_xlabel(r'Fourier Frequency (Hz)', fontsize=15)  # type: ignore
        ax[1].set_ylabel(r'Linewidth (Hz)', fontsize=15)  # type: ignore
        ax[1].set_xlim([lw[0, 0], lw[0, -1]])  # type: ignore
        ax[1].set_ylim([0, np.amax(lw[1]) * 1.2])  # type: ignore
        ax[1].grid(which='major', color='gray', linestyle='-', linewidth=0.75)  # type: ignore
        ax[1].grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)  # type: ignore
        for spine in ax[1].spines.values():
            spine.set_linewidth(1.5)
        ax[1].legend(fontsize=12)

        fig.tight_layout()

        # Ensure output directory exists and save the plot
        os.makedirs('plots', exist_ok=True)
        save_name = os.path.join('plots', f"{fileName}_LW.png")
        fig.savefig(save_name)  # type: ignore
        print(f"Successfully plotted {save_name}")
    finally:
        # Ensure the plot is always closed
        plt.close(fig)  # type: ignore
    return

def plot_Lf_AM(
        measData: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        fileName: str
    ) -> None:
    """
    Plot the L(f) of an AM noise measurement on a semilogarithmic plot and save the plot as a PNG file.

    Parameters:
        measData (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - Row 0: Fourier frequencies (Hz).
            - Row 1: Corresponding L(f) values (dBc/Hz).
        fileName (str): The base name of the file to save the plot as. The plot will be saved in a 
                        directory named 'plots' with the filename format '<fileName>_Lf_AM.png'.

    Returns:
        None

    Raises:
        ValueError: If `measData` does not have shape (2, n).
        ValueError: If `fileName` is not a non-empty string.

    Notes:
        - Existing files with the same name will be overwritten.
    """
    # Validate fileName
    if not fileName.strip():
        raise ValueError(f"fileName must be a non-empty string. Provided: '{fileName}'")

    # Convert measData to a NumPy array with dtype float64
    if not (isinstance(measData, np.ndarray) and measData.dtype == np.float64):
        measData = np.asarray(measData, dtype=np.float64)

    # Validate the structure of measData
    if measData.shape[0] != 2:
        raise ValueError("measData must have shape (2, n).")

    # Create the plot
    try:
        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)  # type: ignore

        ax.semilogx(measData[0], measData[1])  # type: ignore
        ax.set_xlabel(r'Fourier Frequency (Hz)', fontsize=20)  # type: ignore
        ax.set_ylabel(r'SSB AM-PSD (dBc/Hz)', fontsize=20)  # type: ignore

        # Set axis ticks, limits, and grid
        ax.set_xticks(np.power(10, np.arange(0, 8, 1)))  # type: ignore
        ax.set_xlim([measData[0, 0], measData[0, -1]])  # type: ignore
        ax.set_ylim([np.amin(measData[1]), np.amax(measData[1]) + 5])  # type: ignore
        ax.grid(which='major', color='gray', linestyle='-', linewidth=0.75)  # type: ignore
        ax.grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)  # type: ignore

        # Style the plot spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        fig.tight_layout()

        # Ensure output directory exists and save the plot
        os.makedirs('plots', exist_ok=True)
        save_name = os.path.join('plots', f"{fileName}_Lf_AM.png")
        fig.savefig(save_name)  # type: ignore
        print(f"Successfully plotted {save_name}")
    finally:
        # Ensure the plot is always closed
        plt.close(fig)  # type: ignore
    return

def plot_integratedRmsRin(
        measData: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        fileName: str
    ) -> None:
    """
    Plot the L(f) of an AM noise measurement and the corresponding cumulative RMS RIN on a semilogarithmic plot, and save the plot as a PNG file.

    Parameters:
        measData (Union[NDArray[np.float64], Sequence[Sequence[float]]]): A flexible 2D input array-like structure with shape (2, n).
            - Row 0: Fourier frequencies (Hz).
            - Row 1: Corresponding L(f) values (dBc/Hz).
        fileName (str): The base name of the file to save the plot as. The plot will be saved in a 
                        directory named 'plots' with the filename format '<fileName>_intRmsRin.png'.

    Returns:
        None

    Raises:
        ValueError: If `measData` does not have shape (2, n).
        ValueError: If `fileName` is not a non-empty string.

    Notes:
        - Existing files with the same name will be overwritten.
    """
    # Validate fileName
    if not fileName.strip():
        raise ValueError(f"fileName must be a non-empty string. Provided: '{fileName}'")

    # Convert measData to a NumPy array with dtype float64
    if not (isinstance(measData, np.ndarray) and measData.dtype == np.float64):
        measData = np.asarray(measData, dtype=np.float64)

    # Validate the structure of measData
    if measData.shape[0] != 2:
        raise ValueError("measData must have shape (2, n).")

    # Calculate integrated RMS RIN
    integrated_rms_rin = calc_rmsRin(measData)

    # Create the plot
    try:
        fig, ax = plt.subplots(2, 1, sharex=True, squeeze=True)  # type: ignore

        # Top subplot: SSB AM-PSD
        ax[0].semilogx(measData[0], measData[1])  # type: ignore
        ax[0].set_ylabel(r'SSB AM-PSD (dBc/Hz)', fontsize=15)  # type: ignore
        ax[0].set_xlim([measData[0, 0], measData[0, -1]])  # type: ignore
        ax[0].set_ylim([np.amin(measData[1]), np.amax(measData[1]) + 5])  # type: ignore
        ax[0].grid(which='major', color='gray', linestyle='-', linewidth=0.75)  # type: ignore
        ax[0].grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)  # type: ignore
        for spine in ax[0].spines.values():
            spine.set_linewidth(1.5)

        # Bottom subplot: Integrated RMS RIN
        ax[1].semilogx(integrated_rms_rin[0], integrated_rms_rin[1])  # type: ignore
        ax[1].set_xlabel(r'Fourier Frequency (Hz)', fontsize=15)  # type: ignore
        ax[1].set_ylabel(r'Integrated RMS RIN (%)', fontsize=15)  # type: ignore
        ax[1].set_xlim([integrated_rms_rin[0, 0], integrated_rms_rin[0, -1]])  # type: ignore
        ax[1].set_ylim([0, np.amax(integrated_rms_rin[1]) * 1.2])  # type: ignore
        ax[1].grid(which='major', color='gray', linestyle='-', linewidth=0.75)  # type: ignore
        ax[1].grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)  # type: ignore
        for spine in ax[1].spines.values():
            spine.set_linewidth(1.5)

        fig.tight_layout()

        # Ensure output directory exists and save the plot
        os.makedirs('plots', exist_ok=True)
        save_name = os.path.join('plots', f"{fileName}_intRmsRin.png")
        fig.savefig(save_name)  # type: ignore
        print(f"Successfully plotted {save_name}")
    finally:
        # Ensure the plot is always closed
        plt.close(fig)  # type: ignore
    return

def plot_all_noise_files(
        all_fn: List[str],
        frep: Union[int, float, np.float64],
        wl: Union[int, float, np.float64],
        normalize_to_frep: bool = True,
        do_linewidth_estimation: bool = False
    ) -> None:
    """
    Plot various noise characteristics from a list of measurement files.

    This function reads measurement data from each file in `all_fn`, determines the type of measurement (PN or AM),
    and plots various noise characteristics. If the measurement type is 'PN' and `normalize_to_frep` is True, the data
    is normalized to the repetition rate of the laser. If `do_linewidth_estimation` is True, the linewidth is estimated
    using the beta-separation-line approach. The function generates and saves plots for phase noise, frequency noise,
    and other related characteristics. If the measurement type is 'AM', it plots amplitude noise characteristics.

    Parameters:
        all_fn (List[str]): List of file paths to the measurement data files.
        frep (Union[int, float, np.float64]): Repetition rate of the laser in Hz.
        wl (Union[int, float, np.float64]): Wavelength of the laser in meters.
        normalize_to_frep (bool, optional): If True, normalize the data to the repetition rate of the laser. Default is True.
        do_linewidth_estimation (bool, optional): If True, perform linewidth estimation using the beta-separation-line approach. Default is False.

    Returns:
        None

    Raises:
        ValueError: If `all_fn` is empty.
        ValueError: If `frep` or `wl` is not positive.
    """
    # Validate inputs
    if not all_fn:
        raise ValueError("The list of files `all_fn` cannot be empty.")
    if frep <= 0:
        raise ValueError("`frep` must be a positive number.")
    if wl <= 0:
        raise ValueError("`wl` must be a positive number.")

    for file_path in all_fn:
        try:
            # Read measurement data
            meas_type, fcarrier, pcarrier, meas_data = read_RSRSWP_data(file_path)
            curr_file_name = os.path.splitext(os.path.basename(file_path))[0]

            print(f"Processing file: {file_path}")
            print(f"Measurement type: {meas_type}")
            print(f"Carrier power: {pcarrier:.1f} dBm")

            if meas_type == 'PN':
                if normalize_to_frep:
                    plot_data, harmonic = normalize_Lf_frep(meas_data, fcarrier, frep)
                    rf_freq = frep
                    print(f"Normalized to laser repetition rate ({frep * 1e-6:.2f} MHz).")
                    print(f"Carrier frequency: {fcarrier * 1e-6:.2f} MHz (Harmonic: {harmonic})")
                else:
                    plot_data = meas_data
                    rf_freq = fcarrier
                    print(f"Not normalized to laser repetition rate.")
                    print(f"Carrier frequency: {fcarrier * 1e-6:.2f} MHz")

                # Plot phase noise characteristics
                plot_Lf(plot_data, curr_file_name)
                plot_S_phi(plot_data, curr_file_name)
                plot_S_nu(plot_data, curr_file_name)
                plot_S_x(plot_data, rf_freq, curr_file_name)
                plot_S_y(plot_data, rf_freq, curr_file_name)

                # Optional linewidth estimation
                if do_linewidth_estimation:
                    lw_data = scale_Lf_opt(meas_data, wl, fcarrier)
                    linewidth = calc_Linewidth(lw_data)[1, 0]
                    print(f"Estimated linewidth: {linewidth:.2e} Hz")
                    print(f"Optical frequency: {sp.constants.c / wl * 1e-12:.2f} THz")
                    plot_Linewidth(lw_data, curr_file_name)

            elif meas_type == 'AM':
                # Plot amplitude noise characteristics
                plot_Lf_AM(meas_data, curr_file_name)
                plot_integratedRmsRin(meas_data, curr_file_name)

            else:
                print(f"Unknown measurement type: {meas_type}. No plots generated.")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

        finally:
            plt.close('all')  # Ensure all plots are closed after processing each file

    print("Plotting finished!")


def set_plot_params(
        style: str = 'presentation', 
        dpi: int = 600
    ) -> None:
    """
    Set the parameters for plotting with matplotlib based on the specified style.

    Parameters:
        style (str): The style of the plot. Options are 'optica_two_column', 'optica_one_column', 
                     'presentation'. Defaults to 'presentation'.
        dpi (int): The resolution of the plot in dots per inch. Defaults to 600.

    Returns:
        None

    Raises:
        ValueError: If `dpi` is not a positive integer.
    """
    # Validate dpi
    if dpi <= 0:
        raise ValueError("dpi must be a positive integer.")

    # Define figure width and height based on the style
    if style == 'optica_two_column':
        figW, figH = 5.75, 5.75 / 2
    elif style == 'optica_one_column':
        figW, figH = 3.25, 3.25 / 2
    elif style == 'presentation':
        figW, figH = 40 / 2.54, 20 / 2.54  # Convert cm to inches
    else:
        print(f"Unknown style '{style}', using default 'presentation' style.")
        figW, figH = 40 / 2.54, 20 / 2.54  # Default to presentation style

    # Update matplotlib parameters
    plt.rcParams.update({
        'axes.grid': True,
        'axes.grid.axis': 'both',
        'axes.grid.which': 'both',
        'axes.linewidth': 0.75,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'lines.markeredgewidth': 0.75,
        'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round',
        'lines.dash_capstyle': 'round',
        'lines.dash_joinstyle': 'round',
        'axes.labelpad': 1,
        'font.size': 18,
        'axes.labelsize': 15,
        'font.sans-serif': 'Arial',
        'axes.unicode_minus': False,
        'mathtext.default': 'regular',
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Bitstream Vera Sans',
        'mathtext.it': 'Bitstream Vera Sans:italic',
        'mathtext.bf': 'Bitstream Vera Sans:bold',
        'legend.fontsize': 15,
        'legend.numpoints': 1,
        'legend.columnspacing': 0.3,
        'legend.fancybox': True,
        'legend.labelspacing': 0.2,
        'legend.handlelength': 1,
        'legend.handletextpad': 0.5,
        'legend.markerscale': 0.9,
        'legend.frameon': True,
        'legend.labelspacing': 0.02,
        'savefig.dpi': dpi,
        'text.usetex': False,
        'xtick.labelsize': 15,
        'xtick.major.size': 5,
        'xtick.minor.size': 2.5,
        'xtick.major.width': 1.25,
        'xtick.minor.width': 1.25,
        'ytick.labelsize': 15,
        'ytick.major.size': 5,
        'ytick.minor.size': 2.5,
        'ytick.major.width': 1.25,
        'ytick.minor.width': 1.25,
        'figure.figsize': (figW, figH),
    })
    return