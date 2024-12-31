# Changelog

## [0.3.2] - 2024-12-31
### Fixed
- Forgot a few functions in the `__init__.py` file. They were now added, all can be accessed correctly now.

## [0.3.1] - 2024-12-31
### Added
- Added `py.typed` to declare type hint support.
- Included a `.pyi` stub file for improved type checking and autocompletion.

## [0.3.0] - 2024-12-30
### Changed
- **Overhauled Functions**:
  - All calculation and plotting functions have been comprehensively refactored for improved performance and usability:
    - Optimized for speed by relying on NumPy functions wherever possible.
    - Enhanced flexibility with support for dynamic input types.
    - Added detailed type annotations for compatibility with strict type-checking environments.
    - Improved error handling with clear and informative messages.
  - Functions such as `calc_rmsRin`, `scale_Lf_opt`, and `normalize_Lf_frep` now offer consistent behavior and robust input validation.
  - Compatibility with version [0.2.0] is maintained.
  
- **Standardized Plotting Functions**:
  - All plotting functions now follow a uniform structure, ensuring better maintainability and readability.
  - Improved axis configurations, grid styling, and customizable parameters.
  - Added automatic memory management to close figures after saving.

- **Enhanced Utility Functionality**:
  - The `get_all_filenames` function now supports a variable root directory via the `root_dir` parameter. Backwards compatibility with version [0.2.0] is ensured as the default value is `"."`.

### Documentation
- Expanded and standardized docstrings for all functions:
  - Included detailed explanations of input types, return values, and raised exceptions.
  - Added a `Raises` section to clarify potential errors.

### Dependencies
- Updated `requirements.txt` to ensure compatibility with the latest versions of NumPy, SciPy, and Matplotlib.

## [0.2.0] - 2024-12-29
### Changed
- Refactored library into modular structure:
  - Split `VPfunctions.py` into `readers.py`, `manipulators.py`, and `plotters.py`.
- Updated `__init__.py` to allow direct access to all functions via `vp.FUNCTION`.
- Compatibility with version [0.1.1] is maintained.

## [0.1.1] - 2024-12-22
### Added
- Docstrings to all functions for improved clarity.
- Optimized minor code bits for better readability.
- Small changes for functions `calc_FWHM(OSA_data)` and `calc_CWL(OSA_data)`: 
The automated print command for the returned quantities can now be switched off, 
The second argument defaults to `True` and is optional, ensuring compatibility with version [0.1.0].
The second argument is optional and default is `True`, therefore the new functions are compatible with version [0.1.0].

### Fixed
- Resolved minor inconsistencies in function implementations.