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