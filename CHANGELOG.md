## [0.1.1] - 2024-12-22
### Added
- Docstrings to all functions for improved clarity.
- Optimized minor code bits for better readability.
- Small changes for functions `calc_FWHM(OSA_data)` and `calc_CWL(OSA_data)`: 
The automated print command for the returned quantities can now be switched of, by adding `False` as an argument, i.e., `calc_CWL(OSA_data, False)`.
The second argument is optional and default = `True`, therefore the new functions are compatible with version [0.1.0].

### Fixed
- Resolved minor inconsistencies in function implementations.