import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from numpy import sqrt
from scipy.constants import pi, c
from numpy.typing import NDArray
from typing import Union, Sequence, Tuple

class MaterialClass:
    """
    Represents a material with its Sellmeier coefficients and calculated optical properties.

    Attributes:
        instances (list): Class variable storing all instances of MaterialClass.
        B1, B2, B3, C1, C2, C3 (np.float64): Sellmeier coefficients.
        gvd (NDArray[np.float64]): Group velocity dispersion in fs²/mm.
        n (NDArray[np.float64]): Refractive index.

    Parameters:
        wavelength (Union[NDArray[np.float64], Sequence[Sequence[np.float64]]): The wavelength in micrometers.
        mode (str): The mode of initialization. Options are:
            - 'provide Sellmeier': Requires six Sellmeier coefficients.
            - 'provide n': Requires an array of refractive index values.
            - 'provide n and GVD': Requires arrays of refractive index and group velocity dispersion (GVD) values.
        **kwargs: Additional keyword arguments for different modes:
            - coefficients (Union[NDArray[np.float64], Sequence[Sequence[float]]): A sequence of six Sellmeier coefficients (B1, B2, B3, C1, C2, C3) when mode is 'provide Sellmeier'.
            - n (Union[NDArray[np.float64], Sequence[Sequence[float]]): Refractive index values when mode is 'provide n' or 'provide n and GVD'.
            - gvd (Union[NDArray[np.float64], Sequence[Sequence[float]]): GVD values when mode is 'provide n and GVD'.

    Methods:
        GVD_calculator(wavelength, coefficients) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
            Computes group velocity dispersion (GVD) at a given wavelength using Sellmeier coefficients.

        get_sellmeier_coefficients(wavelength, n_data, initial_guess) -> NDArray[np.float64]:
            Fits the Sellmeier equation to refractive index data and returns optimized coefficients.

        sellmeier(coefficients, wavelength) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
            Computes refractive index and its derivatives using the Sellmeier equation.

    Raises:
        ValueError: If required parameters for a given mode are missing or invalid.
    """

    instances = []  # Class variable to store all instances

    def __init__(
            self,
            wavelength: Union[NDArray[np.float64], Sequence[Sequence[float]]],
            mode: str = 'provide Sellmeier',
            **kwargs
        ) -> None:

        self.mode = mode
        self.wavelength = np.asarray(wavelength, dtype=np.float64)

        if mode == 'provide Sellmeier':
            # Requires an array of Sellmeier coefficients
            coefficients = np.asarray(kwargs["coefficients"], dtype=np.float64) if "coefficients" in kwargs else None

            # Check that exactly 6 Sellmeier coefficients are provided
            if coefficients is None or len(coefficients) != 6:
                raise ValueError("Mode 'provide Sellmeier' requires an array of 6 Sellmeier coefficients.")
            
            # Check that all coefficients are numerical values
            if not all(isinstance(coef, (int, float, np.int64, np.float64)) for coef in coefficients):
                raise ValueError("Sellmeier coefficients must be numerical values.")

            # Assign Sellmeier coefficients
            self.B1, self.B2, self.B3, self.C1, self.C2, self.C3 = coefficients

            # Perform calculations
            self.gvd, self.n = self.gvd_calculator(self.wavelength, coefficients)

        elif mode == 'provide n':
            #Requires data for n, GVD is calculated from Sellmeiner fit to data
            self.n = np.asarray(kwargs["n"], dtype=np.float64) if "n" in kwargs else None

            # Check that n data is provided
            if len(self.n) == 0:
                raise ValueError(f"Mode '{mode}' requires `n` data array, but received {kwargs.keys()}.")
            
            # Check that the wavelength and refractive index arrays are of the same length
            if len(self.wavelength) != len(self.n):
                raise ValueError("The wavelength array and refractive index array must be of the same length.")

            #Perform Sellmeier fit to get coefficients and get GVD
            self.B1, self.B2, self.B3, self.C1, self.C2, self.C3 = self.get_sellmeier_coefficients(self.wavelength, self.n)
            coefficients = np.asarray([self.B1, self.B2, self.B3, self.C1, self.C2, self.C3], dtype=np.float64)
            #Calculate GVD from Sellmeier coefficients 
            self.gvd, _ = self.gvd_calculator(self.wavelength, coefficients)


        elif mode == 'provide n and GVD':
            #Requires data for n and GVD
            self.n = np.asarray(kwargs["n"], dtype=np.float64) if "n" in kwargs else None
            self.gvd = np.asarray(kwargs["gvd"], dtype=np.float64) if "gvd" in kwargs else None

            # Check that n and GVD data are provided
            if len(self.n) == 0 or len(self.gvd) == 0:
                raise ValueError(f"Mode '{mode}' requires both `n` and `GVD` arrays, but received {kwargs.keys()}.")
            
            # Check that the wavelength, refractive index and GVD arrays are of the same length
            if len(self.wavelength) != len(self.n) or len(self.wavelength) != len(self.gvd):
                raise ValueError("The wavelength array, refractive index array and GVD array must be of the same length.")

        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'provide Sellmeier', 'provide n', and 'provide n and GVD'.")

        # Add instance to the class variable
        MaterialClass.instances.append(self)

    def gvd_calculator(
            self,
            wavelength: Union[NDArray[np.float64], Sequence[Sequence[float]]],
            coefficients: Union[NDArray[np.float64], Sequence[Sequence[float]]]
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Computes the group velocity dispersion (GVD) and refractive index using the Sellmeier equation.

        Parameters:
            wavelength (Union[NDArray[np.float64], Sequence[Sequence[float]]): Wavelength values in micrometers.
            coefficients (Union[NDArray[np.float64], Sequence[Sequence[float]]): Sellmeier coefficients (B1, B2, B3, C1, C2, C3).

        Returns:
            tuple:
            - GVD (NDArray[np.float64]): Group velocity dispersion in fs²/mm.
            - n (NDArray[np.float64]): Refractive index values.
        """
        # Check that lam is an array-like object
        wavelength = np.asarray(wavelength, dtype=np.float64)

        # Check that exactly 6 Sellmeier coefficients are provided
        if coefficients is None or len(coefficients) != 6:
            raise ValueError("Mode 'provide Sellmeier' requires an array of 6 Sellmeier coefficients.")
        
        # Check that all coefficients are numerical values
        if not all(isinstance(coef, (np.int64, np.float64)) for coef in coefficients):
            raise ValueError("Sellmeier coefficients must be numerical values.")
    
        # Use list comprehension to calculate n, dn, d2n
        n, dn, d2n = self.sellmeier(coefficients, wavelength)
        
        # Calculate GVD in fs^2/mm
        gvd = (wavelength**3) / (2 * pi * c**2) * d2n * 1e21
        
        # Return as numpy arrays
        return np.asarray(gvd, dtype=np.float64), np.asarray(n, dtype=np.float64)
    
    @staticmethod
    def get_sellmeier_coefficients(
        wavelength: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        n_data: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        initial_guess: Union[NDArray[np.float64], Sequence[Sequence[float]], None] = None
    ) -> NDArray[np.float64]:
        """
        Fits the Sellmeier equation to refractive index data and returns optimized coefficients.

        Parameters:
            wavelength (Union[NDArray[np.float64], Sequence[Sequence[float]]): Wavelength values in micrometers.
            n_data (Union[NDArray[np.float64], Sequence[Sequence[float]]): Refractive index values at the given wavelengths.
            initial_guess (Union[Sequence[np.float64], None], optional): Initial guess for the Sellmeier coefficients.

        Returns:
            NDArray[np.float64]: Optimized Sellmeier coefficients (B1, B2, B3, C1, C2, C3).

        Raises:
            RuntimeError: If curve fitting fails due to invalid initial guesses.
        """
        # Defining the Sellmeier function for fitting the data
        def sellmeier_func(wavelength, B1, B2, B3, C1, C2, C3):
            n = sqrt(
                1
                + (B1 * wavelength**2 / (wavelength**2 - C1))
                + (B2 * wavelength**2 / (wavelength**2 - C2))
                + (B3 * wavelength**2 / (wavelength**2 - C3))
            )
            return n

        # Check if initial_guess is provided, if not use a general guess
        if initial_guess is None:
            initial_guess = [1, 1, 1, 0.01, 0.1, 100]  # Generalized guess
        try:
            # Perform curve fitting using the Sellmeier function and return the fitting parameters
            params, covariance = curve_fit(sellmeier_func, wavelength, n_data, p0=initial_guess)
            return np.asarray(params, dtype=np.float64)
        except RuntimeError:
            # Raise an error if curve fitting fails
            raise RuntimeError("Curve fitting failed. Please provide valid initial guesses for the Sellmeier coefficients.")
    
    @staticmethod
    def sellmeier(
        coefficients: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        wavelength: Union[NDArray[np.float64], Sequence[Sequence[float]]]
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute the refractive index and its derivatives using the Sellmeier equation.

        Parameters:
            coefficients (Union[NDArray[np.float64], Sequence[Sequence[float]]): Sellmeier coefficients (B1, B2, B3, C1, C2, C3).
            wavelength (Union[NDArray[np.float64], Sequence[Sequence[float]]): Wavelength values in micrometers.

        Returns:
            tuple:
            - n (NDArray[np.float64]): Refractive index values.
            - dn (NDArray[np.float64]): First derivative of the refractive index with respect to wavelength.
            - d2n (NDArray[np.float64]): Second derivative of the refractive index with respect to wavelength.
        """
        # Ensure wavelength is a numpy array for element-wise operations
        wavelength = np.asarray(wavelength, dtype=np.float64)

        # Unpack the Sellmeier coefficients
        B1, B2, B3, C1, C2, C3 = coefficients
        
        # Compute refractive index using the Sellmeier equation
        n = sqrt(
            1
            + (B1 * wavelength**2 / (wavelength**2 - C1))
            + (B2 * wavelength**2 / (wavelength**2 - C2))
            + (B3 * wavelength**2 / (wavelength**2 - C3))
        )

        #Compute the first derivative of the Sellmeier equation wrt the wavelength
        dn = (
            (
                (2 * B1 * wavelength) / (wavelength**2 - C1)
                - (2 * B1 * wavelength**3) / ((wavelength**2 - C1)**2)
                + (2 * B2 * wavelength) / (wavelength**2 - C2)
                - (2 * B2 * wavelength**3) / ((wavelength**2 - C2)**2)
                + (2 * B3 * wavelength) / (wavelength**2 - C3)
                - (2 * B3 * wavelength**3) / ((wavelength**2 - C3)**2)
            ) / (
                2 * sqrt(
                    (B1 * wavelength**2 / (wavelength**2 - C1))
                    + (B2 * wavelength**2 / (wavelength**2 - C2))
                    + (B3 * wavelength**2 / (wavelength**2 - C3))
                    + 1
                )
            )
        )

        #Compute the second derivative of the Sellmeier equation wrt the wavelength
        d2n = (
            (
                (-10 * B1 * wavelength**2) / ((wavelength**2 - C1)**2)
                + (2 * B1) / (wavelength**2 - C1)
                + (8 * B1 * wavelength**4) / ((wavelength**2 - C1)**3)
                - (10 * B2 * wavelength**2) / ((wavelength**2 - C2)**2)
                + (2 * B2) / (wavelength**2 - C2)
                + (8 * B2 * wavelength**4) / ((wavelength**2 - C2)**3)
                - (10 * B3 * wavelength**2) / ((wavelength**2 - C3)**2)
                + (2 * B3) / (wavelength**2 - C3)
                + (8 * B3 * wavelength**4) / ((wavelength**2 - C3)**3)
            ) / (
                2 * sqrt(
                    (B1 * wavelength**2 / (wavelength**2 - C1))
                    + (B2 * wavelength**2 / (wavelength**2 - C2))
                    + (B3 * wavelength**2 / (wavelength**2 - C3))
                    + 1
                )
            )
            - (
                (
                    (2 * B1 * wavelength) / (wavelength**2 - C1)
                    - (2 * B1 * wavelength**3) / ((wavelength**2 - C1)**2)
                    + (2 * B2 * wavelength) / (wavelength**2 - C2)
                    - (2 * B2 * wavelength**3) / ((wavelength**2 - C2)**2)
                    + (2 * B3 * wavelength) / (wavelength**2 - C3)
                    - (2 * B3 * wavelength**3) / ((wavelength**2 - C3)**2)
                ) ** 2
            ) / (
                4 * (
                    (B1 * wavelength**2 / (wavelength**2 - C1))
                    + (B2 * wavelength**2 / (wavelength**2 - C2))
                    + (B3 * wavelength**2 / (wavelength**2 - C3))
                    + 1
                ) ** (3/2)
            )
        )
        
        # Return n, dn, d2n
        return np.asarray(n, dtype=np.float64), np.asarray(dn, dtype=np.float64), np.asarray(d2n, dtype=np.float64)

class LaserClass:
    """
    Represents a laser system with various components and properties.

    Attributes:
        instances (list): Class variable to store all instances of LaserClass.
        components (list): List to store components added to the laser system.
        center_wavelength (Union[int, float, np.float64]): Center wavelength of the laser in micrometers.
        free_space_length (Union[int, float, np.float64]): Length of the free space in the laser system in millimeters.
        wavelength (Union[NDArray[np.float64], Sequence[Sequence[float]]): Wavelengths to be evaluated.
        total_material_gdd (float): Total group delay dispersion (GDD) of the material in ps².
        ideal_grating_separation (float): Ideal grating separation.
        target_f_rep (Union[int, float, np.float64]): Target repetition frequency in MHz.
        index (int): Index of the center wavelength in `wavelength` to the given `center_wavelength`.
        real_f_rep (float): Real repetition frequency in MHz.
        free_space_minus_components (float): Free space length minus the length of components in the laser system.

    Parameters:
        wavelength (Union[NDArray[np.float64], Sequence[Sequence[float]]): Wavelengths to be evaluated.
        center_wavelength (Union[int, float, np.float64]): Center wavelength of the laser in micrometers.
        free_space_length (Union[int, float, np.float64]): Length of the free space in the laser system in millimeters.
        target_f_rep (Union[int, float, np.float64]): Target repetition frequency in Hz.

    Methods:
        add_component(component):
            Adds a component to the laser system. The component must have attributes: `gdd`, `position`, `length`, and `material`.

        material_gdd():
            Calculates the total material group delay dispersion (GDD) and stores it in `total_material_gdd`.

        ideal_grating_sep(grating_GDD):
            Computes the ideal grating separation based on a given grating GDD.

        calculated_fiber_length(material) -> float:
            Determines the fiber length required for a given material to achieve the target repetition frequency.

        calculate_real_f_rep():
            Computes the real repetition frequency based on the current system configuration.

        calculate_free_space_length_without_components():
            Computes the free space length without the components and updates `free_space_minus_components`.

        calculate_laser(fiber_material, plotting=True, provided_fiber_length=None):
            Runs a full calculation of the laser system, including GDD, grating separation, 
            and fiber length. Optionally generates a plot of dispersion curves.

        gdd_grating(wavelength, grating_distance, grating_period=1.0, alpha=31.3) -> NDArray[np.float64]:
            Calculate the Group Delay Dispersion (GDD) for a grating.

    Raises:
        ValueError: If the provided `center_wavelength` is outside the range of `wavelength`.
        ValueError: If the provided `center_wavelength`, `free_space_length`, or `target_f_rep` is negative or zero.
        TypeError: If a component added to the system lacks required attributes.
    """
    instances = []  # Class variable to store all instances

    def __init__(
        self,
        wavelength: Union[NDArray[np.float64], Sequence[Sequence[float]]], 
        center_wavelength: Union[int, float, np.float64],
        free_space_length: Union[int, float, np.float64],
        target_f_rep: Union[int, float, np.float64]
    ) -> None:
        
        self.components = []
        LaserClass.instances.append(self)

        #Check that wavelength, free_space_length and target_f_rep and not negative and non-zero
        if center_wavelength <= 0 or free_space_length <= 0 or target_f_rep <= 0:
            raise ValueError("Center wavelength, free space length, and target repetition frequency must be positive and non-zero.")

        self.free_space_length = free_space_length # in mm
        self.wavelength = np.asarray(wavelength, dtype=np.float64)

        # Validate that the wavelength exists in the range
        if not (self.wavelength.min() <= center_wavelength <= self.wavelength.max()):
            raise ValueError(f"Center wavelength {center_wavelength} is not in range of wavelength array!")

        self.center_wavelength = center_wavelength # in micrometers
        self.total_material_gdd = 0.0
        self.ideal_grating_separation = 0.0
        self.real_f_rep = 0.0
        self.target_f_rep = target_f_rep*1e6 # in MHz

        # Find the index of the center wavelength in wavelength
        self.index = np.argmin(np.abs(self.wavelength - self.center_wavelength))
        # Calculate the free space length without components (just initialize at full length because no components are added yet)
        self.free_space_minus_components = free_space_length

    def add_component(
        self,
        component: "ComponentClass"
        ) -> None:
        """
        Adds a component to the laser system. The component must have attributes: `gdd`, `position`, `length`, and `material`.

        Parameters:
            component (ComponentClass): The component to be added to the laser system.

        Raises:
            TypeError: If the component lacks required attributes.
        """
        # Check if the component has the required attributes
        required_attrs = ["gdd", "position", "length", "material"]
        if not all(hasattr(component, attr) for attr in required_attrs):
            raise TypeError(f"Component is missing required attributes: {required_attrs}")
        self.components.append(component)

    def material_gdd(
        self
        ) -> None:
        """
        Calculates the total material group delay dispersion (GDD) for all components in the laser system.

        This method iterates through all components in the laser system, sums their GDD values, and stores the result
        in the `total_material_gdd` attribute. The GDD values are converted from fs² to ps².

        Raises:
            ValueError: If any component lacks the required GDD attribute.
        """
        total_gdd = np.zeros_like(self.wavelength, dtype=np.float64)
        for component in self.components:
            total_gdd += component.gdd
        self.total_material_gdd = total_gdd/1e6 # conversion from fs^2 to ps^2
    
    def ideal_grating_sep(
        self,
        grating_GDD: Union[NDArray[np.float64], Sequence[Sequence[float]]]
        ) -> None:
        """
        Computes the ideal grating separation based on the total material GDD and a given grating GDD.

        Parameters:
            grating_GDD (Union[NDArray[np.float64], list[np.float64]]): Grating GDD values.

        Raises:
            ValueError: If the lengths of `total_material_gdd` and `grating_GDD` do not match.
        """
        # Convert grating_GDD to a numpy array
        grating_GDD = np.asarray(grating_GDD, dtype=np.float64)

        # Check that total_material_gdd and grating_GDD have the same length
        if len(self.total_material_gdd) != len(grating_GDD):
            raise ValueError("total_material_gdd and grating_GDD must have the same length.")
        
        # Calculate the ideal grating separation
        self.ideal_grating_separation = -1 * self.total_material_gdd[self.index] / grating_GDD[self.index]

    def calculated_fiber_length(
        self,
        material: "MaterialClass"
        ) -> np.float64:
        """
        Determines the fiber length required for a given material to achieve the target repetition frequency.

        Parameters:
            material (MaterialClass): The material for which the fiber length is to be calculated.

        Returns:
            fiber legnth (np.float64): The calculated fiber length in millimeters.

        Raises:
            TypeError: If the material lacks the required attribute `n`.
            ValueError: If the components included are too long to reach the target repetition rate.
        """
        # Calculate the optical length of each component and add them all up
        product_n_L_components = sum(
            (2 if component.position == "free_space" else 1) * component.length * component.material.n[self.index]
            for component in self.components
        )
        # Calculate the left-over optical length
        left_over_length = (c/1e-3/self.target_f_rep)-product_n_L_components-2*self.free_space_minus_components

        # Check if the left-over optical length is negative
        if left_over_length < 0:
            raise ValueError("The components included are too long to reach the target repetition rate.")

        # Check if the component has the required attribute n
        required_attrs = ["n"]
        if not all(hasattr(material, attr) for attr in required_attrs):
            raise TypeError(f"Component is missing required attribute: {required_attrs}")

        # Calculate the physical length of the fiber from the optical length
        return np.float64(left_over_length/material.n[self.index])
    
    def calculate_real_f_rep(
        self
        ) -> None:
        """
        Computes the real repetition frequency based on the current system configuration.
        """
        # Calculate the optical length of each component and add them all up
        product_n_L_components = sum(
            (2 if component.position == "free_space" else 1) * component.length * component.material.n[self.index]
            for component in self.components
        )
        # Calculate the real repetition frequency from all the components
        self.real_f_rep = np.float64(c/((2*self.free_space_minus_components + product_n_L_components)/1e3)*1e-6)

    def calculate_free_space_length_without_components(
        self
        ) -> None:
        """
        Computes the free space length without the components and updates `free_space_minus_components`.

        Raises:
            ValueError: If the sum of the lengths of components in free space exceeds the total free space length.
        """
        # Calculate the sum of the lengths of components in free space in millimeters
        sum_components = sum(component.length for component in self.components if component.position == "free_space")
        # Calculate the free space length without components in millimeters
        self.free_space_minus_components = np.float64(self.free_space_length - sum_components)
        
        # Check if the free space length without components is negative
        if self.free_space_minus_components < 0:
            raise ValueError("Components in the free space section can't be longer than the total length of the free space section.")



    def calculate_laser(
        self,
        fiber_material: "MaterialClass",
        plotting: bool=True,
        provided_fiber_length: Union[None, int, float, np.float64]=None
        ) -> None:
        """
        Runs a full calculation of the laser system, including GDD, grating separation, and fiber length.

        Parameters:
            fiber_material (MaterialClass): The material of the fiber to be used in the laser system.
            plotting (bool, optional): Whether to generate a plot of dispersion curves. Default is True.
            provided_fiber_length (Union[None, np.float64], optional): Custom fiber length in millimeters. If None,
                then self.calculated_fiber_length is called to get the optimal length to achieve the target repetition rate.
                Default is None.

        Raises:
            ValueError: If the provided fiber length is not a positive, non-zero numerical value.
        """
        # Calculate the total material GDD
        self.calculate_free_space_length_without_components()
        print(f"Free space length without components: {self.free_space_minus_components:.3f}")

        # Check whether the fiber length is provided or needs to be calculated
        if provided_fiber_length is None:
            fiber_length = self.calculated_fiber_length(fiber_material)
        else:
            if not isinstance(provided_fiber_length, Union[int, float, np.float64]):
                raise ValueError("Fiber length must be a float for custom length or None for the optimal one.")
            
            if provided_fiber_length <= 0:
                raise ValueError("Fiber length must be positive and non-zero.")
            
            fiber_length = provided_fiber_length

        # Add the fiber component to the laser system with the desired material and length
        comp_fiber = ComponentClass(fiber_material, fiber_length, "loop", self)
        print(f"For a f_rep of {self.target_f_rep/1e6} MHz, a fiber length of {comp_fiber.length:.3f} mm is needed")
        #Sanity check whether all components wanted are included
        print(f"This is the number of components: {len(self.components)}")

        # Calculate the total material GDD
        self.material_gdd()
        print(f"Total GDD: {self.total_material_gdd[self.index]:.3f} ps^2")

        # Calculate the ideal grating separation
        self.ideal_grating_sep(self.gdd_grating(self.wavelength, 1))
        print(f"Ideal grating separation: {self.ideal_grating_separation:.3f} mm")

        # Calculate the real repetition frequency
        self.calculate_real_f_rep()
        print(f"Real f_rep: {self.real_f_rep:.3f} MHz")

        # Plot the dispersion curves if plotting is enabled
        if plotting:
            fig = plt.figure(figsize=(10,6))
            plt.plot(self.wavelength, self.total_material_gdd, c='r')
            plt.plot(self.wavelength, self.gdd_grating(self.wavelength, self.ideal_grating_separation), c='b')
            plt.plot(self.wavelength, self.total_material_gdd+self.gdd_grating(self.wavelength, self.ideal_grating_separation), c='g')
            plt.xlabel('Wavelength  ($\mu m$)')
            plt.ylabel('Group delay dispersion ($ps^2$)')
            plt.grid(axis='x', color='0.95')
            plt.grid(axis='y', color='0.95')
            plt.show()
        else:
            plt.close()

    @staticmethod
    def gdd_grating(
        wavelength: Union[NDArray[np.float64], Sequence[Sequence[float]]],
        grating_distance: Union[int, float, np.float64],
        grating_period: Union[int, float, np.float64]=1.0,
        alpha: Union[int, float, np.float64]=31.3
        ) -> NDArray[np.float64]:
        """
        Calculate the Group Delay Dispersion (GDD) for a grating.

        Parameters:
            lam (Union[NDArray[np.float64], list[np.float64]]): Wavelength values in micrometers.
            grating_distance (np.float64): Distance between the gratings in millimeters.
            grating_period (np.float64, optional): Grating period in micrometers. Default is 1 micrometer.
            alpha (np.float64, optional): Angle of incidence in degrees. Default is 31.3 degrees.

        Returns:
            NDArray[np.float64]: GDD values in ps² for each wavelength in the input list.
            
        Raises: 
            ValueError: If the grating distance or grating period is zero or negative.
            ValueError: If the alpha value is not between 0 and 90 degrees.

        Notes:
            - The input wavelengths (lam) are converted from micrometers to meters.
            - The input grating distance is converted from millimeters to meters.
            - The grating period is converted from micrometers to meters.
            - The output GDD values are converted from s² to ps².
        """
        # Unit conversions
        wavelength = np.asarray(wavelength, dtype=np.float64) * 1e-6  # Convert wavelength from micrometers to meters

        # Check if grating distance and grating period are not zero or negative
        if grating_distance <= 0:
            raise ValueError("Grating distance must be positive and non-zero.")
        if grating_period <= 0:
            raise ValueError("Grating period must be positive and non-zero.")
        #Check that alpha is between 0 and 90 degrees
        if not (0 <= alpha <= 90):
            raise ValueError("Alpha must be between 0 and 90 degrees.")

        # Unit conversions
        grating_distance = grating_distance * 1e-3  # Convert d from mm to meters
        grating_period = grating_period * 1e-6  # Convert grating_period from micrometers to meters
        alpha_rad = np.deg2rad(alpha)  # Convert to radians
        
        # Calculate the GDD for each wavelength (Equation from Mayer et al., 2020)
        term = (wavelength / grating_period - np.sin(alpha_rad)) ** 2
        denominator = np.power(1 - term, 1.5)
        gdd = -(wavelength**3 * grating_distance) / (pi * c**2 * grating_period**2) / denominator

        return np.asarray(gdd, dtype=np.float64)* 1e24  # Convert from s^2 to ps^2
        
class ComponentClass:
    """
    Represents a component in a laser system with defined material properties and dispersion characteristics.

    Attributes:
        material (MaterialClass): The material of the component, which defines its optical properties.
        length (Union[int, float,  np.float64]): The physical length of the component in millimeters.
        position (str): The position of the component within the laser system. Options are:
            - 'loop': Positioned within the fiber loop.
            - 'free_space': Positioned in free space.
        gdd (np.float64): The group delay dispersion (GDD) of the component in fs².
        laser (LaserClass | None): The laser instance this component is attached to, if any.

    Parameters:
        material (MaterialClass): A valid material instance that provides dispersion properties.
        length (float): The length of the component in millimeters.
        position (str): The position of the component, either 'loop' or 'free_space'.
        laser_instance (LaserClass, optional): A laser instance to attach the component to.

    Methods:
        attach_laser(laser: LaserClass) -> None:
            Attaches the component to a laser instance, updating both the component and the laser.

        gdd_calculator() -> float:
            Computes the Group Delay Dispersion (GDD) of the component based on its material and length.

    Raises:
        ValueError: If the position is invalid or if an invalid laser instance is provided.
        ValueError: If the length of the component is not positive and non-zero.
    """

    def __init__(
        self,
        material: "MaterialClass",
        length: Union[int, float, np.float64],
        position: str,
        laser_instance: Union["LaserClass", None]=None
    ) -> None:
        
        self.material = material
        self.length = length
        # Check if the length is positive and non-zero
        if self.length <= 0:
            raise ValueError("The length of the component must be positive and non-zero.")

        self.position = position

        # Validate position
        valid_positions = {"loop", "free_space"}
        if self.position not in valid_positions:
            raise ValueError(f"Invalid position '{self.position}'. Choose from {valid_positions}.")

        # Calculate and initialize GDD
        self.gdd = self.gdd_calculator()
        self.laser = laser_instance
        if self.laser:
            self.attach_laser(self.laser)

    def attach_laser(
        self,
        laser: "LaserClass"
        ) -> None:
        """
        Attaches the component to a laser instance, updating both the component and the laser.

        Parameters:
            laser (LaserClass): The laser instance to attach the component to.

        Raises:
            ValueError: If the provided laser instance is not valid.
        """
        if self.laser in LaserClass.instances:
            laser.add_component(self)
        else:
            raise ValueError("Invalid laser instance provided.")
        
    def gdd_calculator(
        self
        ) -> np.float64:
        """
        Calculates the Group Delay Dispersion (GDD) for the component based on its material and length.

        Returns:
            float: The calculated GDD value in fs².

        Raises:
            ValueError: If the material is not a valid instance of MaterialClass.
        """
        # Ensure material is a valid instance
        if not isinstance(self.material, MaterialClass):
            raise ValueError(f"Material {self.material} not found in MaterialClass.instances.")

        position_factor = 2 if self.position == "free_space" else 1
        return np.float64(position_factor * self.length * self.material.gvd)  # GDD in fs²

