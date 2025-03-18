from setuptools import setup, find_packages

setup(
    name="VPfunctions",
    version="0.5.1",
    description="A library for extracting, processing, and visualizing data, related to the measurement equipment of the Optical Metrology Research Group at the University of Vienna.",
    author="Vito Fabian Pecile",
    author_email="vito.pecile@univie.ac.at",
    url="https://github.com/PolarimeterVito/VPfunctions",
    packages=find_packages(),
    package_data={
        "VPfunctions": ["*.pyi", "py.typed"],
    },
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "pandas",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
