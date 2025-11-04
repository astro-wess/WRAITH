# WRAITH: A Pipeline for Galaxy Image Analysis with IMFIT

WRAITH (Workflow for Running Astronomical Image-fitting Tasks with HSC) is a Python-based pipeline designed to automate the process of fitting astronomical images with IMFIT. It handles the entire workflow, from downloading data and generating masks to running IMFIT and processing the results.

## Features

-   **Automated Data Downloads**: Downloads galaxy cutout images and Point Spread Functions (PSFs) from the Hyper Suprime-Cam (HSC) data release.
-   **Mask Generation**: Automatically creates segmentation maps and masks to isolate the galaxy of interest.
-   **Flexible IMFIT Configuration**: Supports multiple modes for IMFIT configuration:
    -   Default Sersic profiles for batch processing.
    -   Custom configuration files.
    -   On-the-fly generation of configurations with user-specified functions.
-   **Interactive Parameter Editing**: Allows for manual fine-tuning of initial parameter guesses before running IMFIT.
-   **Robust Fitting**: Iteratively adjusts parameter limits to ensure a reliable fit.
-   **Results Processing**: Extracts and organizes the best-fit parameters and statistical criteria into a CSV file.

## Usage

WRAITH can be run in two main modes: `sample` for processing a list of galaxies, and `single` for processing a single galaxy.

### Sample Mode

In `sample` mode, WRAITH processes a list of galaxies from a CSV file. The CSV file should contain the coordinates and other relevant information for each galaxy.

```bash
python3 wraith.py -c <config_file> -l <sample_csv_file> sample
```

### Single Mode

In `single` mode, WRAITH processes a single galaxy, which can be specified either by a local FITS file or by its celestial coordinates.

#### Using a local FITS file:

```bash
python3 wraith.py -c <config_file> single --fits <path_to_fits_file>
```

#### Using coordinates:

```bash
python3 wraith.py -c <config_file> single --ra <ra> --dec <dec>
```

## Configuration

WRAITH is configured through a text file (e.g., `S0config.txt`). This file specifies various parameters, including:

-   HSC user credentials.
-   Image size and filter.
-   IMFIT settings (e.g., number of components, function types).

## Dependencies

-   Python 3
-   Astropy
-   NumPy
-   OpenCV
-   Pandas
-   Matplotlib
-   Source Extractor
-   IMFIT

## Project Structure

-   `wraith.py`: The main entry point for the pipeline.
-   `wraith/`: A directory containing the core Python modules:
    -   `params.py`: Defines the `Params` class for holding configuration parameters.
    -   `prologue.py`: Handles parameter reading, directory generation, and data downloads.
    -   `mask.py`: Contains functions for generating and applying masks.
    -   `run_imfit.py`: Manages the generation of IMFIT configuration files and the execution of IMFIT.
    -   `process_data.py`: Extracts and processes the results from IMFIT.
    -   `downloadCutout.py`: A script for downloading FITS cutout images.
    -   `downloadPsf.py`: A script for downloading PSF images.
    -   `images.py`: A utility for plotting FITS images.
