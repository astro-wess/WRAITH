#!/usr/bin/env python3
import os
import argparse
from wraith import mask, prologue, run_imfit, process_data
from wraith.params import Params


def process_galaxy(params, gal_file, filename, list_file, custom_funcs=None, manual_edit=False):
    """Executes the full pipeline for a single galaxy."""
    print(f"Processing {filename}...")

    # 1. Masking (common to both modes)
    mask.make_maps(gal_file)
    mask.combine_maps()
    mask.remove_central_object()
    mask.apply_mask(params.name, filename, gal_file)

    # 2. Execution Logic Branch
    if custom_funcs:
        # =================================================
        # BRANCH 1: SINGLE GALAXY MODE (--funcs was used)
        # =================================================
        # This mode runs ONCE based on --funcs, not consecutive_runs
        
        print(f"Generating custom configuration with: {', '.join(custom_funcs)}")
        components_to_run, config_file_path = run_imfit.generate_custom_config(params.name, custom_funcs, filename, gal_file)
        
        if manual_edit:
            print(f"Entering manual edit mode for: {config_file_path}")
            try:
                run_imfit.manually_edit_config_file(config_file_path, custom_funcs)
                print("Manual editing complete.")
            except Exception as e:
                print(f"Error during manual editing: {e}. Aborting.")
                return

        # Use the 'checks1' param as the default for single mode checks
        max_checks = params.checks1 
        
        print(f"Running Imfit (Single Mode) with {components_to_run} components...")
        run_imfit.run_imfit_with_check(
            params.name, 
            str(components_to_run), 
            filename, 
            gal_file, 
            params.imfit_type, # Single mode can just use the default imfit_type
            max_checks
        )

        # Call the new "resetting" results function
        process_data.get_results_single(params.name, filename, gal_file, list_file, str(components_to_run), params.filter)

    else:
        # =================================================
        # BRANCH 2: SAMPLE MODE (default)
        # =================================================
        # This mode loops through consecutive_runs
        
        previous_best_fit_file = None
        last_components_run = "0"
        
        print(f"Starting Sample Mode processing with {params.consecutive} consecutive run(s)...")

        for i in range(params.consecutive):
            run_info = params.run_definitions[i]
            components_to_run = str(run_info['components'])
            imfit_type_to_run = run_info['imfit_type']
            checks_for_this_run = run_info['checks']
            
            print(f"--- Running Pass {i+1}/{params.consecutive} (Components: {components_to_run}) ---")

            # Config Generation Logic
            if i == 0:
                # First run: Use standard default config
                print("Generating default config for first pass...")
                _, config_file_path = run_imfit.generate_defconfig(params.name, components_to_run, filename, gal_file)
            else:
                # Subsequent runs: Use chained config
                if previous_best_fit_file is None:
                    print("Error: Chained run (i > 0) has no previous best_fit file. Aborting.")
                    return
                print(f"Generating chained config from: {previous_best_fit_file}")
                _, config_file_path = run_imfit.generate_chained_config(params.name, components_to_run, filename, gal_file, previous_best_fit_file)

            # Run Imfit
            run_imfit.run_imfit_with_check(
                params.name, 
                components_to_run, 
                filename, 
                gal_file, 
                imfit_type_to_run,
                checks_for_this_run
            )
            
            # Save path for the next loop (or for get_results)
            def_name = os.path.splitext(filename)[0]
            previous_best_fit_file = f'./{params.name}/results/{def_name}_best{components_to_run}comp.dat'
            last_components_run = components_to_run

            if not os.path.exists(previous_best_fit_file):
                print(f"Error: Best-fit file {previous_best_fit_file} was not created. Stopping chained run.")
                return # Stop if a run fails

        # After ALL loops, save the results of the FINAL run
        print("Chained runs complete. Saving final results...")
        process_data.get_results(params.name, filename, gal_file, list_file, last_components_run, params.filter)

    print(f"Galaxy {filename} processed successfully!\n")


def main():
    """
    The main entry point for the WRAITH pipeline script.

    This function parses command-line arguments to determine the operating mode (either 'sample' or 'single')
    and orchestrates the galaxy processing workflow. In 'sample' mode, it iterates over a list of galaxies
    from a CSV file, downloading data and running the pipeline for each. In 'single' mode, it processes a
    single galaxy, which can be specified either by a local FITS file or by celestial coordinates for download.
    It handles the initialization of parameters, directory setup, and calls the `process_galaxy` function
    to execute the core pipeline.

    The script is designed to be run from the command line and uses `argparse` to manage its various options
    and subcommands, providing a flexible interface for different use cases.
    """
    parser = argparse.ArgumentParser(description="Pipeline to run IMFIT on samples or single galaxies.")

    # General arguments
    parser.add_argument('-c', '--config', type=str, required=True, help='Configuration file (e.g., S0config.txt)')
    parser.add_argument('-l', '--list', type=str, help='Sample CSV file (required for sample mode)')
    parser.add_argument('--skip-downloads', action='store_true', help='Skips directory generation and downloads')

    # Subcommands: "sample" and "single"
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # ---- Sample Mode ----
    sample_parser = subparsers.add_parser('sample', help='Process an entire sample (default mode)')

    # ---- Single Mode ----
    single_parser = subparsers.add_parser('single', help='Process a single galaxy')
    single_parser.add_argument('--fits', type=str, help='Local FITS file for the galaxy')
    single_parser.add_argument('--ra', type=float, help='RA coordinate (degrees)')
    single_parser.add_argument('--dec', type=float, help='DEC coordinate (degrees)')
    single_parser.add_argument('--size', type=float, default=30, help='Image size in arcsec (default=30)')
    single_parser.add_argument('--funcs', nargs='+', help='List of Imfit functions (e.g., Sersic FerrersBar2D)')
    single_parser.add_argument('-m', '--manual', action='store_true', help='Activates manual parameter editing mode (requires --funcs)')


    args = parser.parse_args()

    # Load parameters
    params = Params()
    prologue.read_params(params, args.config)

    # === SINGLE GALAXY Mode ===
    if args.mode == 'single':
        os.makedirs(params.name, exist_ok=True)

        if not args.skip_downloads:
            gal_dir = f'./{params.name}/gal'
            psf_dir = f'./{params.name}/psf'
            os.makedirs(gal_dir, exist_ok=True)
            os.makedirs(psf_dir, exist_ok=True)

            if args.fits:
                # Local FITS provided
                gal_file = args.fits
                print(f"Using local FITS file: {gal_file}")

            elif args.ra and args.dec:
                # Download via coordinates
                gal_txt = f'./{params.name}/{params.name}_single_gal.txt'
                psf_txt = f'./{params.name}/{params.name}_single_psf.txt'
                up_filter = params.filter.upper()

                with open(gal_txt, 'w') as f:
                    f.write("#? filter ra dec sw sh rerun image mask variance type\n")
                    f.write(f"HSC-{up_filter} {args.ra} {args.dec} {args.size}asec {args.size}asec pdr3_wide true true true coadd\n")

                with open(psf_txt, 'w') as f:
                    f.write("#? rerun ra dec filter type\n")
                    f.write(f"pdr3_wide {args.ra} {args.dec} HSC-{up_filter} coadd\n")

                print("Downloading galaxy files...")
                os.system(f'python3 ./wraith/downloadCutout.py --list {gal_txt} --user {params.user_id} --password {params.user_pass} --name "single{params.filter}"')
                os.system(f'python3 ./wraith/downloadPsf.py --list {psf_txt} --user {params.user_id} --password {params.user_pass} --name "single{params.filter}psf"') # Added 'psf' to name

                # Search for galaxy files
                gal_files = [f for f in os.listdir('.') if f.endswith(f'{params.filter}.fits')]
                if not gal_files:
                    print("Error: No galaxy FITS file found after download.")
                    return
                gal_file_name = gal_files[0]
                gal_file_path = os.path.join(gal_dir, gal_file_name)
                os.rename(gal_file_name, gal_file_path)
                gal_file = gal_file_path # Update gal_file variable with the full path
                
                # Search for PSF files
                psf_files = [f for f in os.listdir('.') if f.endswith(f'{params.filter}psf.fits')]
                if not psf_files:
                    print("Error: No PSF FITS file found after download.")
                    return
                psf_file_name = psf_files[0]
                os.rename(psf_file_name, os.path.join(psf_dir, psf_file_name))

                print(f"Download complete: {gal_file_name}")
            else:
                print("Error: Provide --fits or (--ra and --dec) to run in 'single' mode.")
                return
        else:
            gal_dir = f'./{params.name}/gal'
            fits_files = [f for f in os.listdir(gal_dir) if f.endswith('.fits')]
            if not fits_files:
                print("Error: No FITS file found in gal/ directory.")
                return
            gal_file = os.path.join(gal_dir, fits_files[0])
            print(f"Using existing FITS: {gal_file}")

        filename = os.path.basename(gal_file)
        process_galaxy(params, gal_file, filename, args.list or "single.csv", custom_funcs=args.funcs, manual_edit=args.manual)
        return

    # === SAMPLE Mode ===
    if not args.list:
        print("Error: The sample CSV must be provided with -l for 'sample' mode.")
        return

    if not args.skip_downloads:
        prologue.generate_dir(args.list, params.name, params.filter)
        prologue.download_files(params.name, params.filter, params.user_id, params.user_pass)

    gal_dir = f'./{params.name}/gal'
    galaxy_files = [f for f in os.listdir(gal_dir) if f.endswith('.fits')]
    total_files = len(galaxy_files)

    for idx, filename in enumerate(galaxy_files, 1):
        gal_file = os.path.join(gal_dir, filename)
        print(f"{idx}/{total_files} - Processing {filename}...")
        process_galaxy(params, gal_file, filename, args.list) # Defaults: custom_funcs=None, manual_edit=False


if __name__ == '__main__':
    main()
