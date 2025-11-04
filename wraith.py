#!/usr/bin/env python3
import os
import argparse
from wraith import mask, prologue, run_imfit, process_data
from wraith.params import Params


def process_galaxy(params, gal_file, filename, list_file, custom_funcs=None, manual_edit=False):
    """Executes the full pipeline for a single galaxy."""
    print(f"Processing {filename}...")

    # Generate maps and mask
    mask.make_maps(gal_file)
    mask.combine_maps()
    mask.remove_central_object()
    mask.apply_mask(params.name, filename, gal_file)

    # Config Generation and Run Logic
    components_to_run = params.components # Default
    imfit_type_to_run = params.imfit_type
    config_file_path = "" # We need the path for manual editing

    if custom_funcs:
        # Single Mode with --funcs
        print(f"Generating custom configuration with: {', '.join(custom_funcs)}")
        # This function must now return (count, file_path)
        components_to_run, config_file_path = run_imfit.generate_custom_config(params.name, custom_funcs, filename, gal_file)
        
        # Manual editing block
        if manual_edit:
            print(f"Entering manual edit mode for: {config_file_path}")
            try:
                # We pass the functions list to name Sersic_1, Sersic_2, etc.
                run_imfit.manually_edit_config_file(config_file_path, custom_funcs)
                print("Manual editing complete.")
            except Exception as e:
                print(f"Error during manual editing: {e}. Aborting.")
                return

    elif params.use_def == 1:
        # Sample Mode (or single without --funcs)
        print(f"Generating default Sersic configuration ({params.components} comps)...")
        # This function must also return (count, file_path)
        components_count, config_file_path = run_imfit.generate_defconfig(params.name, str(params.components), filename, gal_file)
        components_to_run = components_count
    
    else:
        # Mode with custom config file (IMFIT_CONFIGFILE)
        print(f"Using specified config file: {params.imfit_configfile}")
        config_file_path = params.imfit_configfile # Assumes this is a valid path
        components_to_run = params.components 

    # Run Imfit (run_imfit_with_check uses the correct config file)
    print(f"Running Imfit with {components_to_run} components...")
    run_imfit.run_imfit_with_check(params.name, str(components_to_run), filename, gal_file, imfit_type_to_run)

    # Extract results
    process_data.get_results(params.name, filename, gal_file, list_file, str(components_to_run), params.filter)
    print(f"Galaxy {filename} processed successfully!\n")


def main():
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