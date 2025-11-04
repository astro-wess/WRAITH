import os
import sys
import cv2
import math
import numpy as np
import subprocess
from astropy.io import fits

# ===================================================================
# ESTIMATION HELPER FUNCTIONS
# (Extracted from generate_defconfig to be reusable)
# ===================================================================

def _get_major_axis_angle(hdu_data):
    """Calculates the galaxy's angle and ellipticity."""
    # Normalize values to 0-255 range
    hdu_data = np.nan_to_num(hdu_data)  # Remove NaNs
    hdu_data = (hdu_data - np.min(hdu_data)) / (np.max(hdu_data) - np.min(hdu_data)) * 255
    hdu_data = hdu_data.astype(np.uint8)
    
    # Apply adaptive threshold for segmentation
    _, binary = cv2.threshold(hdu_data, np.mean(hdu_data), 255, cv2.THRESH_BINARY)
    
    # Find isophote contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Warning: No contours found, using PA=0, ell=0.")
        return 0.0, 0.0
    
    # Select the largest contour (presumably the main galaxy)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit an ellipse to the contour
    if len(largest_contour) < 5:
        print("Warning: Contour too small, using PA=0, ell=0.")
        return 0.0, 0.0
    
    ellipse = cv2.fitEllipse(largest_contour)
    
    # Return the major axis angle
    angle = ellipse[2]
    
    # Get major (a) and minor (b) axis lengths
    a = max(ellipse[1]) / 2  # Major axis
    b = min(ellipse[1]) / 2  # Minor axis
    
    if a == 0: # Avoid division by zero
        return 0.0, 0.0

    # Calculate ellipticity
    ellipticity = 1 - (b / a)
    
    return angle, ellipticity

def _find_max_value_coordinates(image_data, center_radius):
    """Finds the brightest pixel near the center."""
    center_x, center_y = image_data.shape[1] // 2, image_data.shape[0] // 2
    
    # Ensure the radius doesn't go out of bounds
    ymin = max(0, center_y - center_radius)
    ymax = min(image_data.shape[0], center_y + center_radius + 1)
    xmin = max(0, center_x - center_radius)
    xmax = min(image_data.shape[1], center_x + center_radius + 1)

    center_region = image_data[ymin:ymax, xmin:xmax]
    
    if center_region.size == 0:
        print("Warning: Invalid central region, using image center.")
        return center_x, center_y

    max_value_coords = np.unravel_index(center_region.argmax(), center_region.shape)
    y0, x0 = max_value_coords[0] + ymin, max_value_coords[1] + xmin
    return x0, y0

def _find_pixel_count_until_value(image_data, x0, y0, reference_value):
    """Counts pixels from center until a reference value (r_e estimate)."""
    count = 0
    for i in range(y0, image_data.shape[0]):
        if i >= image_data.shape[0] or x0 >= image_data.shape[1]:
            break
        if image_data[i, x0] <= reference_value:
            break
        count += 1
    return max(1, count) # Ensure r_e is at least 1

# ===================================================================
# NEW FUNCTIONS (for --manual mode)
# ===================================================================

def _validate_param_input(user_input, param_name):
    """Validates user input (e.g., '150.0 100,200')."""
    parts = user_input.split()
    # Must be <value> <min,max>
    if len(parts) != 2:
        print(f"Invalid format. Use: <value> <min,max>. Ex: 150.0 100,200")
        return None
    
    value_str = parts[0]
    limits_str = parts[1]
    
    if ',' not in limits_str:
        print(f"Invalid limits format. Use: <min,max>. Ex: 100,200")
        return None
        
    try:
        # Just test if they are valid floats
        float(value_str)
        min_val, max_val = limits_str.split(',')
        float(min_val)
        float(max_val)
    except ValueError:
        print(f"Invalid values. All must be numbers.")
        return None
    
    # Return the formatted line for the config file
    return f"{param_name}\t{value_str}\t{limits_str}\n"

def manually_edit_config_file(config_file_path, functions_list):
    """Allows the user to interactively edit the config file."""
    
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    with open(config_file_path, 'r') as f:
        original_lines = f.readlines()
    
    new_lines = []
    func_index = 0 # To track which function from the list we are editing
    func_name_counts = {} # To handle duplicate names (e.g., Sersic_1, Sersic_2)
    
    print("-" * 50)
    print("Press [Enter] to keep the default value.")
    print("Or type the new value and limits in the format: <value> <min,max>")
    print("Example: 150.5 100,200")
    print("X0 and Y0 will be kept automatically.")
    print("-" * 50)

    for line in original_lines:
        line_stripped = line.strip()
        
        # If it's a 'FUNCTION' line
        if line_stripped.startswith("FUNCTION"):
            if func_index < len(functions_list):
                # Get the function name from the *list* (to keep order)
                func_name_original = functions_list[func_index]
                
                # Count occurrences (Sersic -> Sersic_1, Sersic -> Sersic_2)
                count = func_name_counts.get(func_name_original, 0) + 1
                func_name_counts[func_name_original] = count
                display_name = f"{func_name_original}_{count}"
                
                print(f"\n--- Editing Parameters for: {display_name} ---")
                func_index += 1
            
            new_lines.append(line)
            continue

        # If it's a comment or blank line
        if not line_stripped or line_stripped.startswith("#"):
            new_lines.append(line)
            continue
        
        # If it's a parameter line
        parts = line_stripped.split()
        if len(parts) < 3: # Not a valid parameter line (e.g., X0 100 99,101)
            new_lines.append(line)
            continue
        
        param_name = parts[0]
        
        # Skip X0 and Y0
        if param_name in ["X0", "Y0"]:
            new_lines.append(line)
            continue
        
        # It's an editable parameter (I_e, r_e, n, etc.)
        print(f"Current parameter: {line_stripped}")
        
        while True: # Validation loop
            user_input = input(f"New value for [ {param_name} ]: ").strip()
            
            if not user_input:
                # User pressed Enter
                new_lines.append(line)
                break
            
            # User typed something, validate it
            validated_line = _validate_param_input(user_input, param_name)
            if validated_line:
                new_lines.append(validated_line)
                break
            else:
                # If validation failed, the loop repeats
                print("Please try again.")

    # After the loop, rewrite the config file
    with open(config_file_path, 'w') as f:
        f.writelines(new_lines)

# ===================================================================
# NEW FUNCTION (for --funcs mode)
# ===================================================================

def generate_custom_config(name, functions_list, gal_name, gal_file):
    """Generates a custom config file from scratch."""
    
    config_name = os.path.splitext(gal_name)[0]
    config_dir = f'./{name}/configs'
    os.makedirs(config_dir, exist_ok=True)
    
    components_count = len(functions_list)
    # The path MUST be this one so run_imfit_with_check can find it
    config_file_path = os.path.join(config_dir, f'{config_name}{components_count}comp.txt')
    
    # 1. Get initial estimates
    center_radius = 20
    with fits.open(gal_file) as hdu:
        gal_hdu = hdu[1].data

    x0, y0 = _find_max_value_coordinates(gal_hdu, center_radius)
    ie_value_guess = math.floor(gal_hdu[y0, x0] / 2) # I_e (y, x)
    re_value_guess = _find_pixel_count_until_value(gal_hdu, x0, y0, ie_value_guess)
    PA_value, ell_value = _get_major_axis_angle(gal_hdu)
    
    # 2. Define parameter templates for functions
    # (Guess values and limits)
    x0_str = f"X0\t{x0 + 1}\t{x0 - 1},{x0 + 3}\n" # +1 for FITS (1-indexed)
    y0_str = f"Y0\t{y0 + 1}\t{y0 - 1},{y0 + 3}\n"
    
    # Limit estimates
    ie_low = max(0.1, ie_value_guess - 20)
    ie_high = ie_value_guess + 20
    re_low = max(1, re_value_guess - 20)
    re_high = re_value_guess + 20
    pa_low = (PA_value - 50) % 360
    pa_high = (PA_value + 50) % 360

    # Templates dictionary
    function_templates = {
        "Sersic": f"""
FUNCTION Sersic
I_e\t{ie_value_guess}\t{ie_low},{ie_high}
r_e\t{re_value_guess}\t{re_low},{re_high}
n\t1.0\t0.1,8.0
PA\t{PA_value}\t{pa_low},{pa_high}
ell\t{ell_value}\t0.0,0.99
{x0_str}{y0_str}""",

        "FerrersBar2D": f"""
FUNCTION FerrersBar2D
I_0\t{ie_value_guess}\t{ie_low},{ie_high}
r_bar\t{re_value_guess}\t{re_low},{re_high + 30}
alpha\t2.0\t1.0, 4.0
beta\t2.0\t1.0, 4.0
ell_bar\t{ell_value}\t0.0,0.99
PA_bar\t{PA_value}\t{pa_low},{pa_high}
{x0_str}{y0_str}""",

        "GaussianRing": f"""
FUNCTION GaussianRing
I_0\t{ie_value_guess / 2}\t{ie_low / 2},{ie_high / 2}
r_ring\t{re_value_guess * 1.5}\t{re_low},{re_high + 50}
width\t5.0\t1.0, 30.0
ell_ring\t{ell_value}\t0.0,0.99
PA_ring\t{PA_value}\t{pa_low},{pa_high}
{x0_str}{y0_str}""",

        "BrokenExponential": f"""
FUNCTION BrokenExponential
I_0\t{ie_value_guess}\t{ie_low},{ie_high}
h_1\t{re_value_guess}\t{re_low},{re_high}
h_2\t{re_value_guess * 2}\t{re_low},{re_high * 3}
r_break\t{re_value_guess * 1.5}\t{re_low * 0.5},{re_high * 1.5}
PA\t{PA_value}\t{pa_low},{pa_high}
ell\t{ell_value}\t0.0,0.99
{x0_str}{y0_str}"""
    }

    # 3. Write the configuration file
    with open(config_file_path, 'w') as f:
        f.write(f"# Auto-generated config file for {gal_name}\n")
        f.write(f"# Functions: {', '.join(functions_list)}\n\n")
        
        for func_name in functions_list:
            # Find the function in the dictionary (case-insensitive)
            func_key = next((key for key in function_templates if key.lower() == func_name.lower()), None)
            
            if func_key:
                f.write(function_templates[func_key])
                f.write("\n\n") # Separator
            else:
                print(f"Warning: Template for function '{func_name}' not found. Skipping.")
    
    print(f"Custom config saved to: {config_file_path}")
    
    # Return component count and path
    return components_count, config_file_path

# ===================================================================
# ORIGINAL FUNCTION (for sample mode) - MODIFIED
# ===================================================================

def generate_defconfig(name, components, gal_name, gal_file):
    """Generates config file from Sersic template (sample mode)."""
    
    config_name = os.path.splitext(gal_name)[0]
    config_dir = f'./{name}/configs'
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)

    def_file = f'./wraith/{components}comp.txt'
    
    # Use the extracted helper functions
    center_radius = 20
    with fits.open(gal_file) as hdu:
        gal_hdu = hdu[1].data

    x0, y0 = _find_max_value_coordinates(gal_hdu, center_radius)
    ie_value = math.floor(gal_hdu[y0, x0] / 2) # I_e (y, x)
    re_value = _find_pixel_count_until_value(gal_hdu, x0, y0, ie_value)
    PA_value, ell_value = _get_major_axis_angle(gal_hdu)

    def update_config_lines(config_lines, ie_value, re_value, x0, y0, PA, ell):
        # This internal function remains the same
        updated_params = {'ell1': False, 'ell2': False, 'PA1': False, 'PA2': False, 'I_e_1': False, 'r_e_1': False, 'I_e_2': False, 'r_e_2': False, 'X0': False, 'Y0': False}
        
        PA_low = (PA - 50) % 360
        PA_high = (PA + 50) % 360
        
        ie_low_1 = max(5, ie_value - 20)
        ie_high_1 = ie_value + 20
        re_low_1 = max(1, re_value - 20)
        re_high_1 = re_value + 20
        
        ie_low_2 = 0.1  # Avoid overlap with I_e_1
        ie_high_2 = min(20, ie_low_1)
        re_low_2 = re_high_1 + 1  # Avoid overlap with r_e_1
        re_high_2 = re_low_2 + 50
        
        ie_value_2 = (ie_low_2 + ie_high_2) // 2  # Average of limits
        re_value_2 = (re_low_2 + re_high_2) // 2  # Average of limits
        
        sersic_count = 0
        
        for i, line in enumerate(config_lines):
            if line.startswith("FUNCTION") and "Sersic" in line:
                sersic_count += 1
            
            if sersic_count == 1:
                if line.startswith("I_e") and not updated_params['I_e_1']:
                    config_lines[i] = f"I_e\t{ie_value}\t{ie_low_1},{ie_high_1}\n"
                    updated_params['I_e_1'] = True
                elif line.startswith("r_e") and not updated_params['r_e_1']:
                    config_lines[i] = f"r_e\t{re_value}\t{re_low_1},{re_high_1}\n"
                    updated_params['r_e_1'] = True
                elif line.startswith("PA") and not updated_params['PA1']:
                    config_lines[i] = f"PA\t{PA}\t{PA_low},{PA_high}\n"
                    updated_params['PA1'] = True
                elif line.startswith("ell") and not updated_params['ell1']:
                    config_lines[i] = f"ell\t{ell}\t0.,0.99\n"
                    updated_params['ell1'] = True     
            elif sersic_count == 2:
                if line.startswith("I_e") and not updated_params['I_e_2']:
                    config_lines[i] = f"I_e\t{ie_value_2}\t{ie_low_2},{ie_high_2}\n"
                    updated_params['I_e_2'] = True
                elif line.startswith("r_e") and not updated_params['r_e_2']:
                    config_lines[i] = f"r_e\t{re_value_2}\t{re_low_2},{re_high_2}\n"
                    updated_params['r_e_2'] = True
                elif line.startswith("PA") and not updated_params['PA2']:
                    config_lines[i] = f"PA\t{PA}\t{PA_low},{PA_high}\n"
                    updated_params['PA2'] = True
                elif line.startswith("ell") and not updated_params['ell2']:
                    config_lines[i] = f"ell\t{ell}\t0.,0.99\n"
                    updated_params['ell2'] = True
            
            if line.startswith("X0") and not updated_params['X0']:
                config_lines[i] = f"X0\t{x0 + 1}\t{x0 - 1},{x0 + 3}\n"
                updated_params['X0'] = True
            elif line.startswith("Y0") and not updated_params['Y0']:
                config_lines[i] = f"Y0\t{y0 + 1}\t{y0 - 1},{y0 + 3}\n"
                updated_params['Y0'] = True
            
            # Adjustment to ensure all params are updated
            # even with only 1 Sersic component
            if components == '1' and sersic_count == 1:
                if all([updated_params['I_e_1'], updated_params['r_e_1'], updated_params['PA1'], updated_params['ell1'], updated_params['X0'], updated_params['Y0']]):
                    break
            elif components == '2' and sersic_count == 2:
                 if all(updated_params.values()):
                    break
        
        return config_lines

    if not os.path.exists(def_file):
        print(f"Error: Template file not found: {def_file}")
        return

    with open(def_file, 'r') as f:
        config_lines = f.readlines()

    config_lines = update_config_lines(config_lines, ie_value, re_value, x0, y0, PA_value, ell_value)

    config_file_path = os.path.join(config_dir, f'{config_name}{components}comp.txt')
    with open(config_file_path, 'w') as f:
        f.writelines(config_lines)

    # Return component count and path
    # (The incoming 'components' is a string, we return int)
    return int(components), config_file_path


# ===================================================================
# EXECUTION FUNCTIONS (Unchanged)
# ===================================================================

def run_imfit(name, components, gal_name, gal_file, imfit_type):
    
    def_name = os.path.splitext(gal_name)[0]
    psf_file = f'./{name}/psf/{def_name}psf.fits'
    mask_file = f'./{name}/mask/{def_name}mask.fits'
    config_file = f'./{name}/configs/{def_name}{components}comp.txt'
    results_dir = f'./{name}/results'
    components_int = int(components)

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    files = [gal_file, psf_file, mask_file, config_file]

    if all(os.path.exists(file) for file in files):

        command = f'./wraith/imfit {gal_file}[1] --config {config_file} --mask {mask_file}[1] --psf {psf_file} --noise {gal_file}[3] --save-model {results_dir}/{def_name}_mod{components}comp.fits --save-residual {results_dir}/{def_name}_res{components}comp.fits --save-params {results_dir}/{def_name}_best{components}comp.dat --errors-are-variances --{imfit_type}'
        os.system(command)

        if components_int > 1:
            command_flux = f"./wraith/makeimage {results_dir}/{def_name}_best{components}comp.dat --refimage {gal_file}[1] --save-fluxes {results_dir}/{def_name}_fluxes{components}comp.txt"
            os.system(command_flux)
    else:
        print(f"Error: Not all required files for imfit exist.")
        print(f"Checking: {files}")
        for file in files:
            if not os.path.exists(file):
                print(f"  Missing file: {file}")
                

def run_imfit_with_check(name, components, gal_name, gal_file, imfit_type):
    def_name = os.path.splitext(gal_name)[0]
    psf_file = f'./{name}/psf/{def_name}psf.fits'
    mask_file = f'./{name}/mask/{def_name}mask.fits'
    config_file = f'./{name}/configs/{def_name}{components}comp.txt'
    results_dir = f'./{name}/results'
    log_dir = f'./log_reports'  # Directory to save logs
    components_int = int(components)
    max_attempts = 5
    attempts = 0
    log_content = ""

    # Create log folder if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    files = [gal_file, psf_file, mask_file, config_file]

    if not all(os.path.exists(file) for file in files):
        print(f"Error: Not all required files for imfit exist.")
        print(f"Checking: {files}")
        for file in files:
            if not os.path.exists(file):
                print(f"  Missing file: {file}")
        return # Do not run Imfit if files are missing

    command = f'./wraith/imfit {gal_file}[1] --config {config_file} --mask {mask_file}[1] --psf {psf_file} --noise {gal_file}[3] --save-model {results_dir}/{def_name}_mod{components}comp.fits --save-residual {results_dir}/{def_name}_res{components}comp.fits --save-params {results_dir}/{def_name}_best{components}comp.dat --errors-are-variances --{imfit_type}'
    
    # Execute the command and capture terminal output
    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        # Process output in real-time
        for stdout_line in iter(proc.stdout.readline, ""):
            sys.stdout.write(stdout_line)  # Display in console
            log_content += stdout_line  # Save to log
        for stderr_line in iter(proc.stderr.readline, ""):
            sys.stderr.write(stderr_line)  # Display in console
            log_content += stderr_line  # Save to log

        proc.stdout.close()
        proc.stderr.close()
        proc.wait()

    while True:
        # Check if adjustment is needed
        param_name, limit_type, limit_value, adjustment_needed = check_for_limits(results_dir, def_name, components_int, config_file)

        if not adjustment_needed:
            print("No adjustments needed.")
            break  # Exit loop if nothing to adjust

        if attempts >= max_attempts:
            print(f"Maximum of {max_attempts} attempts reached. Saving log for {def_name}.")
            log_path = os.path.join(log_dir, f"{def_name}.txt")
            with open(log_path, 'w') as log_file:
                log_file.write(log_content)
            break  # Exit loop if max attempts reached

        adjust_limits(results_dir, def_name, components_int, config_file)

        # Delete old result files *after* adjusting limits
        try:
            os.remove(f'{results_dir}/{def_name}_mod{components}comp.fits')
            os.remove(f'{results_dir}/{def_name}_res{components}comp.fits')
            os.remove(f'{results_dir}/{def_name}_best{components}comp.dat')
        except FileNotFoundError:
            print("Warning: Previous result files not found for removal.")


        # Re-run the command after adjusting limits and capture log
        with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
            # Process output in real-time
            for stdout_line in iter(proc.stdout.readline, ""):
                sys.stdout.write(stdout_line)  # Display in console
                log_content += stdout_line  # Save to log
            for stderr_line in iter(proc.stderr.readline, ""):
                sys.stderr.write(stderr_line)  # Display in console
                log_content += stderr_line  # Save to log

            proc.stdout.close()
            proc.stderr.close()
            proc.wait()

        attempts += 1  # Only count attempt when an adjustment was made

    if components_int > 1:
        best_fit_file = f"{results_dir}/{def_name}_best{components}comp.dat"
        if os.path.exists(best_fit_file):
            command_flux = f"./wraith/makeimage {best_fit_file} --refimage {gal_file}[1] --save-fluxes {results_dir}/{def_name}_fluxes{components}comp.txt"
            os.system(command_flux)
        else:
            print(f"Warning: File {best_fit_file} not found. Skipping 'makeimage'.")

def check_for_limits(results_dir, def_name, components, config_file):
    # Check parameters in results file against limits in config file
    param_file = f'{results_dir}/{def_name}_best{components}comp.dat'
    
    if not os.path.exists(param_file):
        print(f"Warning: Parameter file {param_file} not found. Skipping limit check.")
        return None, None, None, False
        
    with open(param_file, 'r') as f:
        lines = f.readlines()

    # Read the config file
    if not os.path.exists(config_file):
        print(f"Warning: Config file {config_file} not found. Skipping limit check.")
        return None, None, None, False

    with open(config_file, 'r') as f:
        config_lines = f.readlines()

    for line in lines:
        # Ignore empty and commented lines
        line = line.strip()
        if not line or line.startswith('#') or 'FUNCTION' in line:
            continue

        # Extract fitted parameter name and value
        try:
            parts = line.split()
            if len(parts) < 2:
                continue
            param_name, param_value = parts[0], float(parts[1])
        except (IndexError, ValueError):
            # Ignore any line without two values
            continue
            
        # Ignore X0 and Y0
        if param_name in ['X0', 'Y0']:
            continue  

        # Find the parameter in the config file
        for config_line in config_lines:
            config_parts = config_line.split()
            if not config_parts: # Empty line
                continue
            
            if param_name == config_parts[0]:
                # Read min and max value for the parameter
                try:
                    if 'fixed' in config_parts[2]:
                        continue # Fixed parameter, no limits to check
                        
                    min_val_str, max_val_str = config_parts[2].split(',')
                    min_val, max_val = float(min_val_str), float(max_val_str)
                except (IndexError, ValueError):
                    # Ignore lines where limits aren't valid
                    continue

                # Check if the fitted value is outside the limits
                tolerance = 0.05 * min_val  # 5% of min_val
                if abs(param_value - min_val) <= abs(tolerance):
                    print(f"Limit reached: {param_name} hit lower limit: {param_value} (limit was {min_val})")
                    return param_name, 'lower', param_value, True  # Return values and True (adjustment needed)
                
                tolerancemax = 0.05 * max_val  # 5% of max_val
                if abs(param_value - max_val) <= abs(tolerancemax):
                    print(f"Limit reached: {param_name} hit upper limit: {param_value} (limit was {max_val})")
                    return param_name, 'upper', param_value, True  # Return values and True (adjustment needed)
                
                # If parameter was found, no need to search more
                break

    return None, None, None, False


def adjust_limits(results_dir, def_name, components, config_file):
    # Function to adjust config file limits
    if not os.path.exists(config_file):
        print(f"Error: Config file {config_file} not found. Cannot adjust limits.")
        return

    with open(config_file, 'r') as f:
        lines = f.readlines()

    # Check if any parameter hit a limit
    param_name, limit_type, param_value, adjustment_needed = check_for_limits(results_dir, def_name, components, config_file)

    if not adjustment_needed:
        print("No limits hit. No adjustments made.")
        return  # If no limit to adjust, exit function

    # Flag to check if any limit was adjusted
    adjusted = False
    new_lines = []

    for line in lines:
        # Ignore commented or function lines
        if line.startswith('#') or line.startswith('FUNCTION'):
            new_lines.append(line)
            continue

        # Split line and check if param_name is in it
        parts = line.split()
        if len(parts) < 3:
            new_lines.append(line)
            continue

        # Check if line contains param_name and adjust limits
        if param_name == parts[0] and not 'fixed' in parts[2]:
            try:
                min_val_str, max_val_str = parts[2].split(',')
                min_val, max_val = float(min_val_str), float(max_val_str)
                current_val = float(parts[1]) # Parameter's central value

                if limit_type == 'lower':
                    tolerance = 0.05 * min_val
                    if abs(param_value - min_val) <= abs(tolerance):
                        new_min_val = min_val * 0.8  # Decrease lower limit by 20%
                        new_lines.append(f'{param_name}\t{current_val}\t{new_min_val},{max_val}\n')
                        print(f"Adjusting lower limit of {param_name} to {new_min_val}")
                        adjusted = True
                    else:
                        new_lines.append(line)
                elif limit_type == 'upper':
                    tolerance = 0.05 * max_val
                    if abs(param_value - max_val) <= abs(tolerance):
                        new_max_val = max_val * 1.2  # Increase upper limit by 20%
                        new_lines.append(f'{param_name}\t{current_val}\t{min_val},{new_max_val}\n')
                        print(f"Adjusting upper limit of {param_name} to {new_max_val}")
                        adjusted = True
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            except ValueError:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    # Write the lines back to the config file
    with open(config_file, 'w') as f:
        f.writelines(new_lines)

    if adjusted:
        print("Limits adjusted successfully.")
    else:
        # This can happen if check_for_limits finds a limit,
        # but the adjustment logic fails to find it (e.g., re-check)
        print("No limits were adjusted (double check).")