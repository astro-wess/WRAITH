import os
import re
import csv

def get_results(name, gal_name, gal_file, output_csv, components, filter):
    result_dir = f'./{name}/results'
    
    # Extract the base name (e.g., "2r")
    base_name_match = re.match(r'(\d+[a-zA-Z]+)', gal_name)
    if not base_name_match:
        print(f"Error: Could not parse galaxy name {gal_name}")
        return
    base_name = base_name_match.group(1) # e.g., "2r"
    
    # file_number is the integer part
    file_number = int(re.findall(r'\d+', base_name)[0]) # e.g., 2

    # 1. Read the .dat file and parse ALL parameters
    # =================================================
    arquivo_dat = f'{base_name}_best{components}comp.dat'
    caminho_arquivo = os.path.join(result_dir, arquivo_dat)
    if not os.path.exists(caminho_arquivo):
        print(f"File {caminho_arquivo} not found.")
        return

    new_params = {} # Will store all new values {col_name: value}
    component_count = 0
    func_names = [] # To store "Sersic", "FerrersBar2D", etc.

    with open(caminho_arquivo, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('#   Reduced value:'):
                new_params['Reduced_value'] = line.split(': ')[1].strip()
            elif line.startswith('#   AIC:'):
                new_params['AIC'] = line.split(': ')[1].strip()
            elif line.startswith('#   BIC:'):
                new_params['BIC'] = line.split(': ')[1].strip()
            elif line.startswith('FUNCTION'):
                component_count += 1
                # Store function name, e.g., "Sersic"
                func_names.append(line.split()[1]) 
            
            # Check if it's a parameter line (not a comment, not blank, not FUNCTION)
            elif not line.startswith('#') and line.split():
                parts = line.split()
                if len(parts) >= 2:
                    param_name = parts[0]
                    param_value = parts[1]
                    
                    # Skip X0, Y0
                    if param_name in ['X0', 'Y0']:
                        continue
                        
                    # Create dynamic column name: param_name + _ + component_index
                    col_name = f"{param_name}_{component_count}"
                    new_params[col_name] = param_value

    # 2. Read the CSV and update headers
    # =================================================
    with open(output_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        linhas_csv = list(reader)

    header_row = linhas_csv[0]
    existing_headers = set(header_row)
    
    # Find only the headers that are missing
    all_new_headers = list(new_params.keys())
    missing_headers = [h for h in all_new_headers if h not in existing_headers]

    if missing_headers:
        print(f"Adding new columns to CSV: {', '.join(missing_headers)}")
        header_row.extend(missing_headers)
        # Add '0' placeholders to all existing data rows
        for i in range(1, len(linhas_csv)):
            linhas_csv[i].extend(['0'] * len(missing_headers))

    # 3. Create Header Map and update the correct row
    # =================================================
    header_map = {header_name: index for index, header_name in enumerate(header_row)}

    row_index_to_update = file_number - 1 # e.g., file_number 2 updates index 1
    
    # Robust check for valid data row index
    if row_index_to_update >= 1 and row_index_to_update < len(linhas_csv):
        row_to_update = linhas_csv[row_index_to_update]
        
        # Update all values using the map
        for col_name, value in new_params.items():
            if col_name in header_map:
                row_to_update[header_map[col_name]] = value
            else:
                # This should not happen due to step 2, but as a safeguard
                print(f"Warning: Column {col_name} was not added correctly.")
        
        # Put the updated row back
        linhas_csv[row_index_to_update] = row_to_update
    else:
        print(f"Error: File number {file_number} (row {row_index_to_update}) is out of bounds for CSV (len: {len(linhas_csv)}).")
        return

    # 4. Write data back to CSV
    # =================================================
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(linhas_csv)

def get_results_single(name, gal_name, gal_file, output_csv, components, filter):
    """
    Gets results for single mode. This function RESETS all old parameter
    values in the row before writing new ones.
    """
    result_dir = f'./{name}/results'
    
    base_name = os.path.splitext(gal_name)[0] # ex: "15r" ou "singler"
    
    # Tenta extrair o número do início do nome base
    number_match = re.search(r'^(\d+)', base_name)
    
    if not number_match:
        # Caso 1: É uma run 'single' verdadeira (ex: "singler.fits")
        # Vamos atualizar a primeira linha de dados (índice 1)
        row_index_to_update = 1
        print(f"No number in filename '{gal_name}'. Assuming single-entry CSV, updating row 1.")
    else:
        # Caso 2: É uma re-run 'single' de um membro da amostra (ex: "15r.fits")
        file_number = int(number_match.group(1)) # ex: 15
        
        # O download (lineno) começa em 2 para a primeira linha de dados (índice 1).
        # Portanto, arquivo 2 -> linhas_csv[1]. Arquivo 15 -> linhas_csv[14].
        row_index_to_update = file_number - 1 
        print(f"Found number {file_number} in filename. Updating row {row_index_to_update} in CSV.")
    
    # 1. Read the .dat file and parse ALL parameters
    # =================================================
    arquivo_dat = f'{base_name}_best{components}comp.dat'
    caminho_arquivo = os.path.join(result_dir, arquivo_dat)
    if not os.path.exists(caminho_arquivo):
        print(f"File {caminho_arquivo} not found.")
        return

    new_params = {} # Stores all new values {col_name: value}
    component_count = 0

    with open(caminho_arquivo, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('#   Reduced value:'):
                new_params['Reduced_value'] = line.split(': ')[1].strip()
            elif line.startswith('#   AIC:'):
                new_params['AIC'] = line.split(': ')[1].strip()
            elif line.startswith('#   BIC:'):
                new_params['BIC'] = line.split(': ')[1].strip()
            elif line.startswith('FUNCTION'):
                component_count += 1
            
            elif not line.startswith('#') and line.split():
                parts = line.split()
                if len(parts) >= 2:
                    param_name = parts[0]
                    param_value = parts[1]
                    if param_name in ['X0', 'Y0']:
                        continue
                    col_name = f"{param_name}_{component_count}"
                    new_params[col_name] = param_value

    # 2. Read the CSV and update headers
    # =================================================
    with open(output_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        linhas_csv = list(reader)

    header_row = linhas_csv[0]
    existing_headers = set(header_row)
    
    all_new_headers = list(new_params.keys())
    missing_headers = [h for h in all_new_headers if h not in existing_headers]

    if missing_headers:
        print(f"Adding new columns to CSV: {', '.join(missing_headers)}")
        header_row.extend(missing_headers)
        for i in range(1, len(linhas_csv)):
            linhas_csv[i].extend(['0'] * len(missing_headers))

    # 3. Create Header Map
    # =================================================
    header_map = {header_name: index for index, header_name in enumerate(header_row)}

    # 4. Get Row and RESET old values
    # =================================================
    # For 'single' mode, we ALWAYS update the first data row (index 1)
    if len(linhas_csv) < 2:
        print(f"Error: {output_csv} is empty or has no data rows.")
        return
        
    row_to_update = linhas_csv[1]
    
    print("Resetting old parameter values in single.csv...")
    for header_name, index in header_map.items():
        # Reset criteria
        if header_name in ['Reduced_value', 'AIC', 'BIC']:
            row_to_update[index] = '0'
        # Reset any dynamic parameter (e.g., "r_e_1", "I_0_2")
        elif re.search(r'_\d+$', header_name): 
            row_to_update[index] = '0'

    # 5. Write NEW values
    # =================================================
    print(f"Writing new parameters: {list(new_params.keys())}")
    for col_name, value in new_params.items():
        row_to_update[header_map[col_name]] = value
    
    # Put the updated row back
    linhas_csv[row_index_to_update] = row_to_update

    # 6. Write data back to CSV
    # =================================================
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(linhas_csv)
