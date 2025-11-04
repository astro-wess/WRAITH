import os
import pandas as pd
import shutil

def read_params(params, config_file):

    with open(config_file) as f_in:

        lines = (line.rstrip() for line in f_in) # All lines including the blank ones
        lines = (line.split('#', 1)[0] for line in lines) #remove comments
        lines = (line for line in lines if line) # Non-blank lines


        for line in lines:

            (param,val)=line.split()

            # param 1
            flag = False

            if param == "SAMPLE_NAME":

                (name) = val.split()[0]
                params.name=str(name)

                flag = True 

            if param == "HSC_ID":

                (user_id) = val.split()[0]
                params.user_id=str(user_id)

                flag = True 

            if param == "HSC_PASS":

                (user_pass) = val.split()[0]
                params.user_pass=str(user_pass)

                flag = True 

            if param == "IMAGE_SIZE":

                (image_size) = val.split()[0]
                params.image_size=int(image_size)

                flag = True 

            if param == "FILTER":

                (filter) = val.split()[0]
                params.filter=str(filter)

                flag = True 

            if param == "USE_DEFAULT_SERSIC":

                (use_def) = val.split()[0]
                params.use_def=int(use_def)

                flag = True 

            if param == "IMFIT_CONFIGFILE":

                (imfit_configfile) = val.split()[0]
                params.imfit_configfile=str(imfit_configfile)

                flag = True 

            if param == "CONSECUTIVE_RUNS":

                (consecutive) = val.split()[0]
                params.consecutive=int(consecutive)

                flag = True 

            if param == "COMPONENTS":

                (components) = val.split()[0]
                params.components=int(components)

                flag = True 

            if param == "IMFIT_TYPE":

                (imfit_type) = val.split()[0]
                params.imfit_type=str(imfit_type)

                flag = True 

def generate_dir(csv_file, name, filter):

    # Criação da pasta para os resultados
    os.mkdir(name)

    #Gerar arquivos para download
    csv_data = pd.read_csv(csv_file)
    up_filter = filter.upper()

    # Galáxias
    gal_file = f'{name}gal.txt'
    with open(gal_file, 'w') as file:
        # Escreve o cabeçalho
        file.write("#? filter ra dec sw sh rerun image mask variance type\n")
        
        # Itera sobre cada linha do DataFrame e escreve no formato especificado
        for index, row in csv_data.iterrows():
            nome = index + 1
            ra = row['ra']
            dec = row['dec']
            r90 = row['R90_r']
            size = 4 * r90
            if size < 12:
                size = 12
            file.write(f"HSC-{up_filter} {ra} {dec} {size}asec {size}asec pdr3_wide true true true coadd\n")

    # PSFs
    psf_file = f'{name}psf.txt'
    with open(psf_file, 'w') as file:
        # Escreve o cabeçalho
        file.write("#? rerun ra dec filter type\n")
        
        # Itera sobre cada linha do DataFrame e escreve no formato especificado
        for index, row in csv_data.iterrows():
            nome = index + 1
            ra = row['ra']
            dec = row['dec']
            file.write(f"pdr3_wide {ra} {dec} HSC-{up_filter} coadd\n")

    os.rename(gal_file, f'{name}/{gal_file}'), os.rename(psf_file, f'{name}/{psf_file}')

def download_files(name, filter, userid, userpass):

    actual_folder = os.getcwd()

    # Baixar galáxias
    gal_folder = f'./{name}/gal'
    command_gal = f'python3 ./wraith/downloadCutout.py --list ./{name}/{name}gal.txt --user {userid} --password {userpass} --name "{{lineno}}{filter}"'
    os.system(command_gal)

    os.makedirs(gal_folder, exist_ok=True)
    files = os.listdir(actual_folder)
    fits_files = [file for file in files if file.endswith(f'{filter}.fits')]
    
    for file in fits_files:
        origin = os.path.join(actual_folder, file)
        destiny = os.path.join(gal_folder, file)
        shutil.move(origin, destiny)

    # Baixar PSFs
    psf_folder = f'./{name}/psf'
    command_psf = f'python3 ./wraith/downloadPsf.py --list ./{name}/{name}psf.txt --user {userid} --password {userpass} --name "{{lineno}}{filter}psf"'
    os.system(command_psf)

    os.makedirs(psf_folder, exist_ok=True)
    files = os.listdir(actual_folder)
    fits_files = [file for file in files if file.endswith(f'{filter}psf.fits')]
    
    for file in fits_files:
        origin = os.path.join(actual_folder, file)
        destiny = os.path.join(psf_folder, file)
        shutil.move(origin, destiny)


