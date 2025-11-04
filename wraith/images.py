import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def plot_fits_images_side_by_side(fits_files, output_file, number):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, fits_file in enumerate(fits_files):
        # Determinar o HDU a ser aberto
        hdu_index = 1 if i in [0, 2] else 0
        
        # Abrir o arquivo FITS
        with fits.open(fits_file) as hdul:
            data = hdul[hdu_index].data
        
        # Normalizar os dados para evitar problemas com valores negativos ou zeros
        data = np.clip(data, a_min=1e-10, a_max=None)
        
        if i == 0:
            # Primeiro arquivo: escala logarítmica
            ax = axes[i]
            data_log = np.log10(data + 1)  # Adiciona um pequeno valor para evitar log(0)
            img = ax.imshow(data_log, cmap='gray', origin='lower')
            ax.set_title('Galáxia')
        
        elif i == 1:
            # Segundo arquivo: escala logarítmica
            ax = axes[i]
            data_log = np.log10(data + 1e-1)  # Adiciona um pequeno valor para evitar log(0)
            img = ax.imshow(data_log, cmap='gray', origin='lower')
            ax.set_title('Modelo')
        
        elif i == 2:
            # Terceiro arquivo: escala logarítmica 99%
            ax = axes[i]
            data_log = np.log10(data + 1e-10)  # Adiciona um pequeno valor para evitar log(0)
            vmin, vmax = np.percentile(data_log, (0, 99))
            img = ax.imshow(data_log, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            ax.set_title('Máscara')
        
        elif i == 3:
            # Quarto arquivo: escala normal
            ax = axes[i]
            img = ax.imshow(data, cmap='gray', origin='lower')
            ax.set_title('Resíduo')
        
        ax.axis('off')
    plt.suptitle(f'Galáxia {number}', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file)