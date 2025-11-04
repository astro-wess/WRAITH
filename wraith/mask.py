from astropy.io import fits
import numpy as np
import shutil
import os

def make_maps(gal_file):

    comando_cold = f'source-extractor {gal_file} -c ./wraith/configCold.sex'
    os.system(comando_cold)

    comando_hot = f'source-extractor {gal_file} -c ./wraith/configHot.sex'
    os.system(comando_hot)

def combine_maps():
    cold_map = 'cold.fits'
    hot_map = 'hot.fits'

    with fits.open(cold_map) as hdul1, fits.open(hot_map) as hdul2:
        data1 = hdul1[1].data
        data2 = hdul2[1].data
        
        max_value = data1.max()
        
        data2_offset = np.where(data2 != 0, data2 + max_value + 1, 0)
        
        combined_data = np.where(data1 != 0, data1, data2_offset)
        
        header1 = hdul1[1].header
        primary_header = hdul1[0].header
    
    new_hdu = fits.ImageHDU(data=combined_data, header=header1)
    primary_hdu = fits.PrimaryHDU(header=primary_header)
    
    combined_hdul = fits.HDUList([primary_hdu, new_hdu])
    combined_hdul.writeto('combined_map.fits', overwrite=True)
    os.remove(cold_map), os.remove(hot_map)

def remove_central_object():
    comb_map = 'combined_map.fits'

    with fits.open(comb_map) as hdul:
        data = hdul[1].data
        
        center_y, center_x = data.shape[0] // 2, data.shape[1] // 2
        central_value = data[center_y, center_x]
        
        modified_data = np.where(data == central_value, 0, data)
        
        header = hdul[1].header
        primary_header = hdul[0].header
    
    new_hdu = fits.ImageHDU(data=modified_data, header=header)
    primary_hdu = fits.PrimaryHDU(header=primary_header)
    
    combined_hdul = fits.HDUList([primary_hdu, new_hdu])
    combined_hdul.writeto('modified_map.fits', overwrite=True)
    os.remove(comb_map)

def apply_mask(name, gal_name, gal_file):
    actual_folder = os.getcwd()
    map_file = 'modified_map.fits'
    mask_file = os.path.splitext(gal_name)[0]
    mask_name = f'{mask_file}mask.fits'
    mask_dir = f'./{name}/mask'
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)

    with fits.open(gal_file) as hdugal, fits.open(map_file) as hdumap:
        datasinal = hdugal[1].data
        dataruido = hdugal[3].data
        datamap = hdumap[1].data

        sr = datasinal / dataruido

        sr[np.where(sr < 3)] = 1
        sr[np.where(sr >= 3)] = 0

        mask = sr + datamap

        hdu_mask = fits.ImageHDU(data=mask)
        hdumap[1] = hdu_mask
        hdumap.writeto(mask_name, overwrite=True)
        os.remove(map_file)
        origin = os.path.join(actual_folder, mask_name)
        destiny = os.path.join(mask_dir, mask_name)
        shutil.move(origin, destiny)

