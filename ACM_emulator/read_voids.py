import os
from astropy.table import Table
import fitsio
import numpy as np

def get_cosmo_hod_voids (cosmo, hod, pipeline, phase='ph000'):
    if pipeline == 'ACM' or pipeline == 'ACM_AP':
        voidfinder_catalog_path = f'/pscratch/sd/h/hrincon/desigroup/VoidFinder/{pipeline}/Cubic/LRG/{cosmo}/'
        if phase != 'ph000':
            voidfinder_catalog_path+=f'{phase}/'
        voidfinder_catalog_path+=f'z0.500/Abacus_hod{str(hod).zfill(3)}_VoidFinder_Output.fits'
    elif pipeline == 'ACM_phase' or pipeline == 'ACM_AP_phase':
        voidfinder_catalog_path = f'/pscratch/sd/h/hrincon/desigroup/VoidFinder/{pipeline}/Cubic/LRG/c000/covariance/z0.500/Abacus_{phase}_VoidFinder_Output.fits'
    else:
        raise ValueError(f'Invalid pipeline {pipeline}')
    if not os.path.exists(voidfinder_catalog_path):
        return False
    voidfinder_voids = Table.read(voidfinder_catalog_path, hdu = 1)
    return voidfinder_voids


def get_cosmo_hod_galaxies (cosmo, hod, seed, redshift, pipeline, phase='ph000'):
    
    zz = redshift[:4] #remove trailing 0s
    
    if pipeline == "ACM":
        
        coords_xyz = fitsio.read(f'/pscratch/sd/e/epaillas/emc/hods/cosmo+hod/{zz}/yuan23_prior/{cosmo}_{phase}/{seed}/{"hod"+str(hod).zfill(3)}.fits', columns=['X', 'Y', 'Z'])
        coords_xyz = np.array([coords_xyz['X'], coords_xyz['Y'], coords_xyz['Z']])
   
    elif pipeline == 'ACM_phase' or pipeline == 'ACM_AP_phase':
        coords_xyz = fitsio.read(f'/pscratch/sd/e/epaillas/emc/hods/{zz}/yuan23_prior/small/hod466/{phase}_hod466.fits', columns=['X', 'Y', 'Z', 'VX'])
        if pipeline == 'ACM_AP_phase':
            
            box_size = 500.
            corner = -250.

            from cosmoprimo.fiducial import AbacusSummit
            fid_cosmo = AbacusSummit(0)
            hubble = 100 * fid_cosmo.efunc(float(redshift[1:]))
            scale_factor = 1 / (1 + float(redshift[1:]))
             
            x_coord = coords_xyz['X'] + coords_xyz['VX'] / (hubble * scale_factor) # RSD in x direction
            x_coord = ((x_coord - corner) % box_size ) + corner # periodic wrapping
        
            coords_xyz = np.array([x_coord, coords_xyz['Y'], coords_xyz['Z']])
        else:
            coords_xyz = np.array([coords_xyz['X'], coords_xyz['Y'], coords_xyz['Z']])
    elif pipeline == "ACM_AP":
        coords_xyz = fitsio.read(f'/pscratch/sd/n/ntbfin/emulator/hods/{zz}/yuan23_prior/{cosmo}_{phase}/{seed}/{"hod"+str(hod).zfill(3)}.fits', columns=['X_RSD', 'Y_PERP', 'Z_PERP'])
        coords_xyz = np.array([coords_xyz['X_RSD'], coords_xyz['Y_PERP'], coords_xyz['Z_PERP']])
    else:
        raise ValueError (f'"{pipeline}" is not a valid pipeline')
    return coords_xyz