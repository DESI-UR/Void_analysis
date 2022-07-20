'''Identify galaxies as being in a void or not.'''


################################################################################
# IMPORT LIBRARIES
#-------------------------------------------------------------------------------
import numpy as np

from astropy.table import QTable, Table
from astropy.io import fits
import astropy.units as u

import pickle

#import sys
#sys.path.insert(1, '/Users/kellydouglass/Documents/Research/VoidFinder/VoidFinder/python')
from vast.voidfinder.vflag import determine_vflag
from vast.voidfinder.distance import z_to_comoving_dist
################################################################################




################################################################################
# USER INPUT
#-------------------------------------------------------------------------------
# FILE OF VOID HOLES
#-------------------------------------------------------------------------------
#void_catalog_directory = '/Users/kellydouglass/Documents/Research/voids/void_catalogs/SDSS/python_implementation/'
#void_filename = void_catalog_directory + 'kias1033_5_MPAJHU_ZdustOS_main_comoving_holes.txt'

void_catalog_directory = '/Users/kellydouglass/Documents/Research/voids/void_catalogs/public/v1.1.0/'
void_filename = void_catalog_directory + 'VoidFinder-nsa_v1_0_1_main_comoving_holes.txt'


dist_metric = 'comoving'
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# SURVEY MASK FILE
#-------------------------------------------------------------------------------
#mask_filename = void_catalog_directory + 'kias_main_mask.pickle'
mask_filename = void_catalog_directory + 'NSA_main_mask.pickle'
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# FILE OF OBJECTS TO BE CLASSIFIED
#-------------------------------------------------------------------------------
#data_directory = '/Users/kellydouglass/Documents/Research/data/'
data_directory = '/Users/kellydouglass/Documents/Research/data/SDSS/dr17/manga/spectro/redux/v3_1_1/'

#galaxy_file = input('Galaxy data file (with extension): ')
#galaxy_filename = '/Users/kellydouglass/Documents/Drexel/Research/Data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_correctVflag.txt'
#galaxy_filename = '/Users/kellydouglass/Documents/Drexel/Research/Data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_correctVflag_Voronoi_CMD.txt'
#galaxy_filename = data_directory + 'kias1033_5_MPAJHU_ZdustOS_NSAv012_CMDJan2020.txt'
galaxy_filename = data_directory + 'drpall-v3_1_1.fits'

#galaxy_file_format = 'commented_header'
galaxy_file_format = 'fits'
################################################################################





################################################################################
# CONSTANTS
#-------------------------------------------------------------------------------
c = 3e5 # km/s

h = 1
H = 100*h

Omega_M = 0.315 # 0.26 for KIAS-VAGC

# Redshift range of void catalog
z_range = [0, 0.114]

DtoR = np.pi/180
################################################################################





################################################################################
# IMPORT DATA
#-------------------------------------------------------------------------------
print('Importing data')

#-------------------------------------------------------------------------------
# Read in list of void holes
#-------------------------------------------------------------------------------
voids = Table.read(void_filename, format='ascii.commented_header')
'''
voids['x'] == x-coordinate of center of void (in Mpc/h)
voids['y'] == y-coordinate of center of void (in Mpc/h)
voids['z'] == z-coordinate of center of void (in Mpc/h)
voids['R'] == radius of void (in Mpc/h)
voids['voidID'] == index number identifying to which void the sphere belongs
'''
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Read in list of objects to be classified
#-------------------------------------------------------------------------------
if galaxy_file_format == 'ecsv':
    galaxies = QTable.read(galaxy_filename, format='ascii.ecsv')
    DtoR = 1.

    z_col = 'redshift'
    ra_col = 'ra'
    dec_col = 'dec'

elif galaxy_file_format == 'commented_header':
    galaxies = Table.read( galaxy_filename, format='ascii.' + galaxy_file_format)

    z_col = 'redshift'
    ra_col = 'ra'
    dec_col = 'dec'

elif galaxy_file_format == 'fits':
    hdul = fits.open(galaxy_filename)
    galaxies = Table(hdul[1].data)
    hdul.close()

    z_col = 'z'
    ra_col = 'objra'
    dec_col = 'objdec'

else:
    print('Galaxy file format not known.')
    exit()
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Read in survey mask
#-------------------------------------------------------------------------------
mask_infile = open(mask_filename, 'rb')
mask, mask_resolution = pickle.load(mask_infile)
mask_infile.close()
#-------------------------------------------------------------------------------

print('Data and mask imported')
################################################################################





################################################################################
# CONVERT GALAXY ra,dec,z TO x,y,z
#
# Conversions are from http://www.physics.drexel.edu/~pan/VoidCatalog/README
#-------------------------------------------------------------------------------
print('Converting coordinate system')

# Convert redshift to distance
if dist_metric == 'comoving':
    if 'Rgal' not in galaxies.columns:
        galaxies['Rgal'] = z_to_comoving_dist(galaxies[z_col].data.astype(np.float32), Omega_M, h)
    galaxies_r = galaxies['Rgal']

    r_range = z_to_comoving_dist(np.array(z_range, dtype=np.float32), Omega_M, h)
else:
    galaxies_r = c*galaxies[z_col]/H

    r_range = c*z_range/H


# Calculate x-coordinates
galaxies_x = galaxies_r*np.cos(galaxies[dec_col]*DtoR)*np.cos(galaxies[ra_col]*DtoR)

# Calculate y-coordinates
galaxies_y = galaxies_r*np.cos(galaxies[dec_col]*DtoR)*np.sin(galaxies[ra_col]*DtoR)

# Calculate z-coordinates
galaxies_z = galaxies_r*np.sin(galaxies[dec_col]*DtoR)

print('Coordinates converted')
################################################################################




################################################################################
# IDENTIFY LARGE-SCALE ENVIRONMENT
#-------------------------------------------------------------------------------
print('Identifying environment')

galaxies['vflag'] = 9

for i in range(len(galaxies)):

    #print('Galaxy #', galaxies['NSA_index'][i])
    
    if np.all(np.isfinite([galaxies_x[i], galaxies_y[i], galaxies_z[i]])):
        galaxies['vflag'][i] = determine_vflag(galaxies_x[i], 
                                               galaxies_y[i], 
                                               galaxies_z[i], 
                                               voids, 
                                               mask, 
                                               mask_resolution, 
                                               r_range[0], 
                                               r_range[1])

print('Environments identified')
################################################################################





################################################################################
# SAVE RESULTS
#-------------------------------------------------------------------------------
# Output file name
galaxy_file_name, extension = galaxy_filename.split('.')

if galaxy_file_format == 'fits':
    ext = '.fits'
else:
    ext = '.txt'
outfile = galaxy_file_name + '_vflag_' + dist_metric + ext


if galaxy_file_format == 'ecsv':
    galaxies.write( outfile, format='ascii.ecsv', overwrite=True)

elif galaxy_file_format == 'commented_header':
    galaxies.write(outfile, 
                   format='ascii.' + galaxy_file_format, 
                   overwrite=True)

elif galaxy_file_format == 'fits':
    galaxies.write(outfile, format='fits', overwrite=True)
################################################################################



