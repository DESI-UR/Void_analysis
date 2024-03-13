'''
This code was used to correct the published V2 void catalogs, in which Dahlia 
filled the NSAID column with the row index into NSA_v1_0_1_VAGC_vflag-V2-VF.fits
'''

################################################################################
# Load modules
#-------------------------------------------------------------------------------
from astropy.table import Table
################################################################################



################################################################################
# Import data
#-------------------------------------------------------------------------------
# NSA file
#-------------------------------------------------------------------------------
# nsa_directory = '/scratch/kdougla7/data/NSA/'
nsa_directory = '/Users/kdouglass/Documents/Research/data/'

nsa_filename = 'NSA_v1_0_1_VAGC_vflag-V2-VF.fits'

nsa = Table.read(nsa_directory + nsa_filename, 
                 format='fits')
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Void catalog
#-------------------------------------------------------------------------------
catalog_directory = '/Users/kdouglass/Documents/Research/Voids/void_catalogs/public/v1.3.0/'

# cosmology = 'Planck2018'
cosmology = 'WMAP5'

# pruning = 'REVOLVER'
pruning = 'VIDE'

v2_filename = cosmology + '/V2_' + pruning + '-nsa_v1_0_1_' + cosmology + '_galzones.dat'

v2_table = Table.read(catalog_directory + v2_filename, 
                      format='ascii.commented_header')
#-------------------------------------------------------------------------------
################################################################################




################################################################################
# Replace values in NSAID column of V2 catalog with the actualy NSA ID values
#-------------------------------------------------------------------------------
v2_table['NSAID'] = nsa['NSAID'][v2_table['NSAID']]
################################################################################




################################################################################
# Save updated V2 file
#-------------------------------------------------------------------------------
updated_catalog_directory = catalog_directory[:-4] + '4.0/'

v2_table.write(updated_catalog_directory + v2_filename, 
               format='ascii.commented_header')
################################################################################


