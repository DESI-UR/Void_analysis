'''
This takes in a list of galaxies and the voids found using V2 and appends to the 
list of galaxies columns that denote which are void, wall, edge, and outside the 
survey.

This is modeled after the Environment Classification for V2 and VoidFinder.ipynb 
notebook.
'''

################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as const

from vast.voidfinder.distance import z_to_comoving_dist
from vast.voidfinder.voidfinder import ra_dec_to_xyz

import numpy as np

import pickle

from sklearn import neighbors
################################################################################



################################################################################
# Define miscellaneous functions
#-------------------------------------------------------------------------------
def flatten(array): # when np.flatten() doesn't work
	temp = []
	for sublist in array:
		for item in sublist:
			temp.append(item)
	return np.array(temp)


def rmv(array, out): # removing galaxies in "out" from galaxies in "array"
	test_out = np.ones(len(array), dtype=bool)

	for i, obj in enumerate(array):
		if obj in out:
			test_out[i] = False

	return array[test_out]
################################################################################



################################################################################
# Import galaxies
#-------------------------------------------------------------------------------
gal_filename = '../../../../data/nsa_v1_0_1_VAGC.fits'

hdu = fits.open(gal_filename)
data = Table(hdu[1].data)
hdu.close()
################################################################################



################################################################################
# V2 void catalog
#-------------------------------------------------------------------------------
V2_dir = '../../../void_catalogs/SDSS/V2/VIDE/'

V2_file = 'SDSS_dc0.2_mr0_V2_VIDE_Output.fits'

hdu = fits.open(V2_dir + V2_file)
galzones = Table(hdu[4].data)
zonevoids = Table(hdu[3].data)
hdu.close()
################################################################################



################################################################################
# Identify edge galaxies in the volume-limited sample
#-------------------------------------------------------------------------------
edge_gal = galzones['GAL'][galzones['EDGE'].astype(bool)]
################################################################################



################################################################################
# Identify galaxies outsde the volume-limited mask
#-------------------------------------------------------------------------------
out_gal = galzones['GAL'][galzones['OUT'].astype(bool)]
################################################################################



################################################################################
# Identify void galaxies in the volume-limited sample
#-------------------------------------------------------------------------------
void_zones = zonevoids['ZONE'][(zonevoids['VOID0'] != -1) & (zonevoids['VOID1'] != -1)]

void_gal_ = []
for i in void_zones:
	void_gal_.append(list(galzones['GAL'][galzones['ZONE'] == i]))
void_gal_flat = flatten(void_gal_)

void_gal = rmv(void_gal_flat, edge_gal) # eliminate edge galaxies in void zones
void_gal = rmv(void_gal, out_gal) # eliminate galaxies outside the mask
################################################################################



################################################################################
# Identify wall galaxies in the volume-limited sample
#-------------------------------------------------------------------------------
non_void_zones = zonevoids['ZONE'][zonevoids['VOID0'] == -1] # zones that are not in voids

non_void_gal = []
for i in non_void_zones:
	non_void_gal.append(list(galzones['GAL'][galzones['ZONE'] == i]))

wall_gal = rmv(flatten(non_void_gal), edge_gal)
wall_gal = rmv(wall_gal, out_gal)
################################################################################



################################################################################
# Add vflag column to main galaxy table
#-------------------------------------------------------------------------------








