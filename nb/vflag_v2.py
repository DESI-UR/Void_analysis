'''
Function to classify galaxy environment (void v. wall) for V2 void catalogs
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from vast.voidfinder.distance import z_to_comoving_dist
from vast.voidfinder.voidfinder import ra_dec_to_xyz

import numpy as np

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
# Define vflag_V2
#-------------------------------------------------------------------------------
def determine_vflag_V2(gals, galzones, zonevoids, z_max, Omega_M0, h):
	'''
	Determine the galaxy environment (void, wall, edge, out) for galaxies based 
	on a V2 void catalog.


	PARAMETERS
	==========

	gals : astropy table
		List of galaxies for which to define the environment

	galzones : astropy table
		Output of V2 that defines which zone each galaxy belongs to.  This list 
		of galaxies is expected to be a subset of gals.

	zonevoids : astropy table
		Output of V2 that defines which void each zone belongs to.

	z_max : float
		Maximum redshift of void catalog

	Omega_M0 : float
		Value of Omega_M0 used in void-finding

	h : float
		Value of reduced Hubble constant used during void-finding


	RETURNS
	=======

	vflag_V2 : ndarray of shape (n,)
		Galaxy environment classification for each galaxy in gals.
		0 = wall
		1 = void
		2 = edge (too close to the survey boundary to accurately classify)
		9 = out (outside the void catalog boundary)
	'''

	############################################################################
	# Initialize output column
	#---------------------------------------------------------------------------
	N_gal = len(gals)

	vflag_V2 = -9*np.ones(N_gal)
	############################################################################


	############################################################################
	# Identify edge galaxies in the volume-limited sample
	#---------------------------------------------------------------------------
	edge_gal = galzones['TARGET'][galzones['EDGE'].astype(bool)]
	############################################################################


	############################################################################
	# Identify galaxies outsde the volume-limited mask
	#---------------------------------------------------------------------------
	out_gal = galzones['TARGET'][galzones['OUT'].astype(bool)]
	############################################################################


	############################################################################
	# Identify void galaxies in the volume-limited sample
	#---------------------------------------------------------------------------
	void_zones = zonevoids['ZONE'][(zonevoids['VOID0'] != -1) & (zonevoids['VOID1'] != -1)]

	void_gal_ = []
	for i in void_zones:
		void_gal_.append(list(galzones['TARGET'][galzones['ZONE'] == i]))
	void_gal_flat = flatten(void_gal_)

	void_gal = rmv(void_gal_flat, edge_gal) # eliminate edge galaxies in void zones
	void_gal = rmv(void_gal, out_gal) # eliminate galaxies outside the mask
	############################################################################


	############################################################################
	# Identify wall galaxies in the volume-limited sample
	#---------------------------------------------------------------------------
	non_void_zones = zonevoids['ZONE'][zonevoids['VOID0'] == -1] # zones that are not in voids

	non_void_gal = []
	for i in non_void_zones:
		non_void_gal.append(list(galzones['TARGET'][galzones['ZONE'] == i]))

	wall_gal = rmv(flatten(non_void_gal), edge_gal) # eliminate edge galaxies
	wall_gal = rmv(wall_gal, out_gal) # eliminate galaxies outside the mask
	############################################################################


	############################################################################
	# Set vflag value for galaxies in volume-limited sample
	#---------------------------------------------------------------------------
	for i in range(N_gal):

		if gals['NSAID'][i] in wall_gal:
			vflag_V2[i] = 0

		elif gals['NSAID'][i] in void_gal:
			vflag_V2[i] = 1

		elif gals['NSAID'][i] in edge_gal:
			vflag_V2[i] = 2

		elif gals['NSAID'][i] in out_gal:
			vflag_V2[i] = 9
	############################################################################


	############################################################################
	# For objects that are within the same volume as the volume-limited sample 
	# but are not part of the volume-limited sample, they get the classification 
	# of their nearest neighbor that is in the volume-limited sample.
	#
	# Note: We need to match via Cartesian coordinates, since we want to find 
	# which Voronoi cell each galaxy lives in, and these are defined in 
	# Cartesian space (not redshift space).
	#---------------------------------------------------------------------------
	missing_bool = (vflag_V2 == -9) & (gals['Z'] <= z_max)

	gals['Rgal'] = z_to_comoving_dist(gals['Z'].data.astype(np.float32), 
									  Omega_M0, 
									  h)

	missing_gal_xyz = ra_dec_to_xyz(gals[missing_bool])
	vollim_xyz = ra_dec_to_xyz(gals[vflag_V2 != -9])

	vollim_tree = neighbors.KDTree(vollim_xyz)

	_,idx_V2gals = vollim_tree.query(missing_gal_xyz) # Find the nearest galaxy

	vflag_V2[missing_bool] = vflag_V2[vflag_V2 != -9][idx_V2gals[:,0]]
	############################################################################


	return np.abs(vflag_V2) # flip any remaining -9 to 9, because they are outside the survey
################################################################################












