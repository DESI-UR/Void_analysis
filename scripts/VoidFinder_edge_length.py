'''VoidFinder - Hoyle & Vogeley (2002)'''



################################################################################
#
# If you have control over your python environment, voidfinder can be installed
# as a normal python package via 'python setup.py install', in which case the 
# below import of 'sys' and 'sys.path.insert(0, '/abspath/to/VoidFinder/python'
# is unnecessary.  If you aren't able to install the voidfinder package,
# you can use the sys.path.insert to add it to the list of available packages
# in your python environment.
#
# Alternately, "python setup.py develop" will 'install' some symlinks which
# point back to the current directory and you can run off the same voidfinder
# repository that you're working on as if it was installed
#
################################################################################


import sys

sys.path.insert(1, '/home/ddunham7/Voids/VoidFinder/python/')

################################################################################
#
#   IMPORT MODULES
#
################################################################################

from voidfinder import find_voids, filter_galaxies

from voidfinder.multizmask import generate_mask
from voidfinder.preprocessing import file_preprocess
from voidfinder.table_functions import to_vector, to_array

from astropy.table import Table
import pickle
import numpy as np





################################################################################
#
#   USER INPUTS
#
################################################################################


# Number of CPUs available for analysis.
# A value of None will use one less than all available CPUs.
num_cpus = 1

#-------------------------------------------------------------------------------
survey_name = 'SDSS_dr7_'

# File header

in_directory = '/home/ddunham7/Voids/VoidFinder/python/voidfinder/'
out_directory = '/scratch/ddunham7/MyFiles/EdgeLength/'

# Input file name
galaxies_filename = 'dr7Data.dat'  # File format: RA, dec, redshift, comoving distance, absolute magnitude
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Survey parameters
# Note: These can be set to None, in which case VoidFinder will use the limits 
# of the galaxy catalog.
min_z = 0
max_z = 0.1026

# Cosmology (uncomment and change values to change cosmology)
# Need to also uncomment relevent inputs in function calls below
#Omega_M = 0.3
#h = 1

# Uncomment if you do NOT want to use comoving distances
# Need to also uncomment relevent inputs in function calls below
#dist_metric = 'redshift'
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Uncomment if you do NOT want to remove galaxies with Mr > -20
# Need to also uncomment relevent input in function call below
#mag_cut = False


# Uncomment if you do NOT want to remove isolated galaxies
# Need to also uncomment relevent input in function call below
#rm_isolated = False
#-------------------------------------------------------------------------------




################################################################################
#
#   PREPROCESS DATA
#
################################################################################

galaxy_data_table, dist_limits, out1_filename, out2_filename = file_preprocess(galaxies_filename, 
                                                                               in_directory, 
                                                                               out_directory, 
                                                                               #mag_cut=mag_cut,
                                                                               #rm_isolated=rm_isolated,
                                                                               #dist_metric=dist_metric,
                                                                               min_z=min_z, 
                                                                               max_z=max_z,
                                                                               #Omega_M=Omega_M,
                                                                               #h=h,
                                                                               verbose=1)

print("Dist limits: ", dist_limits)

################################################################################
#
#   GENERATE MASK
#
################################################################################

mask, mask_resolution = generate_mask(galaxy_data_table, verbose=1, smooth_mask=True)


temp_outfile = open(out_directory + survey_name + 'mask.pickle', 'wb')
pickle.dump((mask, mask_resolution), temp_outfile)
temp_outfile.close()

################################################################################
#
#   FILTER GALAXIES
#
################################################################################

temp_infile = open(out_directory + survey_name + 'mask.pickle', 'rb')
mask, mask_resolution = pickle.load(temp_infile)
temp_infile.close()

out1_filename = out1_filename[:-4] + '_edge_length_' + '.txt'
out2_filename = out2_filename[:-4] + '_edge_length_' + '.txt'

for edge_length in np.arange(1.0,6.0,1):
	
	Out1_Filename = out1_filename[:-4] + str(edge_length)[:-2] + '.txt'
	Out2_Filename = out2_filename[:-4] + str(edge_length)[:-2] + '.txt'
	
	wall_coords_xyz, field_coords_xyz, hole_grid_shape, coords_min = filter_galaxies(galaxy_data_table,
	                                                                                   survey_name,
	                                                                                   #distance_metric=dist_metric,
	                                                                                   #h=h,
	                                                                                   verbose=1,
	                                                                                   hole_grid_edge_length=edge_length)
	
	del galaxy_data_table
	
	
	temp_outfile = open(survey_name + "filter_galaxies_output.pickle", 'wb')
	pickle.dump((wall_coords_xyz, field_coords_xyz, hole_grid_shape, coords_min), temp_outfile)
	temp_outfile.close()
	
	
	
	################################################################################
	#
	#   FIND VOIDS
	#
	################################################################################
	
	
	temp_infile = open(survey_name + "filter_galaxies_output.pickle", 'rb')
	wall_coords_xyz, field_coords_xyz, hole_grid_shape, coords_min = pickle.load(temp_infile)
	temp_infile.close()
	
	
	
	find_voids(wall_coords_xyz, 
	           dist_limits,
	           mask, 
	           mask_resolution,
	           coords_min,
	           hole_grid_shape,
	           survey_name,
	           #save_after=50000,
	           #use_start_checkpoint=True,
	           hole_grid_edge_length=edge_length,
	           galaxy_map_grid_edge_length=None,
	           hole_center_iter_dist=1.0,
	           maximal_spheres_filename=Out1_Filename,
	           void_table_filename=Out2_Filename,
	           potential_voids_filename=survey_name+'potential_voids_list.txt',
	           num_cpus=num_cpus,
	           batch_size=10000,
	           verbose=1,
	           print_after=5.0)
	











