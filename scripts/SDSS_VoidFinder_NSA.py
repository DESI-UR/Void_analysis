'''VoidFinder - Hoyle & Vogeley (2002)'''



################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------
from vast.voidfinder import find_voids, filter_galaxies
from vast.voidfinder.multizmask import generate_mask
from vast.voidfinder.preprocessing import file_preprocess

import pickle

#import numpy as np
################################################################################





################################################################################
# USER INPUTS
#-------------------------------------------------------------------------------
# Number of CPUs available for analysis.
# A value of None will use one less than all available CPUs.
num_cpus = 1

#-------------------------------------------------------------------------------
survey_name = 'NSA_main_'

# File header
in_directory = '/Users/kellydouglass/Documents/Research/data/SDSS/dr7/'
out_directory = '/Users/kellydouglass/Documents/Research/Voids/void_catalogs/SDSS/VoidFinder/python_implementation/'


# Input file name
galaxies_filename = 'nsa_v1_0_1_main.txt'  # File format: RA, dec, redshift, comoving distance, absolute magnitude
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Survey parameters
# Note: These can be set to None, in which case VoidFinder will use the limits 
# of the galaxy catalog.
min_z = 0
max_z = 0.114

# Cosmology (uncomment and change values to change cosmology)
# Need to also uncomment relevent inputs in function calls below
Omega_M = 0.315 # Planck 2018
#Omega_M = 0.258 # WMAP-5
#h = 1

# Uncomment if you do NOT want to use comoving distances
# Need to also uncomment relevent inputs in function calls below
dist_metric = 'comoving'
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Uncomment if you do NOT want to remove galaxies with Mr > -20
# Need to also uncomment relevent input in function calls below
#mag_cut = False
magnitude_limit = -20


# Uncomment if you do NOT want to remove isolated galaxies
# Need to also uncomment relevent input in function calls below
#rm_isolated = False
#-------------------------------------------------------------------------------
################################################################################





################################################################################
# PREPROCESS DATA
#-------------------------------------------------------------------------------
galaxy_data_table, dist_limits, out1_filename, out2_filename = file_preprocess(galaxies_filename, 
                                                                               in_directory, 
                                                                               out_directory, 
                                                                               #mag_cut=mag_cut,
                                                                               #rm_isolated=rm_isolated,
                                                                               dist_metric=dist_metric,
                                                                               min_z=min_z, 
                                                                               max_z=max_z,
                                                                               Omega_M=Omega_M,
                                                                               #h=h,
                                                                               verbose=1)

print("Dist limits: ", dist_limits)
################################################################################





################################################################################
# GENERATE MASK
#-------------------------------------------------------------------------------
mask, mask_resolution = generate_mask(galaxy_data_table, 
                                      max_z, 
                                      dist_metric=dist_metric,
                                      #h=h,
                                      smooth_mask=True)


temp_outfile = open(out_directory + survey_name + 'mask.pickle', 'wb')
pickle.dump((mask, mask_resolution, dist_limits), temp_outfile)
temp_outfile.close()
################################################################################





################################################################################
# FILTER GALAXIES
#-------------------------------------------------------------------------------
'''
temp_infile = open(out_directory + survey_name + 'mask.pickle', 'rb')
mask, mask_resolution, dist_limits = pickle.load(temp_infile)
temp_infile.close()
'''
wall_coords_xyz, field_coords_xyz = filter_galaxies(galaxy_data_table,
                                                    survey_name,
                                                    out_directory, 
                                                    dist_limits=dist_limits,
                                                    #mag_cut_flag=mag_cut,
                                                    #rm_isolated_flag=rm_isolated,
                                                    #hole_grid_edge_length=5.0,
                                                    dist_metric=dist_metric,
                                                    #h=h,
                                                    magnitude_limit=magnitude_limit,
                                                    verbose=1)

#del galaxy_data_table

temp_outfile = open(out_directory + survey_name + "filter_galaxies_output.pickle", 'wb')
pickle.dump((wall_coords_xyz, field_coords_xyz), temp_outfile)
temp_outfile.close()
################################################################################





################################################################################
# FIND VOIDS
#-------------------------------------------------------------------------------
'''
temp_infile = open(out_directory + survey_name + "filter_galaxies_output.pickle", 'rb')
wall_coords_xyz, field_coords_xyz = pickle.load(temp_infile)
temp_infile.close()
'''
find_voids([wall_coords_xyz, field_coords_xyz],
           survey_name, 
           mask_type='ra_dec_z',
           mask=mask, 
           mask_resolution=mask_resolution,
           dist_limits=dist_limits,
           #save_after=50000,
           #use_start_checkpoint=True,
           #hole_grid_edge_length=5.0,
           #galaxy_map_grid_edge_length=None,
           #hole_center_iter_dist=1.0,
           maximal_spheres_filename=out1_filename,
           void_table_filename=out2_filename,
           potential_voids_filename=out_directory + survey_name + 'potential_voids_list.txt',
           num_cpus=num_cpus,
           batch_size=10000,
           verbose=1,
           print_after=5.0)
################################################################################












