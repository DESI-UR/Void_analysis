################################################################################
# Import modules
#-------------------------------------------------------------------------------
import pickle

import time

from astropy.table import Table

import numpy as np

from volume_comparison_functions import calc_volume_boundaries, generate_grid_points

from vast.voidfinder._voidfinder_cython_find_next import not_in_mask
from vast.voidfinder.viz import VoidRender
################################################################################




################################################################################
# Void catalog
#-------------------------------------------------------------------------------
VF_directory = '../../void_catalogs/SDSS/VoidFinder/python_implementation/'

VF_holes_file_name = VF_directory + "nsa_v1_0_1_main_comoving_holes.txt"

VF_voids = Table.read(VF_holes_file_name, format="ascii.commented_header")
################################################################################




################################################################################
# Survey mask
#-------------------------------------------------------------------------------
mask_file_name = VF_directory + "NSA_main_mask.pickle"

temp_infile = open(mask_file_name, "rb")
mask, mask_resolution = pickle.load(temp_infile)
temp_infile.close()
################################################################################




################################################################################
# Generate grid points
#-------------------------------------------------------------------------------
#This line creates the boundaries (like the whole min and max)
xmin, xmax, ymin, ymax, zmin, zmax = calc_volume_boundaries(VF_voids, VF_voids)

#This line makes creates the points in between 
pts = generate_grid_points(xmin, xmax, ymin, ymax, zmin, zmax)
################################################################################




################################################################################
# Remove all points outside of the survey mask
#-------------------------------------------------------------------------------
start_time = time.time()

rmin = 0
rmax = 332.38565

points_boolean = np.ones(pts.shape[1], dtype=bool)

for i in range(pts.shape[1]):
    points_boolean[i] = not_in_mask(pts[:,i].reshape((1,3)), 
                                    mask, 
                                    mask_resolution, 
                                    rmin, 
                                    rmax)        


points_in_survey = pts[:,~points_boolean]
#(var, n_points) = points_in_survey.shape

print(time.time() - start_time)
print(points_in_survey.shape)
print(np.sum(points_boolean))
print(np.sum(~points_boolean))
print(points_boolean.shape)
################################################################################




################################################################################
# Remove points that are within 10 Mpc/h of the survey boundary
#-------------------------------------------------------------------------------
start_time = time.time()

coords_min = points_in_survey - 10
coords_max = points_in_survey + 10

pts_boolean = np.ones(points_in_survey.shape[1], dtype=bool)

for i in range(points_in_survey.shape[1]):

    x_coords = [coords_min[0,i], 
                coords_max[0,i], 
                points_in_survey[0,i], 
                points_in_survey[0,i], 
                points_in_survey[0,i], 
                points_in_survey[0,i]]
    y_coords = [points_in_survey[1,i], 
                points_in_survey[1,i], 
                coords_min[1,i], 
                coords_max[1,i], 
                points_in_survey[1,i], 
                points_in_survey[1,i]]
    z_coords = [points_in_survey[2,i], 
                points_in_survey[2,i], 
                points_in_survey[2,i], 
                points_in_survey[2,i], 
                coords_min[2,i], 
                coords_max[2,i]]
    extreme_coords = np.array([x_coords, y_coords, z_coords]).T

    j = 0
    while pts_boolean[i] and j <= 5:
        # Check to see if any of these are now outside the mask
        if not_in_mask(extreme_coords[j].reshape(1,3), mask, mask_resolution, rmin, rmax):
            pts_boolean[i] = False
        j += 1

points_in_mask = points_in_survey[:,pts_boolean]

(var, n_points) = points_in_mask.shape

print('Removed edge points', time.time() - start_time)
print(points_in_mask.shape)
print(np.sum(pts_boolean))
print(np.sum(~pts_boolean))
################################################################################




################################################################################
# Visualize point distribution with VoidRender
#-------------------------------------------------------------------------------
viz = VoidRender(galaxy_xyz=points_in_survey[:,~pts_boolean].T)

viz.run()
################################################################################


