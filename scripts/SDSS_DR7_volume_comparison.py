################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table

#import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d

import numpy as np

import time

from sklearn import neighbors

from vast.voidfinder import ra_dec_to_xyz
#from vast.voidfinder.voidfinder_functions import not_in_mask
from vast.voidfinder._voidfinder_cython_find_next import not_in_mask

import pickle

import sys

from volume_comparison_functions import calc_volume_boundaries, generate_grid_points, point_query
################################################################################



################################################################################
# Void catalogs to compare
#-------------------------------------------------------------------------------

VF_directory = "/scratch/sbenzvi_lab/desi/dylanbranch/VAST/VoidFinder/scripts/VF3/"
V2_directory = "/scratch/sbenzvi_lab/desi/dylanbranch/VAST/Vsquared/data/"
data_directory = "/scratch/sbenzvi_lab/desi/dylanbranch/data/"
'''
VF_directory = '../../void_catalogs/SDSS/VoidFinder/python_implementation/'
V2_directory = '../../void_catalogs/SDSS/V2/'
data_directory = ''
'''
'''
hh = int(sys.argv[1])
ii = int(hh/16)
jj = int(hh/4)%4
kk = hh%4
'''

#-------------------------------------------------------------------------------
# Mask
#
# This is an output of VoidFinder, and should be saved in the same directory as 
# the catalog files
#-------------------------------------------------------------------------------
mask_file_name = VF_directory + "SDSS_dr7_"+str(ii)+str(jj)+str(kk)+"_mask.pickle"
#mask_file_name = VF_directory + 'NSA_main_mask.pickle'

temp_infile = open(mask_file_name, "rb")
mask, mask_resolution = pickle.load(temp_infile)
temp_infile.close()
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# VoidFinder catalog
#-------------------------------------------------------------------------------
VF_file_name = VF_directory + "DR7m_"+str(ii)+str(jj)+str(kk)+"_comoving_holes.txt"
#VF_file_name = VF_directory + 'nsa_v1_0_1_main_comoving_holes.txt'

VF_table = Table.read(VF_file_name, format="ascii.commented_header")
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# V2 catalogs
#
# This script compares two V2 catalogs with the one VoidFinder catalog
#-------------------------------------------------------------------------------

V2_VIDE_gz_file_name = V2_directory + "V4/DR7_"+str(ii)+str(jj)+str(kk)+"_galzones.dat"
V2_VIDE_zv_file_name = V2_directory + "V4/DR7_"+str(ii)+str(jj)+str(kk)+"_zonevoids.dat"
V2_REVOLVER_gz_file_name = V2_directory + "V4_5/DR7_"+str(ii)+str(jj)+str(kk)+"_5_galzones.dat"
V2_REVOLVER_zv_file_name = V2_directory + "V4_5/DR7_"+str(ii)+str(jj)+str(kk)+"_5_zonevoids.dat"
#vollim_galaxies_file_name = "/scratch/sbenzvi_lab/desi/dylanbranch/NSA_gv2.npy"
vollim_galaxies_file_name = data_directory + "DR7_mocks2/DR7m_"+str(ii)+str(jj)+str(kk)+".dat"
'''
V2_REVOLVER_gz_file_name = V2_directory + 'REVOLVER/nsa_v1_0_1-4_galzones.dat'
V2_REVOLVER_zv_file_name = V2_directory + 'REVOLVER/nsa_v1_0_1-4_zonevoids.dat'
V2_VIDE_gz_file_name = V2_directory + 'VIDE/nsa_v1_0_1-0_galzones.dat'
V2_VIDE_zv_file_name = V2_directory + 'VIDE/nsa_v1_0_1-0_zonevoids.dat'
vollim_galaxies_file_name = data_directory + ''
'''
V2_REVOLVER_gzdata = Table.read(V2_REVOLVER_gz_file_name, 
                                format='ascii.commented_header')

V2_REVOLVER_zvdata = Table.read(V2_REVOLVER_zv_file_name, 
                                format='ascii.commented_header')

V2_VIDE_gzdata = Table.read(V2_VIDE_gz_file_name, 
                            format='ascii.commented_header')

V2_VIDE_zvdata = Table.read(V2_VIDE_zv_file_name, 
                            format='ascii.commented_header')

REVOLVER = np.zeros(len(V2_REVOLVER_gzdata), dtype=bool)
VIDE = np.zeros(len(V2_VIDE_gzdata), dtype=bool)

for i,z in enumerate(V2_REVOLVER_gzdata['zone']):
    if z > -1:
        if V2_REVOLVER_zvdata['void1'][z] > -1:
            REVOLVER[i] = True

for i,z in enumerate(V2_VIDE_gzdata['zone']):
    if z > -1:
        if V2_VIDE_zvdata['void1'][z] > -1:
            VIDE[i] = True


#V2_table = np.load(vollim_galaxies_file_name)
V2_table = Table.read(vollim_galaxies_file_name, 
                      format="ascii.commented_header")

V2_table['VIDE'] = VIDE
V2_table['REVOLVER'] = REVOLVER
galaxies_xyz = ra_dec_to_xyz(V2_table)

V2_table['x'] = galaxies_xyz[:,0]
V2_table['y'] = galaxies_xyz[:,1]
V2_table['z'] = galaxies_xyz[:,2]
#-------------------------------------------------------------------------------
################################################################################





################################################################################
# Generate grid of points
#-------------------------------------------------------------------------------
print('Generating grid points')
start_time = time.time()

xmin, xmax, ymin, ymax, zmin, zmax = calc_volume_boundaries(VF_table, V2_table)

pts = generate_grid_points(xmin, xmax, ymin, ymax, zmin, zmax)

print('Generated grid points', time.time() - start_time)
################################################################################




################################################################################
# Filter out all the points outside the survey
#-------------------------------------------------------------------------------
print('Removing points outside survey')
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

print('Removed points outside survey', time.time() - start_time)
################################################################################




################################################################################
# Remove all points that are within 10 Mpc/h of the survey bounds
#-------------------------------------------------------------------------------
print('Removing edge points')
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
################################################################################




################################################################################
# Determine which points sit inside a void for each of the catalogs
#-------------------------------------------------------------------------------
#start_time = time.time()

true_inside_VF = point_query(points_in_mask, VF_table, True)
true_inside_VIDE = point_query(points_in_mask, V2_table, False, 'VIDE')
true_inside_REVOLVER = point_query(points_in_mask, V2_table, False, 'REVOLVER')

count_in_VF = np.sum(true_inside_VF)
count_in_VIDE = np.sum(true_inside_VIDE)
count_in_REVOLVER = np.sum(true_inside_REVOLVER)
'''
# The "~" inverts the array. So we have true_inside inverted to add up the 
# falses instead of the trues
count_out_VF = np.sum(~true_inside_VF)
count_out_VIDE = np.sum(~true_inside_VIDE)
count_out_REVOLVER = np.sum(~true_inside_REVOLVER)

total_VF = count_in_VF + count_out_VF
total_VIDE = count_in_VIDE + count_out_VIDE
total_REVOLVER = count_in_REVOLVER + count_out_REVOLVER
'''
vfv2_VIDE = np.sum(true_inside_VF*true_inside_VIDE)
vfv2_REVOLVER = np.sum(true_inside_VF*true_inside_REVOLVER)
v2_VIDE_REVOLVER = np.sum(true_inside_VIDE*true_inside_REVOLVER)

'''
print('\nFraction of points inside VoidFinder:', count_in_VF/n_points)
print('\nFraction of points inside V2 (VIDE):', count_in_VIDE/n_points)
print('\nFraction of points inside V2 (REVOLVER):', count_in_REVOLVER/n_points)
print('\nFraction of points in both VoidFinder and V2 (VIDE):', vfv2_VIDE/n_points)
print('\nFraction of points in both VoidFinder and V2 (REVOLVER):', vfv2_REVOLVER/n_points)
print('\nFraction of points in both V2 (VIDE) and V2 (REVOLVER):', v2_VIDE_REVOLVER/n_points)
'''
np.save("/scratch/sbenzvi_lab/desi/dylanbranch/data/DR7_mocks2/vcatprops_"+str(ii)+str(jj)+str(kk)+"_X.npy",
        np.array([count_in_VF, count_in_VIDE, count_in_REVOLVER, vfv2_VIDE, vfv2_REVOLVER, v2_VIDE_REVOLVER, n_points]))
################################################################################
