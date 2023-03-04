import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from astropy.table import Table
import numpy as np
import time
from sklearn import neighbors
import pickle
import argparse
import sys
from vast.voidfinder._voidfinder_cython_find_next import MaskChecker
from vast.voidfinder.distance import z_to_comoving_dist
from vast.voidfinder import ra_dec_to_xyz
import pandas as pd

# Create argument parser
parser = argparse.ArgumentParser()

# Add argument for pickle file input
parser.add_argument('--mask_file_name', type=str,
                    required=True, help='Pickle file input for mask')

# Add argument for rmin input
parser.add_argument('--rmin', type=float, required=True,
                    help='Minimum number input')

# Add argument for rmax input
parser.add_argument('--rmax', type=float, required=True,
                    help='Maximum number input')

# Add argument for algorithm comparison option
parser.add_argument('--compare', type=int, required=True,
                    help='Option for algorithm comparison: 0 for VF vs VF, 1 for V2 vs V2, or 2 for VF vs V2')

parser.add_argument('--data_files', type=str, nargs='+',
                    help='Txt files for comoving holes/maximal of galaxy')

# Add argument for whether or not to show plots
parser.add_argument('--show_plots', action='store_true',
                    help='Show plots created with matplotlib')


# Parse arguments
try:
    args = parser.parse_args()
except SystemExit:
    print('Error: Invalid arguments provided')
    sys.exit()

# Get pickle file input of ‘mask_file_name’.
mask_file_name = args.mask_file_name
temp_infile = open(mask_file_name, "rb")
mask, mask_resolution = pickle.load(temp_infile)
temp_infile.close()

# Get rmin and rmax input
rmin = args.rmin
rmax = args.rmax

# Get algorithm comparison option
compare = args.compare
data_files = args.data_files
# file1 = open(data_files[0], "r")
# file2 = open(data_files[1], "r")
# file3 = open(data_files[2], "r")
# file4 = open(data_files[3], "r")

if compare == 0:
    # Get 4 txt files from user: 2 data table files for comoving holes of galaxy, 2 data table files for comoving maximal of galaxy.
    # comoving_holes_galaxy_file_1 = input(
    #     "Enter the first data table file for comoving holes of galaxy: ")
    # data_table_V1 = Table.read(
    #     comoving_holes_galaxy_file_1, format="ascii.commented_header")

    # comoving_holes_galaxy_file_2 = input(
    #     "Enter the second data table file for comoving holes of galaxy: ")
    # data_table_V2 = Table.read(
    #     comoving_holes_galaxy_file_2, format="ascii.commented_header")

    # comoving_maximal_galaxy_file_1 = input(
    #     "Enter the first data table file for comoving maximal of galaxy: ")
    # data_table_V1max = Table.read(
    #     comoving_maximal_galaxy_file_1, format="ascii.commented_header")

    # comoving_maximal_galaxy_file_2 = input(
    #     "Enter the second data table file for comoving maximal of galaxy: ")
    # data_table_V2max = Table.read(
    #     comoving_maximal_galaxy_file_2, format="ascii.commented_header")
    comoving_holes_galaxy_file_1 = open(data_files[0], "r")
    data_table_V1 = Table.read(
        comoving_holes_galaxy_file_1, format="ascii.commented_header")

    comoving_holes_galaxy_file_2 = open(data_files[1], "r")
    data_table_V2 = Table.read(
        comoving_holes_galaxy_file_2, format="ascii.commented_header")

    comoving_maximal_galaxy_file_1 = open(data_files[2], "r")
    data_table_V1max = Table.read(
        comoving_maximal_galaxy_file_1, format="ascii.commented_header")

    comoving_maximal_galaxy_file_2 = open(data_files[3], "r")
    data_table_V2max = Table.read(
        comoving_maximal_galaxy_file_2, format="ascii.commented_header")

elif compare == 1:
    V2_galzones_file = open(data_files[0], "r")
    V2_galzones = Table.read(V2_galzones_file, format='ascii.commented_header')

    V2_zonevoids_file = open(data_files[1], "r")
    V2_zonevoids = Table.read(
        V2_zonevoids_file, format='ascii.commented_header')

    # V2_voids_file = open(data_files[2], "r")
    V2_gz = np.zeros(len(V2_galzones['zone']), dtype=int)

    for i in range(len(V2_gz)):

        if V2_zonevoids['void1'][V2_galzones['zone'][i]] > -1:
            V2_gz[i] = 1

    # file_name = "/Users/lorenzomendoza/Desktop/Research/Function/V2_nsa_v1_0_1_gal.txt"
    V2_voids_file = open(data_files[2], "r")
    data_table_vl = Table.read(V2_voids_file, format="ascii.commented_header")

    omega_M = np.float32(0.3)
    h = np.float32(1.0)
    Rgal = z_to_comoving_dist(
        data_table_vl['redshift'].astype(np.float32), omega_M, h)
    data_table_vl['Rgal'] = Rgal

    # Edge Case: 513626 = [[-0.  0.  0.]]
    z_boolean = data_table_vl['redshift'] > 0
    data_table_vl = data_table_vl[z_boolean]

    galaxies_xyz = ra_dec_to_xyz(data_table_vl)

    data_table_vl['x'] = galaxies_xyz[:, 0]
    data_table_vl['y'] = galaxies_xyz[:, 1]
    data_table_vl['z'] = galaxies_xyz[:, 2]
    # create boolean mask
    boolmask = np.isin(data_table_vl['index'], V2_galzones['gal'])

    # assign values using boolean indexing
    V2_galzones['x'] = data_table_vl['x'][boolmask]
    V2_galzones['y'] = data_table_vl['y'][boolmask]
    V2_galzones['z'] = data_table_vl['z'][boolmask]


def calc_volume_boundaries(void_cat_A, void_cat_B):
    """Compute the boundaries of the minimal rectangular volume (parallelpiped)
    that completely contains two void catalogs.

    Parameters
    ----------
    void_cat_A : astropy.Table
        Table of void data from first catalog.
    void_cat_B : astropy.Table
        Table of void data from second catalog.

    Returns
    -------
    x_min : float
    x_max : float
    y_min : float
    y_max : float
    z_min : float
    z_max : float
    """
    x_min = np.minimum(np.min(void_cat_A['x']), np.min(void_cat_B['x']))
    x_max = np.maximum(np.max(void_cat_A['x']), np.max(void_cat_B['x']))

    y_min = np.minimum(np.min(void_cat_A['y']), np.min(void_cat_B['y']))
    y_max = np.maximum(np.max(void_cat_A['y']), np.max(void_cat_B['y']))

    z_min = np.minimum(np.min(void_cat_A['z']), np.min(void_cat_B['z']))
    z_max = np.maximum(np.max(void_cat_A['z']), np.max(void_cat_B['z']))

    return x_min, x_max, y_min, y_max, z_min, z_max


def generate_grid_points(x_min, x_max, y_min, y_max, z_min, z_max):
    """Creates a dense rectangular grid of points in 3D for the void volume calculation.

    Returns
    -------
    xyz : list
        2D list of points in 3D space.
    """

    x_range = np.arange(x_min, x_max)  # default spacing: 1 Megaparsec
    y_range = np.arange(y_min, y_max)
    z_range = np.arange(z_min, z_max)

    # Creating a meshgrid from the ranges to
    X, Y, Z = np.meshgrid(x_range, y_range, z_range)

    x_points = np.ravel(X)
    y_points = np.ravel(Y)
    z_points = np.ravel(Z)

    point_coords = np.array([x_points, y_points, z_points])

    return point_coords


def kd_tree(void_cat):
    """We are creating a function to make a KDTree to find the number of points in 
    and out of a catalogue.

    Parameters
    ----------
    point_coords: ndarray has a shape of (3,N)
        This is the list of points to query the given void catalogue. N is the number of points given. 
    void_cat: Astropy Table
        This is the given void catalogue.

    Returns
    -------
    true_inside: ndarray of shape (N,1)
        Is this the boolean array of length N (same length as point_coords). True means that 1 point 
        is inside the hole.
    """
#############
    cx = void_cat['x']
    cy = void_cat['y']
    cz = void_cat['z']

    sphere_coords = np.array([cx, cy, cz])

    # The .T is meant to transpose the array from (3,1054) to (1054,3)
    sphere_tree = neighbors.KDTree(sphere_coords.T)
    print("KDTree")

##############

    return sphere_tree


def point_query(point_coords, sphere_tree, void_cat, compare):
    if compare == 0:
        dist, idx = sphere_tree.query(point_coords.T, k=1)

        true_inside = dist < void_cat['radius'][idx]

        true_inside = np.zeros(point_coords.shape[1])
    elif compare == 1:
        idx = sphere_tree.query(point_coords.T, k=1, return_distance=False)

        #true_inside = void_cat[idx]
        for i in range(len(idx)):
            true_inside[i] = void_cat[idx[i]]

    return true_inside


def mask_point_filter(pts, mask, mask_resolution, rmin=0, rmax=312.89816):
    start_time = time.time()
    points_boolean = np.ones(pts.shape[1], dtype=bool)

    mask_checker = MaskChecker(0,
                               mask,
                               mask_resolution,
                               rmin,
                               rmax)

    for i in range(pts.shape[1]):
        curr_pt = pts[:, i]
        not_in_mask = mask_checker.not_in_mask(curr_pt)
        points_boolean[i] = not bool(not_in_mask)

    points_in_mask = pts[:, points_boolean]
    (var, n_points) = points_in_mask.shape

    print('Time taken:', time.time() - start_time)
    print('Points in Mask Shape:', points_in_mask.shape)
    print('Sum of Points IN:', np.sum(points_boolean))
    print('Sum of Points OUT:', np.sum(~points_boolean))
    print('Boolean Shape:', points_boolean.shape)
    print('Points in Mask:', points_in_mask)
    return points_in_mask, points_boolean, var, n_points


def count_points(U, points_in_mask, data_table_V1, data_table_V2, compare):

    if compare == 0:
        start_time = time.time()

        count_in_V1 = np.zeros(U)
        count_out_V1 = np.zeros(U)

        count_in_V2 = np.zeros(U)
        count_out_V2 = np.zeros(U)

        inside_both = np.zeros(U)
        inside_neither = np.zeros(U)
        inside_V1 = np.zeros(U)
        inside_V2 = np.zeros(U)

        points_in_mask_copy = points_in_mask.copy()

        kdTree_V1 = kd_tree(data_table_V1)
        kdTree_V2 = kd_tree(data_table_V2)

        for i in range(U):

            delta = np.random.rand(3)

            points_in_mask_copy[0] = points_in_mask[0] + delta[0]
            points_in_mask_copy[1] = points_in_mask[1] + delta[1]
            points_in_mask_copy[2] = points_in_mask[2] + delta[2]

            true_inside_V1 = point_query(
                points_in_mask_copy, kdTree_V1, data_table_V1)

            count_in_V1[i] = np.sum(true_inside_V1)

            # The "~" inverts the array. So we have true_inside inverted to add up the falses instead of the trues
            count_out_V1[i] = np.sum(~true_inside_V1)

            true_inside_V2 = point_query(
                points_in_mask_copy, kdTree_V2, data_table_V2)

            count_in_V2[i] = np.sum(true_inside_V2)

            # The "~" inverts the array. So we have true_inside inverted to add up the falses instead of the trues
            count_out_V2[i] = np.sum(~true_inside_V2)

            # This is the number of points that are inside both A and B
            inside_V1_and_V2 = np.logical_and(true_inside_V1, true_inside_V2)
            inside_both[i] = np.sum(inside_V1_and_V2)

            # This is the number of points that are in neither A and B
            not_inside_V1_and_V2 = np.logical_and(
                ~true_inside_V1, ~true_inside_V2)
            inside_neither[i] = np.sum(not_inside_V1_and_V2)

            # This is the number of points that are in A but not B
            inside_v1 = np.logical_and(true_inside_V1, ~true_inside_V2)
            inside_V1[i] = np.sum(inside_v1)

            # This is the number of points that are not in A but are in B
            inside_v2 = np.logical_and(~true_inside_V1, true_inside_V2)
            inside_V2[i] = np.sum(inside_v2)

    elif compare == 1:
        start_time = time.time()
        (var, n_points) = points_in_mask.shape

        # Takes about 1.5 mins per query
        points_in_mask_copy = points_in_mask.copy()

        kdTree_V1 = kd_tree(data_table_V1)
        kdTree_V2 = kd_tree(data_table_V2)

        true_inside_V1 = point_query(points_in_mask_copy, kdTree_V1, V2_gz)
        count_in_V1 = np.sum(true_inside_V1)
        count_out_V1 = n_points - count_in_V1

        true_inside_V2 = point_query(points_in_mask_copy, kdTree_V2, V2_gz)
        count_in_V2 = np.sum(true_inside_V2)
        count_out_V2 = n_points - count_in_V2

        inside_both = np.sum(np.logical_and(true_inside_V1, true_inside_V2))
        inside_neither = np.sum(np.logical_not(
            np.logical_or(true_inside_V1, true_inside_V2)))
        inside_V1 = np.sum(np.logical_and(
            true_inside_V1, np.logical_not(true_inside_V2)))
        inside_V2 = np.sum(np.logical_and(
            true_inside_V2, np.logical_not(true_inside_V1)))

    print("Runtime:", time.time() - start_time)
    print('\nNumber of points inside V1:', count_in_V1)
    print('\nNumber of points outside V2:', count_out_V1)
    print('\nNumber of points inside V1:', count_in_V2)
    print('\nNumber of points outside V2:', count_out_V2)
    print("\nThis is the total number of points: {}".format(n_points))

    print(time.time() - start_time)
    print('\nNumber of points inside V1:', count_in_V1)
    print('\nNumber of points outside V2:', count_out_V1)
    print('\nNumber of points inside V1:', count_in_V2)
    print('\nNumber of points outside V2:', count_out_V2)
    print("\nThis is the total number of points: {}".format(n_points))
    return count_in_V1, count_out_V1, count_in_V2, count_out_V2, inside_both, inside_neither, inside_V1, inside_V2, n_points


def count_points(points_in_mask, galzones_V1, galzones_V2, V2_gz):
    start_time = time.time()
    (var, n_points) = points_in_mask.shape

    # Takes about 1.5 mins per query
    points_in_mask_copy = points_in_mask.copy()

    kdTree_V1 = kd_tree(galzones_V1)
    kdTree_V2 = kd_tree(galzones_V2)

    true_inside_V1 = point_query(points_in_mask_copy, kdTree_V1, V2_gz)
    count_in_V1 = np.sum(true_inside_V1)
    count_out_V1 = n_points - count_in_V1

    true_inside_V2 = point_query(points_in_mask_copy, kdTree_V2, V2_gz)
    count_in_V2 = np.sum(true_inside_V2)
    count_out_V2 = n_points - count_in_V2

    inside_both = np.sum(np.logical_and(true_inside_V1, true_inside_V2))
    inside_neither = np.sum(np.logical_not(
        np.logical_or(true_inside_V1, true_inside_V2)))
    inside_V1 = np.sum(np.logical_and(
        true_inside_V1, np.logical_not(true_inside_V2)))
    inside_V2 = np.sum(np.logical_and(
        true_inside_V2, np.logical_not(true_inside_V1)))

    print("Runtime:", time.time() - start_time)
    print('\nNumber of points inside V1:', count_in_V1)
    print('\nNumber of points outside V2:', count_out_V1)
    print('\nNumber of points inside V1:', count_in_V2)
    print('\nNumber of points outside V2:', count_out_V2)
    print("\nThis is the total number of points: {}".format(n_points))
    # print("\nThis is the total number of points in Delaunay: {}".format(total_DEL))
    return (count_in_V1, count_out_V1, count_in_V2, count_out_V2, inside_both, inside_neither, inside_V1, inside_V2, n_points)


def calculate_ratios_and_stats(count_in_V1, count_out_V1, count_in_V2, count_out_V2, inside_both, inside_neither, inside_V1, inside_V2, n_points):
    r_V1 = count_in_V1 / n_points
    r_V2 = count_in_V2 / n_points
    r_V1_V2 = np.sum(inside_both) / n_points
    r_not_V1_V2 = np.sum(inside_neither) / n_points
    r_V1_not_V2 = np.sum(inside_V1) / n_points
    r_V2_not_V1 = np.sum(inside_V2) / n_points

    average_V1 = np.mean(count_in_V1)
    r_average_V1 = average_V1 / n_points
    std_V1 = np.std(count_in_V1)
    r_std_V1 = std_V1 / n_points

    average_V2 = np.mean(count_in_V2)
    r_average_V2 = average_V2 / n_points
    std_V2 = np.std(count_in_V2)
    r_std_V2 = std_V2 / n_points

    average_inside = np.mean(inside_both)
    r_average_inside = average_inside / n_points
    std_both = np.std(inside_both)
    r_std_both = std_both / n_points

    average_outside = np.mean(inside_neither)
    r_average_outside = average_outside / n_points
    std_outside = np.std(inside_neither)
    r_std_outside = std_outside / n_points

    results = pd.DataFrame({
        'Category': ['V1', 'V2', 'Both', 'Neither', 'V1 not V2', 'V2 not V1'],
        'Number of points': [count_in_V1, count_in_V2, np.sum(inside_both), np.sum(inside_neither), np.sum(inside_V1), np.sum(inside_V2)],
        'Number of points outside': [count_out_V1, count_out_V2, np.nan, np.nan, np.nan, np.nan],
        'Ratio of points': [r_V1, r_V2, r_V1_V2, r_not_V1_V2, r_V1_not_V2, r_V2_not_V1],
        'Average number of points': [average_V1, average_V2, average_inside, average_outside, np.nan, np.nan],
        'Standard deviation of points': [std_V1, std_V2, std_both, std_outside, np.nan, np.nan],
        'Ratio of average number of points': [r_average_V1, r_average_V2, r_average_inside, r_average_outside, np.nan, np.nan],
        'Ratio of standard deviation': [r_std_V1, r_std_V2, r_std_both, r_std_outside, np.nan, np.nan]
    })

    return results.set_index('Category')


###########################################################################################
# This line creates the boundaries (like the whole min and max)
xmin, xmax, ymin, ymax, zmin, zmax = calc_volume_boundaries(
    data_table_V1, data_table_V2)

# This line makes creates the points in between
pts = generate_grid_points(xmin, xmax, ymin, ymax, zmin, zmax)

###########################################################################################

###########################################################################################
points_in_mask, points_boolean, var, n_points = mask_point_filter(
    pts, mask, mask_resolution)
###########################################################################################


b = pts.shape
print(b)

(count_in_V1, count_out_V1, count_in_V2, count_out_V2, inside_both, inside_neither, inside_V1,
 inside_V2, n_points) = count_points(1, points_in_mask, data_table_V1, data_table_V2)

results = calculate_ratios_and_stats(count_in_V1, count_out_V1, count_in_V2,
                                     count_out_V2, inside_both, inside_neither, inside_V1, inside_V2, n_points)
print(results)


xmin, xmax, ymin, ymax, zmin, zmax = calc_volume_boundaries(
    V2_galzones, V2_galzones)

# This line makes creates the points in between
pts = generate_grid_points(xmin, xmax, ymin, ymax, zmin, zmax)

b = pts.shape
print(b)

points_in_mask, points_boolean = mask_point_filter(pts, mask, mask_resolution)
(count_in_V1, count_out_V1, count_in_V2, count_out_V2, inside_both, inside_neither, inside_V1,
 inside_V2, n_points) = count_points(points_in_mask, V2_galzones, V2_galzones, V2_gz)

results = calculate_ratios_and_stats(count_in_V1, count_out_V1, count_in_V2,
                                     count_out_V2, inside_both, inside_neither, inside_V1, inside_V2, n_points)
print(results)
