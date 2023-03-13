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

# Add argument for rmin input
parser.add_argument('--omega', type=float,
                    help='Omega input')

# Add argument for rmax input
parser.add_argument('--h', type=float,
                    help='H input')

parser.add_argument('--data_files', type=str, nargs='+',
                    help='Txt files for comoving holes/maximal of galaxy')

# Add argument for whether or not to show plots
parser.add_argument('--show_plots', action='store_true',
                    help='Show plots created with matplotlib')

parser.add_argument('--s', action='store_true',
                    help='Save Results')


# Parse arguments
try:
    args = parser.parse_args()
except SystemExit:
    print('Error: Invalid arguments provided')
    sys.exit()

datafolder = "data/"

# Get pickle file input of ‘mask_file_name’.
mask_file_name = args.mask_file_name
temp_infile = open("data/"+mask_file_name, "rb")
mask, mask_resolution = pickle.load(temp_infile)
temp_infile.close()

# Get rmin and rmax input
rmin = args.rmin
rmax = args.rmax

# Get algorithm comparison option
compare = args.compare
data_files = args.data_files

show_plots = args.show_plots
save = argparse.s


def load_VF(data_files):
    comoving_holes_galaxy_file_1 = "data/"+data_files[0]
    data_table_V1 = Table.read(
        comoving_holes_galaxy_file_1, format="ascii.commented_header")

    comoving_holes_galaxy_file_2 = "data/"+data_files[2]
    data_table_V2 = Table.read(
        comoving_holes_galaxy_file_2, format="ascii.commented_header")

    comoving_maximal_galaxy_file_1 = "data/"+data_files[1]
    data_table_V1max = Table.read(
        comoving_maximal_galaxy_file_1, format="ascii.commented_header")

    comoving_maximal_galaxy_file_2 = "data/"+data_files[3]
    data_table_V2max = Table.read(
        comoving_maximal_galaxy_file_2, format="ascii.commented_header")

    V2_gz = None
    V2_gz2 = None

    return data_table_V1, data_table_V2, data_table_V1max, data_table_V2max, V2_gz, V2_gz2


def loadV2(data_files, omega_=0.3089, h_=1.0):
    V2_galzones_file = "data/"+data_files[0]
    V2_galzones = Table.read(V2_galzones_file, format='ascii.commented_header')
    V2_galzones_file2 = "data/"+data_files[3]
    V2_galzones2 = Table.read(
        V2_galzones_file2, format='ascii.commented_header')

    V2_zonevoids_file = "data/"+data_files[1]
    V2_zonevoids = Table.read(
        V2_zonevoids_file, format='ascii.commented_header')
    V2_zonevoids_file2 = "data/"+data_files[4]
    V2_zonevoids2 = Table.read(
        V2_zonevoids_file2, format='ascii.commented_header')

    V2_gz = np.zeros(len(V2_galzones['zone']), dtype=int)
    V2_gz2 = np.zeros(len(V2_galzones2['zone']), dtype=int)

    for i in range(len(V2_gz)):
        if V2_zonevoids['void1'][V2_galzones['zone'][i]] > -1:
            V2_gz[i] = 1

    for i in range(len(V2_gz2)):
        if V2_zonevoids2['void1'][V2_galzones2['zone'][i]] > -1:
            V2_gz2[i] = 1

    V2_voids_file = "data/"+data_files[2]
    data_table_vl = Table.read(V2_voids_file, format="ascii.commented_header")
    V2_voids_file2 = "data/"+data_files[5]
    data_table_vl2 = Table.read(
        V2_voids_file2, format="ascii.commented_header")

    omega_M = np.float32(omega_)
    h = np.float32(h_)
    Rgal = z_to_comoving_dist(
        data_table_vl['redshift'].astype(np.float32), omega_M, h)
    data_table_vl['Rgal'] = Rgal

    Rgal2 = z_to_comoving_dist(
        data_table_vl2['redshift'].astype(np.float32), omega_M, h)
    data_table_vl2['Rgal'] = Rgal2

    z_boolean = data_table_vl['redshift'] > 0
    data_table_vl = data_table_vl[z_boolean]

    galaxies_xyz = ra_dec_to_xyz(data_table_vl)

    data_table_vl['x'] = galaxies_xyz[:, 0]
    data_table_vl['y'] = galaxies_xyz[:, 1]
    data_table_vl['z'] = galaxies_xyz[:, 2]

    boolmask = np.isin(data_table_vl['index'], V2_galzones['gal'])
    V2_galzones['x'] = data_table_vl['x'][boolmask]
    V2_galzones['y'] = data_table_vl['y'][boolmask]
    V2_galzones['z'] = data_table_vl['z'][boolmask]

    z_boolean2 = data_table_vl2['redshift'] > 0
    data_table_vl2 = data_table_vl2[z_boolean2]

    galaxies_xyz2 = ra_dec_to_xyz(data_table_vl2)

    data_table_vl2['x'] = galaxies_xyz2[:, 0]
    data_table_vl2['y'] = galaxies_xyz2[:, 1]
    data_table_vl2['z'] = galaxies_xyz2[:, 2]

    boolmask2 = np.isin(data_table_vl2['index'], V2_galzones2['gal'])
    V2_galzones2['x'] = data_table_vl2['x'][boolmask2]
    V2_galzones2['y'] = data_table_vl2['y'][boolmask2]
    V2_galzones2['z'] = data_table_vl2['z'][boolmask2]

    return V2_galzones, V2_galzones2, V2_gz, V2_gz2


def load_VfvV2(data_files, omega_=0.3089, h_=1.0):
    comoving_holes_galaxy_file = "data/"+data_files[0]
    data_table_VF = Table.read(
        comoving_holes_galaxy_file, format="ascii.commented_header")

    comoving_maximal_galaxy_file = "data/"+data_files[1]
    data_table_VFmax = Table.read(
        comoving_maximal_galaxy_file, format="ascii.commented_header")

    V_galzones_file = "data/"+data_files[2]
    V_galzones = Table.read(
        V_galzones_file, format='ascii.commented_header')

    V_zonevoids_file = "data/"+data_files[3]
    V_zonevoids = Table.read(
        V_zonevoids_file, format='ascii.commented_header')

    V2_gz = np.zeros(len(V_galzones['zone']), dtype=int)
    for i in range(len(V2_gz)):
        if V_zonevoids['void1'][V_galzones['zone'][i]] > -1:
            V2_gz[i] = 1

    V_voids_file = "data/"+data_files[4]
    data_table_vl = Table.read(V_voids_file, format="ascii.commented_header")

    omega_M = np.float32(omega_)
    h = np.float32(h_)
    Rgal = z_to_comoving_dist(
        data_table_vl['redshift'].astype(np.float32), omega_M, h)
    data_table_vl['Rgal'] = Rgal

    z_boolean = data_table_vl['redshift'] > 0
    data_table_vl = data_table_vl[z_boolean]

    galaxies_xyz = ra_dec_to_xyz(data_table_vl)

    data_table_vl['x'] = galaxies_xyz[:, 0]
    data_table_vl['y'] = galaxies_xyz[:, 1]
    data_table_vl['z'] = galaxies_xyz[:, 2]

    boolmask = np.isin(data_table_vl['index'], V_galzones['gal'])
    V_galzones['x'] = data_table_vl['x'][boolmask]
    V_galzones['y'] = data_table_vl['y'][boolmask]
    V_galzones['z'] = data_table_vl['z'][boolmask]

    V2_gz2 = None

    return data_table_VF, data_table_VFmax, V_galzones, V2_gz, V2_gz2


if compare == 0:
    data_table_V1, data_table_V2, data_table_V1max, data_table_V2max, V2_gz, V2_gz2 = load_VF(
        data_files)


elif compare == 1:

    data_table_V1, data_table_V2, V2_gz, V2_gz2 = loadV2(
        data_files, args.omega, args.h)


elif compare == 2:

    data_table_V1, data_table_V1max, data_table_V2, V2_gz, V2_gz2 = load_VfvV2(
        data_files, args.omega, args.h)


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
    print("\n")
    print("Calculating volume boundaries...")
    x_min = np.minimum(np.min(void_cat_A['x']), np.min(void_cat_B['x']))
    x_max = np.maximum(np.max(void_cat_A['x']), np.max(void_cat_B['x']))

    y_min = np.minimum(np.min(void_cat_A['y']), np.min(void_cat_B['y']))
    y_max = np.maximum(np.max(void_cat_A['y']), np.max(void_cat_B['y']))

    z_min = np.minimum(np.min(void_cat_A['z']), np.min(void_cat_B['z']))
    z_max = np.maximum(np.max(void_cat_A['z']), np.max(void_cat_B['z']))

    print("Space boundaries: x_min = {}, x_max = {}, y_min = {}, y_max = {}, z_min = {}, z_max = {}".format(
        x_min, x_max, y_min, y_max, z_min, z_max), "\n")
    return x_min, x_max, y_min, y_max, z_min, z_max


def generate_grid_points(x_min, x_max, y_min, y_max, z_min, z_max):
    """Creates a dense rectangular grid of points in 3D for the void volume calculation.

    Returns
    -------
    xyz : list
        2D list of points in 3D space.
    """

    print("Generating grid points...")

    x_range = np.arange(x_min, x_max)  # default spacing: 1 Megaparsec
    y_range = np.arange(y_min, y_max)
    z_range = np.arange(z_min, z_max)

    # Creating a meshgrid from the ranges to
    X, Y, Z = np.meshgrid(x_range, y_range, z_range)

    x_points = np.ravel(X)
    y_points = np.ravel(Y)
    z_points = np.ravel(Z)

    point_coords = np.array([x_points, y_points, z_points])

    print("Number of grid points: {}".format(point_coords.shape[1]), "\n")

    return point_coords


def mask_point_filter(pts, mask, mask_resolution, rmin=0, rmax=312.89816):
    """Filter points in a 3D space by a boolean mask.

    Parameters
    ----------
    pts : np.ndarray
        3D points to filter.
    mask : np.ndarray
        Boolean mask.
    mask_resolution : float
        Resolution of the mask.
    rmin : float
        Minimum radius of the mask.
    rmax : float
        Maximum radius of the mask.

    Returns
    -------
    points_in_mask : np.ndarray
        3D points that are in the mask.
    """

    print('Filtering points in mask...')

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
    print('Time taken:', time.time() - start_time)
    print('Points in Mask Shape:', points_in_mask.shape)
    print('Sum of Points IN:', np.sum(points_boolean))
    print('Sum of Points OUT:', np.sum(~points_boolean))
    print('Boolean Shape:', points_boolean.shape)
    print('Points in Mask:', points_in_mask)
    print("\n")
    return points_in_mask, points_boolean


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

##############

    return sphere_tree


def point_query(point_coords, sphere_tree, void_cat, compare_bool):
    """We are creating a function to query the KDTree to find the number of points in
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

    print('Querying points in mask...')

    if compare_bool:
        dist, idx = sphere_tree.query(point_coords.T, k=1)

        true_inside = dist < void_cat['radius'][idx]

    else:
        true_inside = np.zeros(point_coords.shape[1])

        idx = sphere_tree.query(point_coords.T, k=1, return_distance=False)
        print('idx shape:', idx.shape)
        print('idx:', idx)
        #true_inside = void_cat[idx]
        for i in range(len(idx)):
            true_inside[i] = void_cat[idx[i]]

    print('True Inside Shape:', true_inside.shape)

    return true_inside


def count_points(U, points_in_mask, data_table_V1, data_table_V2, compare=0, V2_gzA=None, V2_gzB=None):
    """We are creating a function to count the number of points in and out of a catalogue.

    Parameters
    ----------
    U: int
        This is the number of random points to be generated.
    points_in_mask: ndarray has a shape of (3,N)
        This is the list of points to query the given void catalogue. N is the number of points given.
    data_table_V1: Astropy Table
        This is the 1st void catalogue for comparison.
    data_table_V2: Astropy Table
        This is the 2nd void catalogue for comparison.

    """

    print('Counting points in mask...')

    (var, n_points) = points_in_mask.shape

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
            print('Counting points in V1...', count_in_V1)
            print('Counting points out of V1...', count_out_V1)
            true_inside_V1 = point_query(
                points_in_mask_copy, kdTree_V1, data_table_V1, True)
            print('True Inside V1:', true_inside_V1)

            count_in_V1[i] = np.sum(true_inside_V1)

            # The "~" inverts the array. So we have true_inside inverted to add up the falses instead of the trues
            count_out_V1[i] = np.sum(~true_inside_V1)

            true_inside_V2 = point_query(
                points_in_mask_copy, kdTree_V2, data_table_V2, True)
            print('True Inside V2:', true_inside_V2)

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

        true_inside_V1 = point_query(
            points_in_mask_copy, kdTree_V1, V2_gzA, False)
        count_in_V1 = np.sum(true_inside_V1)
        count_out_V1 = n_points - count_in_V1

        true_inside_V2 = point_query(
            points_in_mask_copy, kdTree_V2, V2_gzB, False)
        count_in_V2 = np.sum(true_inside_V2)
        count_out_V2 = n_points - count_in_V2

        inside_both = np.sum(np.logical_and(true_inside_V1, true_inside_V2))
        inside_neither = np.sum(np.logical_not(
            np.logical_or(true_inside_V1, true_inside_V2)))
        inside_V1 = np.sum(np.logical_and(
            true_inside_V1, np.logical_not(true_inside_V2)))
        inside_V2 = np.sum(np.logical_and(
            true_inside_V2, np.logical_not(true_inside_V1)))

    elif compare == 2:
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

        true_inside_V2 = point_query(
            points_in_mask_copy, kdTree_V2, V2_gzA, False)
        count_in_V2 = np.sum(true_inside_V2)
        count_out_V2 = n_points - count_in_V2

        for i in range(U):

            delta = np.random.rand(3)

            points_in_mask_copy[0] = points_in_mask[0] + delta[0]
            points_in_mask_copy[1] = points_in_mask[1] + delta[1]
            points_in_mask_copy[2] = points_in_mask[2] + delta[2]
            print('Counting points in V1...', count_in_V1)
            print('Counting points out of V1...', count_out_V1)
            true_inside_V1 = point_query(
                points_in_mask_copy, kdTree_V1, V2_gzA, True)
            print('True Inside V1:', true_inside_V1)

            count_in_V1[i] = np.sum(true_inside_V1)

            # The "~" inverts the array. So we have true_inside inverted to add up the falses instead of the trues
            count_out_V1[i] = np.sum(~true_inside_V1)

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


def calculate_ratios_and_stats(count_in_V1, count_out_V1, count_in_V2, count_out_V2, inside_both, inside_neither, inside_V1, inside_V2, n_points):

    print("Calculating ratios and stats", "\n")
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
points_in_mask, points_boolean = mask_point_filter(
    pts, mask, mask_resolution)
###########################################################################################


#

(count_in_V1, count_out_V1, count_in_V2, count_out_V2, inside_both, inside_neither, inside_V1,
 inside_V2, n_points) = count_points(1, points_in_mask, data_table_V1, data_table_V2, compare, V2_gz, V2_gz2)

results = calculate_ratios_and_stats(count_in_V1, count_out_V1, count_in_V2,
                                     count_out_V2, inside_both, inside_neither, inside_V1, inside_V2, n_points)
print(results)


def plot_mask(points_in_mask):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(points_in_mask[0, ::100], points_in_mask[1, ::100], points_in_mask[2, ::100],
               color='blue', s=10, alpha=1,
               label="Points in Mask")

    ax.set(xlabel='X [Mpc/h]',
           ylabel='Y [Mpc/h]',
           zlabel='Z [Mpc/h]')

    plt.legend()
    if save == True:
        fig.savefig('mask.png')
    plt.show()


def plot_maskandboundaries(points_in_mask, pts):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(points_in_mask[0, ::100], points_in_mask[1, ::100], points_in_mask[2, ::100],
               color='blue', s=10, alpha=1,
               label="Points in Mask")

    ax.scatter(pts[0, ::100], pts[1, ::100], pts[2, ::100],
               color='red', s=1, alpha=0.2,
               label="Points from Void Volume")

    ax.set(xlabel='X [Mpc/h]',
           ylabel='Y [Mpc/h]',
           zlabel='Z [Mpc/h]')

    plt.legend()
    if save == True:
        fig.savefig('mask_volume.png')
    plt.show()


if show_plots == True:
    plot_mask(points_in_mask, save)
    plot_maskandboundaries(points_in_mask, pts, save)

if save == True:
    results.to_csv('results.csv')
