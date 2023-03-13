from astropy.table import Table
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import time
from sklearn import neighbors
from vast.voidfinder._voidfinder_cython_find_next import MaskChecker
import pickle

mask_file_name = "/Users/lorenzomendoza/Desktop/Research/Function/NSA_main_mask.pickle"
temp_infile = open(mask_file_name, "rb")
mask, mask_resolution = pickle.load(temp_infile)
temp_infile.close()

file_name1 = "/Users/lorenzomendoza/Desktop/Research/Function/VoidFinder-nsa_v1_0_1_main_comoving_holes.txt"
data_table_V1 = Table.read(file_name1, format="ascii.commented_header")

file_name2 = "/Users/lorenzomendoza/Desktop/Research/Function/VoidFinder-nsa_v1_0_1_main_comoving_maximal.txt"
data_table_V1max = Table.read(file_name2, format="ascii.commented_header")

file_name3 = "/Users/lorenzomendoza/Desktop/Research/Function/VoidFinder-nsa_v1_0_1_main_comoving_holes.txt"
data_table_V2 = Table.read(file_name3, format="ascii.commented_header")

file_name4 = "/Users/lorenzomendoza/Desktop/Research/Function/VoidFinder-nsa_v1_0_1_main_comoving_maximal.txt"
data_table_V2max = Table.read(file_name4, format="ascii.commented_header")


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


###########################################################################################
# This line creates the boundaries (like the whole min and max)
xmin, xmax, ymin, ymax, zmin, zmax = calc_volume_boundaries(
    data_table_V1, data_table_V2)

# This line makes creates the points in between
pts = generate_grid_points(xmin, xmax, ymin, ymax, zmin, zmax)

###########################################################################################


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


def point_query(point_coords, sphere_tree, void_cat):

    dist, idx = sphere_tree.query(point_coords.T, k=1)

    true_inside = dist < void_cat['radius'][idx]

    print("Point Query")

    return true_inside


def mask_point_filter(pts, mask, mask_resolution, rmin, rmax):
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


###########################################################################################
points_in_mask, points_boolean, var, n_points = mask_point_filter(
    pts, mask, mask_resolution)
###########################################################################################


def count_points(U, points_in_mask, data_table_V1, data_table_V2):
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
        not_inside_V1_and_V2 = np.logical_and(~true_inside_V1, ~true_inside_V2)
        inside_neither[i] = np.sum(not_inside_V1_and_V2)

        # This is the number of points that are in A but not B
        inside_v1 = np.logical_and(true_inside_V1, ~true_inside_V2)
        inside_V1[i] = np.sum(inside_v1)

        # This is the number of points that are not in A but are in B
        inside_v2 = np.logical_and(~true_inside_V1, true_inside_V2)
        inside_V2[i] = np.sum(inside_v2)
    print(time.time() - start_time)
    print('\nNumber of points inside V1:', count_in_V1)
    print('\nNumber of points outside V2:', count_out_V1)
    print('\nNumber of points inside V1:', count_in_V2)
    print('\nNumber of points outside V2:', count_out_V2)
    print("\nThis is the total number of points: {}".format(n_points))
    return time.time() - start_time, count_in_V1, count_out_V1, count_in_V2, count_out_V2, inside_both, inside_neither, inside_V1, inside_V2


rmin = 0
rmax = 312.89816

inside_V1_and_V2 = np.logical_and(true_inside_V1, true_inside_V2)

np.sum(inside_V1_and_V2), count_in_V1, count_in_V2

not_inside_V1_and_V2 = np.logical_and(~true_inside_V1, ~true_inside_V2)

np.sum(not_inside_V1_and_V2), count_in_V1, count_in_V2

inside_V1 = np.logical_and(true_inside_V1, ~true_inside_V2)

np.sum(inside_V1)

inside_V2 = np.logical_and(~true_inside_V1, true_inside_V2)

np.sum(inside_V2)

r_V1 = count_in_V1 / n_points
print(r_V1)

r_V2 = count_in_V2 / n_points
print(r_V2)

r_V1_V2 = np.sum(inside_V1_and_V2) / n_points
print(r_V1_V2)

r_not_V1_V2 = np.sum(not_inside_V1_and_V2) / n_points
print(r_not_V1_V2)

r_V1_not_V2 = np.sum(inside_V1) / n_points
print(r_V1_not_V2)

r_V2_not_V1 = np.sum(inside_V2) / n_points
print(r_V2_not_V1)

Sum = r_V1 + r_not_V1_V2 + r_V1_not_V2 + r_V2_not_V1

average_V1 = np.mean(count_in_V1)
r_average_V1 = average_V1 / n_points
std_V1 = np.std(count_in_V1)
r_std_V1 = std_V1 / n_points

print('\nRatio of V1 Points:', r_average_V1)
print('\nRatio SD:', r_std_V1)

average_V2 = np.mean(count_in_V2)
r_average_V2 = average_V2 / n_points
std_V2 = np.std(count_in_V2)
r_std_V2 = std_V2 / n_points

print('\nRatio of V2 Points:', r_average_V2)
print('\nRatio SD:', r_std_V2)

average_inside = np.mean(inside_both)
r_average_inside = average_inside / n_points

std_both = np.std(inside_both)
r_std_both = std_both / n_points

print('\nRatio of Points Inside:', r_average_inside)
print('\nRatio SD:', r_std_both)

average_outside = np.mean(inside_neither)
r_average_outside = average_outside / n_points

std_outside = np.std(inside_neither)
r_std_outside = std_outside / n_points

print('\nRatio of Points Outside:', r_average_outside)
print('\nRatio SD:', r_std_outside)

average_in_V1 = np.mean(inside_V1)
r_average_in_V1 = average_in_V1 / n_points

std_in_V1 = np.std(inside_neither)
r_std_in_V1 = std_in_V1 / n_points


print('\Ratio of Points in V1:', r_average_in_V1)
print('\nRatio SD:', r_std_in_V1)

average_in_V2 = np.mean(inside_V2)
r_average_in_V2 = average_in_V2 / n_points

std_in_V2 = np.std(inside_neither)
r_std_in_V2 = std_in_V2 / n_points

print('\Ratio of Points in V2:', average_in_V2)
print('\nRatio SD:', r_std_in_V2)
