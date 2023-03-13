from astropy.table import Table
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import time
from sklearn import neighbors
from sklearn import decomposition
import joblib
from vast.voidfinder._voidfinder_cython_find_next import MaskChecker
from vast.voidfinder.distance import z_to_comoving_dist
from vast.voidfinder import ra_dec_to_xyz
import pickle


mask_file_name = "/Users/lorenzomendoza/Desktop/Research/Function/NSA_main_mask.pickle"

temp_infile = open(mask_file_name, "rb")
mask, mask_resolution = pickle.load(temp_infile)
temp_infile.close()

V2_galzones = Table.read(
    "/Users/lorenzomendoza/Desktop/Research/Function/V2_REVOLVER-nsa_v1_0_1_galzones.dat", format='ascii.commented_header')
V2_zonevoids = Table.read(
    "/Users/lorenzomendoza/Desktop/Research/Function/V2_REVOLVER-nsa_v1_0_1_zonevoids.dat", format='ascii.commented_header')
V2_gz = np.zeros(len(V2_galzones['zone']), dtype=int)

for i in range(len(V2_gz)):

    if V2_zonevoids['void1'][V2_galzones['zone'][i]] > -1:
        V2_gz[i] = 1

file_name = "/Users/lorenzomendoza/Desktop/Research/Function/V2_nsa_v1_0_1_gal.txt"

data_table_vl = Table.read(file_name, format="ascii.commented_header")

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
    print('Time taken:', time.time() - start_time)
    print('Points in Mask Shape:', points_in_mask.shape)
    print('Sum of Points IN:', np.sum(points_boolean))
    print('Sum of Points OUT:', np.sum(~points_boolean))
    print('Boolean Shape:', points_boolean.shape)
    print('Points in Mask:', points_in_mask)
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
    # print("KDTree")

##############

    return sphere_tree


def point_query(point_coords, sphere_tree, void_cat):
    # print("Starting Query")
    # Void cat classifcation
    true_inside = np.zeros(point_coords.shape[1])

    idx = sphere_tree.query(point_coords.T, k=1, return_distance=False)

    #true_inside = void_cat[idx]
    for i in range(len(idx)):
        true_inside[i] = void_cat[idx[i]]

    return true_inside


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


xmin, xmax, ymin, ymax, zmin, zmax = calc_volume_boundaries(
    V2_galzones, V2_galzones)

# This line makes creates the points in between
pts = generate_grid_points(xmin, xmax, ymin, ymax, zmin, zmax)

b = pts.shape
print(b)

points_in_mask, points_boolean = mask_point_filter(pts, mask, mask_resolution)
count_points(points_in_mask, V2_galzones, V2_galzones, V2_gz)
