from astropy.table import Table
import numpy as np
import time
from sklearn import neighbors
import pickle
import argparse
import sys

# Create argument parser
# import argparse
import pickle

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

if compare == 0:
    # Get 4 txt files from user: 2 data table files for comoving holes of galaxy, 2 data table files for comoving maximal of galaxy.
    comoving_holes_galaxy_file_1 = input(
        "Enter the first data table file for comoving holes of galaxy: ")
    data_table_V1 = Table.read(
        comoving_holes_galaxy_file_1, format="ascii.commented_header")

    comoving_holes_galaxy_file_2 = input(
        "Enter the second data table file for comoving holes of galaxy: ")
    data_table_V2 = Table.read(
        comoving_holes_galaxy_file_2, format="ascii.commented_header")

    comoving_maximal_galaxy_file_1 = input(
        "Enter the first data table file for comoving maximal of galaxy: ")
    data_table_V1max = Table.read(
        comoving_maximal_galaxy_file_1, format="ascii.commented_header")

    comoving_maximal_galaxy_file_2 = input(
        "Enter the second data table file for comoving maximal of galaxy: ")
    data_table_V2max = Table.read(
        comoving_maximal_galaxy_file_2, format="ascii.commented_header")


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


# This line creates the boundaries (like the whole min and max)
xmin, xmax, ymin, ymax, zmin, zmax = calc_volume_boundaries(
    data_table_V1, data_table_V2)

# This line makes creates the points in between
pts = generate_grid_points(xmin, xmax, ymin, ymax, zmin, zmax)

b = pts.shape

print(b)
