from astropy.table import Table
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import time
from sklearn import neighbors 
from vast.voidfinder.voidfinder_functions import not_in_mask
import pickle
from vast.voidfinder import ra_dec_to_xyz
import sys

hh = int(sys.argv[1])
ii = int(hh/16)
jj = int(hh/4)%4
kk = hh%4

mask_file_name = "/scratch/sbenzvi_lab/desi/dylanbranch/VAST/VoidFinder/scripts/VF3/SDSS_dr7_"+str(ii)+str(jj)+str(kk)+"_mask.pickle"
temp_infile = open(mask_file_name, "rb")
mask, mask_resolution = pickle.load(temp_infile)
temp_infile.close()

file_name1 = "/scratch/sbenzvi_lab/desi/dylanbranch/VAST/VoidFinder/scripts/VF3/DR7m_"+str(ii)+str(jj)+str(kk)+"_comoving_holes.txt"
data_table_NSA = Table.read(file_name1, format = "ascii.commented_header")
file_name2 = "/scratch/sbenzvi_lab/desi/dylanbranch/VAST/VoidFinder/scripts/VF3/DR7m_"+str(ii)+str(jj)+str(kk)+"_comoving_maximal.txt"
data_table_NSA_max = Table.read(file_name2, format = "ascii.commented_header")
gzdata = Table.read("/scratch/sbenzvi_lab/desi/dylanbranch/VAST/Vsquared/data/V4/DR7_"+str(ii)+str(jj)+str(kk)+"_galzones.dat",format='ascii.commented_header')
zvdata = Table.read("/scratch/sbenzvi_lab/desi/dylanbranch/VAST/Vsquared/data/V4/DR7_"+str(ii)+str(jj)+str(kk)+"_zonevoids.dat",format='ascii.commented_header')
gz = gzdata['zone']
zv = zvdata['void1']
data_table_V2 = np.zeros(len(gz),dtype=bool)
for i,z in enumerate(gz):
    if z>-1:
        if zv[z]>-1:
            data_table_V2[i] = True
gzdata = Table.read("/scratch/sbenzvi_lab/desi/dylanbranch/VAST/Vsquared/data/V4_5/DR7_"+str(ii)+str(jj)+str(kk)+"_5_galzones.dat",format='ascii.commented_header')
zvdata = Table.read("/scratch/sbenzvi_lab/desi/dylanbranch/VAST/Vsquared/data/V4_5/DR7_"+str(ii)+str(jj)+str(kk)+"_5_zonevoids.dat",format='ascii.commented_header')
gz = gzdata['zone']
zv = zvdata['void1']
data_table_V4 = np.zeros(len(gz),dtype=bool)
for i,z in enumerate(gz):
    if z>-1:
        if zv[z]>-1:
            data_table_V4[i] = True
#file_name3 = "/scratch/sbenzvi_lab/desi/dylanbranch/NSA_gv2.npy"
#data_table_V2 = np.load(file_name3)
file_name4 = "/scratch/sbenzvi_lab/desi/dylanbranch/data/DR7_mocks2/DR7m_"+str(ii)+str(jj)+str(kk)+".dat"
data_table_vl = Table.read(file_name4, format = "ascii.commented_header")

data_table_vl['V2'] = data_table_V2
data_table_vl['V4'] = data_table_V4
galaxies_xyz = ra_dec_to_xyz(data_table_vl)

data_table_vl['x'] = galaxies_xyz[:,0]
data_table_vl['y'] = galaxies_xyz[:,1]
data_table_vl['z'] = galaxies_xyz[:,2]

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
    x_range = np.arange(x_min, x_max)
    y_range = np.arange(y_min, y_max)
    z_range = np.arange(z_min, z_max)

    # Creating a meshgrid from the ranges to 
    X,Y,Z = np.meshgrid(x_range,y_range,z_range)

    x_points = np.ravel(X)
    y_points = np.ravel(Y)
    z_points = np.ravel(Z)
    
    point_coords = np.array([x_points, y_points, z_points])
    
    return point_coords

#This line creates the boundaries (like the whole min and max)
xmin, xmax, ymin, ymax, zmin, zmax = calc_volume_boundaries(data_table_NSA, data_table_vl)

#This line makes creates the points in between 
pts = generate_grid_points(xmin, xmax, ymin, ymax, zmin, zmax)

b = pts.shape

# This is the cell to filter out all the points that we do not want 🥴
start_time = time.time()

rmin = 0
rmax = 332.38565

points_boolean = np.ones(pts.shape[1], dtype = bool)

start_time = time.time()
for i in range(pts.shape[1]):
    
    #   print(pts[:,i].reshape((1,3)).shape)
    #   print(pts[:,i].reshape((1,3)))
    #   print(pts[:,i])
    points_boolean[i] = not_in_mask(pts[:,i].reshape((1,3)), mask, mask_resolution, rmin, rmax)        


points_in_mask = pts[:,~points_boolean]
(var, n_points) = points_in_mask.shape

def point_query(point_coords, void_cat, vf, rev=False):
    """We are creating a function to make a KDTree to find the number of points in 
    and out of a catalogue.
    
    Parameters
    ----------
    point_coords: ndarray has a shape of (3,N)
        This is the list of points to query the given void catalogue. N is the number of points given. 
    void_cat: Astropy Table
        This is the given void catalogue.
    V2: boolean 
        This tells me if my catalog is a V2 catalog or not a V2 catalog.
    
    Returns
    -------
    true_inside: ndarray of shape (N,1)
        Is this the boolean array of length N (same length as point_coords). True means that 1 point 
        is inside the hole.
    """
    
    cx = void_cat['x']
    cy = void_cat['y']
    cz = void_cat['z']

    sphere_coords = np.array([cx, cy, cz])

    start_time = time.time()

    #The .T is meant to transpose the array from (3,1054) to (1054,3)
    sphere_tree = neighbors.KDTree(sphere_coords.T)

    print(time.time() - start_time)

    start_time = time.time()

    dist, idx = sphere_tree.query(point_coords.T, k = 1)

    if vf:

        true_inside = dist < void_cat['radius'][idx]
    
    else: 
        """What goes into the square braket is whatever the name of the column that tells
        me what is in a wall and what in a void
        
        May need adjust 
        """

        if rev:

            true_inside = void_cat["V4"][idx]

        else:

            true_inside = void_cat["V2"][idx] 
    
    return true_inside

start_time = time.time()

true_inside_NSA = point_query(points_in_mask, data_table_NSA, True)

count_in_NSA = np.sum(true_inside_NSA)



# The "~" inverts the array. So we have true_inside inverted to add up the falses instead of the trues
count_out_NSA = np.sum(~true_inside_NSA)

total_NSA = count_in_NSA + count_out_NSA

true_inside_vl = point_query(points_in_mask, data_table_vl, False)

true_inside_v4 = point_query(points_in_mask, data_table_vl, False,True)

count_in_vl = np.sum(true_inside_vl)

# The "~" inverts the array. So we have true_inside inverted to add up the falses instead of the trues
count_out_vl = np.sum(~true_inside_vl)

total_vl = count_in_vl + count_out_vl

count_in_v4 = np.sum(true_inside_v4)

vfv2 = np.sum(true_inside_NSA*true_inside_vl)
vfv4 = np.sum(true_inside_NSA*true_inside_v4)
v2v4 = np.sum(true_inside_vl*true_inside_v4)

#print('\nNumber of points inside NSA VoidFinder: {}'.format(count_in_NSA))
#print('\nNumber of points outside NSA VoidFinder: {}'.format(count_out_NSA))
#print('\nNumber of points inside NSA V2: {}'.format(count_in_vl))
#print('\nNumber of points outside NSA V2: {}'.format(count_out_vl))
#print("\nThis is the total number of points: {}".format(total_vl))

np.save("/scratch/sbenzvi_lab/desi/dylanbranch/data/DR7_mocks2/vcatprops_"+str(ii)+str(jj)+str(kk)+"_X.npy",np.array([count_in_NSA,count_in_vl,count_in_v4,vfv2,vfv4,v2v4,total_vl]))
