################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np

from sklearn import neighbors
################################################################################



def calc_volume_boundaries(void_cat_A, void_cat_B):
    """
    Compute the boundaries of the minimal rectangular volume (parallelpiped)
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
    """
    Creates a dense rectangular grid of points in 3D for the void volume c
    alculation.
    
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





def point_query(point_coords, void_cat, vf, prune=None):
    """
    We are creating a function to make a KDTree to find the number of points in 
    and out of a catalogue.
    
    Parameters
    ----------
    point_coords: ndarray has a shape of (3,N)
        This is the list of points to query the given void catalogue. N is the 
        number of points given. 

    void_cat: Astropy Table
        This is the given void catalogue.

    vf: boolean 
        This tells me if my catalog is a VoidFinder catalog or not.

    prune: string
        If vf is False (so the catalog is V2), prune contains the field name of 
        the pruning method used.  Default is None.

    
    Returns
    -------
    true_inside: ndarray of shape (N,1)
        This is the boolean array of length N (same length as point_coords). 
        True means that the point is inside a void.
    """
    
    cx = void_cat['x']
    cy = void_cat['y']
    cz = void_cat['z']

    sphere_coords = np.array([cx, cy, cz])

    #start_time = time.time()

    #The .T is meant to transpose the array from (3,1054) to (1054,3)
    sphere_tree = neighbors.KDTree(sphere_coords.T)

    #print(time.time() - start_time)

    #start_time = time.time()

    dist, idx = sphere_tree.query(point_coords.T, k = 1)

    if vf:

        true_inside = dist < void_cat['radius'][idx]
    
    else: 
        """
        What goes into the square braket is whatever the name of the column that 
        tells me what is in a wall and what in a void
        """

        true_inside = void_cat[prune][idx]
    
    return true_inside


