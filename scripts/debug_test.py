

import os
import numpy as np
from sklearn import neighbors
from astropy.table import Table, setdiff, vstack

from vast.voidfinder.constants import c
from vast.voidfinder import find_voids, filter_galaxies
from vast.voidfinder.multizmask import generate_mask
from vast.voidfinder.preprocessing import file_preprocess
from vast.voidfinder._voidfinder_cython_find_next import GalaxyMap, \
                                                          Cell_ID_Memory, \
                                                          GalaxyMapCustomDict, \
                                                          HoleGridCustomDict, \
                                                          NeighborMemory, \
                                                          MaskChecker, \
                                                          find_next_prime, \
                                                          _query_first

from vast.voidfinder.viz import VoidRender, \
                                load_galaxy_data, \
                                load_void_data


#RESOURCE_DIR = "/dev/shm"
RESOURCE_DIR = "/tmp"

mask_mode = 0



def test_4_find_voids():
    """
    Identify maximal spheres and holes in the galaxy distribution
    """
    
    
    ra_range = np.arange(10, 30, 0.5)
    dec_range = np.arange(-10, 10, 0.5)
    redshift_range = np.arange(0.0005, 0.011, 0.0005) # 0.0001

    RA, DEC, REDSHIFT = np.meshgrid(ra_range, 
                                    dec_range, 
                                    redshift_range)

    galaxies_table = Table()
    galaxies_table['ra'] = np.ravel(RA)
    galaxies_table['dec'] = np.ravel(DEC)
    galaxies_table['redshift'] = np.ravel(REDSHIFT)
    
    
    rng = np.random.default_rng(107)
    galaxies_shuffled = Table(rng.permutation(galaxies_table))
    galaxies_shuffled['Rgal'] = c*galaxies_shuffled['redshift']/100.
    N_galaxies = len(galaxies_shuffled)

    # All galaxies will be brighter than the magnitude limit, so that none
    # of them are removed
    galaxies_shuffled['rabsmag'] = 5*np.random.rand(N_galaxies) - 25.1

    galaxies_filename = 'test_galaxies.txt'
    galaxies_shuffled.write(galaxies_filename,
                                 format='ascii.commented_header',
                                 overwrite=True)

    gal = np.zeros((N_galaxies+1,3))
    gal[:-1,0] = galaxies_shuffled['Rgal']*np.cos(galaxies_shuffled['ra']*np.pi/180.)*np.cos(galaxies_shuffled['dec']*np.pi/180.)
    gal[:-1,1] = galaxies_shuffled['Rgal']*np.sin(galaxies_shuffled['ra']*np.pi/180.)*np.cos(galaxies_shuffled['dec']*np.pi/180.)
    gal[:-1,2] = galaxies_shuffled['Rgal']*np.sin(galaxies_shuffled['dec']*np.pi/180.)
    gal[-1,:] = np.zeros(3)

    # Minimum maximal sphere radius
    min_maximal_radius = 1. # Mpc/h
    
    
    
    
    
    #f_galaxy_table, f_dist_limits, f_out1_filename, f_out2_filename = \
    #        file_preprocess(galaxies_filename, '', '', dist_metric='redshift')

    # Check the distance limits
    dist_limits = np.zeros(2)
    dist_limits[1] = c*redshift_range[-1]/100.
        
    
    
    
    
    
    #f_mask, f_mask_resolution = generate_mask(galaxies_shuffled, 
    #                                          redshift_range[-1], 
    #                                          dist_metric='redshift', 
    #                                          min_maximal_radius=min_maximal_radius)

    # Check the mask
    mask = np.zeros((360,180), dtype=bool)
    for i in range(int(ra_range[0]), int(ra_range[-1]+1)):
        for j in range(int(dec_range[0] + 90), int(dec_range[-1] + 90)+1):
            mask[i, j] = True
        
    
    
    
    
    
    
    
    f_wall, f_field = filter_galaxies(galaxies_shuffled, 
                                          'test_', 
                                          '', 
                                          dist_metric='redshift', 
                                          )

    # Check the wall galaxy coordinates
    gal_tree = neighbors.KDTree(gal)
    distances, indices = gal_tree.query(gal, k=4)
    dist3 = distances[:,3]
    wall = gal[dist3 < (np.mean(dist3) + 1.5*np.std(dist3))]
    
    # Check the field galaxy coordinates
    field = gal[dist3 >= (np.mean(dist3) + 1.5*np.std(dist3))]
    
    
    
    
    
    
    
    maximals = Table()
    maximals['x'] = [25., 10.]
    maximals['y'] = [8., 3.]
    maximals['z'] = [0., -1.]
    maximals['r'] = [2.5, 1.5]
    maximals['flag'] = [0, 1]

    holes = Table()
    holes['x'] = [24., 10.5]
    holes['y'] = [7.9, 3.2]
    holes['z'] = [0.1, -0.5]
    holes['r'] = [2., 0.5]
    holes['flag'] = [0, 1]
    holes = vstack([holes, maximals])

    # Remove points which fall inside holes
    remove_boolean = np.zeros(len(wall), dtype=bool)
    for i in range(len(holes)):
        d = (holes['x'][i] - wall[:,0])**2 + (holes['y'][i] - wall[:,1])**2 + (holes['z'][i] - wall[:,2])**2
        remove_boolean = remove_boolean | (d < holes['r'][i]**2)


    viz = VoidRender(galaxy_xyz=wall[remove_boolean],
                     wall_galaxy_xyz=wall[~remove_boolean],
                     galaxy_display_radius=20.0,
                     wall_distance=None)

    viz.run()


    ############################################################################
    # Test query_first
    #---------------------------------------------------------------------------
    wall_tree = neighbors.KDTree(wall[~remove_boolean])

    #---------------------------------------------------------------------------
    # Setup the VoidFinder GalaxyMap to test
    #---------------------------------------------------------------------------
    coords_min = np.min(gal, axis=0)

    coords_max = np.max(gal, axis=0)

    box = coords_max - coords_min

    galaxy_map_grid_edge_length = 15.

    ngrid_galaxymap = box/galaxy_map_grid_edge_length
            
    galaxy_map_grid_shape = tuple(np.ceil(ngrid_galaxymap).astype(int))

    print("galaxy_map_grid_shape: ", galaxy_map_grid_shape)
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    mesh_indices = ((wall[~remove_boolean] - coords_min)/galaxy_map_grid_edge_length).astype(np.int64)
    
    pre_galaxy_map = {}

    for idx in range(mesh_indices.shape[0]):

        bin_ID_pqr = tuple(mesh_indices[idx])
        
        if bin_ID_pqr not in pre_galaxy_map:
            
            pre_galaxy_map[bin_ID_pqr] = []
        
        pre_galaxy_map[bin_ID_pqr].append(idx)
        
    del mesh_indices

    num_in_galaxy_map = len(pre_galaxy_map)
    #---------------------------------------------------------------------------


    #---------------------------------------------------------------------------
    galaxy_search_cell_dict = GalaxyMapCustomDict(galaxy_map_grid_shape,
                                                  RESOURCE_DIR)

    offset = 0

    galaxy_map_list = []

    for key in pre_galaxy_map:
        
        indices = np.array(pre_galaxy_map[key], dtype=np.int64)
        
        num_elements = indices.shape[0]
        
        galaxy_map_list.append(indices)
        
        galaxy_search_cell_dict.setitem(*key, offset, num_elements)
        
        offset += num_elements

    galaxy_map_array = np.concatenate(galaxy_map_list)

    del galaxy_map_list

    num_galaxy_map_elements = len(galaxy_search_cell_dict)


    galaxy_map = GalaxyMap(RESOURCE_DIR,
                           mask_mode,
                           wall[~remove_boolean], 
                           coords_min.reshape(1,3), 
                           galaxy_map_grid_edge_length,
                           galaxy_search_cell_dict,
                           galaxy_map_array)
    #---------------------------------------------------------------------------


    #---------------------------------------------------------------------------
    # Test the KDTree against _query_first
    #---------------------------------------------------------------------------
    cell_ID_mem = Cell_ID_Memory(10)

    tree_results = []
    vf_results = []

    '''
    #Get some random points in the fake galaxy survey to check against
    check_points = np.zeros((100,3))
    check_points_ra = rng.uniform(np.min(ra_range), 
                                  np.max(ra_range), 
                                  check_points.shape[0])
    check_points_dec = rng.uniform(np.min(dec_range), 
                                   np.max(dec_range), 
                                   check_points.shape[0])
    check_points_r = 0.01*c*rng.uniform(np.min(redshift_range), 
                                        np.max(redshift_range), 
                                        check_points.shape[0])
    check_points[:,0] = check_points_r*np.cos(check_points_ra*np.pi/180)*np.cos(check_points_dec*np.pi/180)
    check_points[:,1] = check_points_r*np.sin(check_points_ra*np.pi/180)*np.cos(check_points_dec*np.pi/180)
    check_points[:,2] = check_points_r*np.sin(check_points_dec*np.pi/180)
    '''

    # 

    for idx in range(check_points.shape[0]):
        
        curr_point = check_points[idx:idx+1, :]
        
        tree_dist, tree_idx = wall_tree.query(curr_point, 1)

        tree_results.append(tree_idx[0][0])

        
        distidxpair = _query_first(galaxy_map.reference_point_ijk,
                                   galaxy_map.coord_min,
                                   galaxy_map.dl,
                                   galaxy_map.shell_boundaries_xyz,
                                   galaxy_map.cell_center_xyz,
                                   galaxy_map,
                                   cell_ID_mem,
                                   curr_point.astype(np.float64)
                                   )
        

        vf_idx = distidxpair['idx']
        vf_dist = distidxpair['dist']

        vf_results.append(vf_idx)


        if tree_idx[0][0] != vf_idx:
            print("KDTree:", tree_dist[0][0], tree_idx[0][0], wall[~remove_boolean][tree_idx[0][0]])
            print("_query_first:", vf_dist, vf_idx, wall[~remove_boolean][vf_idx])
        
        


    tree_results = np.array(tree_results)
    vf_results = np.array(vf_results)

    print("Found all same neighbors?:", np.all(tree_results==vf_results))
    #---------------------------------------------------------------------------

    ############################################################################



    find_voids([wall[~remove_boolean], np.concatenate([field, wall[remove_boolean]])], 
               'test_', 
               mask=mask, 
               mask_resolution=1,
               dist_limits=dist_limits,
               hole_grid_edge_length=1.0,
               hole_center_iter_dist=0.2, 
               min_maximal_radius=min_maximal_radius, 
               num_cpus=1, 
               pts_per_unit_volume=0.01, # 5
               void_table_filename='test_galaxies_redshift_holes.txt', 
               maximal_spheres_filename='test_galaxies_redshift_maximal.txt', 
               verbose=1)

    # Check maximal spheres
    #f_maximals = Table.read('test_galaxies_redshift_maximal.txt', 
    #                       format='ascii.commented_header')
    #maximals_truth = Table.read('python/vast/voidfinder/tests/test_galaxies_redshift_maximal_truth.txt', 
    #                            format='ascii.commented_header')
    
    #self.assertEqual(len(setdiff(f_maximals, maximals_truth)), 0)

    # Check holes
    f_holes = Table.read('test_galaxies_redshift_holes.txt', 
                         format='ascii.commented_header')
    #holes_truth = Table.read('python/vast/voidfinder/tests/test_galaxies_redshift_holes_truth.txt', 
    #                         format='ascii.commented_header')
    
    holes_xyz, holes_radii, holes_flags = load_void_data('test_galaxies_redshift_holes.txt')
    
    
    print(field.shape)
    print(wall.shape)
    
    viz = VoidRender(holes_xyz,
                     holes_radii,
                     holes_flags,
                     wall[remove_boolean],
                     wall_galaxy_xyz=wall[~remove_boolean],
                     wall_distance=None,
                     galaxy_display_radius=10.0,
                     remove_void_intersects=1,
                     SPHERE_TRIANGULARIZATION_DEPTH=2
                     )
    
    viz.run()


if __name__ == "__main__":
    
    
    test_4_find_voids()
    
    
    





