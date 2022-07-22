'''
Visually inspect the unit test galaxies and voids for VoidFinder.
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np

from vast.voidfinder import filter_galaxies

from vast.voidfinder.viz import VoidRender, load_void_data, load_galaxy_data

from astropy.table import Table, vstack
################################################################################




################################################################################
# Load data
#-------------------------------------------------------------------------------
old_maximals_filename = '/Users/kellydouglass/Documents/Research/Voids/VAST/python/vast/voidfinder/tests/test_galaxies_redshift_maximal_truth.txt'
new_maximals_filename = '/Users/kellydouglass/Documents/Research/Voids/VAST/test_galaxies_redshift_maximal.txt'

old_maximals_xyz, old_maximals_radii, old_maximals_flags = load_void_data(old_maximals_filename)
new_maximals_xyz, new_maximals_radii, new_maximals_flags = load_void_data(new_maximals_filename)

old_holes_filename = '/Users/kellydouglass/Documents/Research/Voids/VAST/python/vast/voidfinder/tests/test_galaxies_redshift_holes_truth.txt'
new_holes_filename = '/Users/kellydouglass/Documents/Research/Voids/VAST/test_galaxies_redshift_holes.txt'

old_holes_xyz, old_holes_radii, old_holes_flags = load_void_data(old_holes_filename)
new_holes_xyz, new_holes_radii, new_holes_flags = load_void_data(new_holes_filename)
################################################################################




################################################################################
# Generate "truth" data (galaxies and voids)
#-------------------------------------------------------------------------------
# Voids
#-------------------------------------------------------------------------------
num_voids = 2

maximals = Table()
maximals['x'] = [25., 10.]
maximals['y'] = [8., 3.]
maximals['z'] = [0., -1.]

maximals_xyz = np.concatenate((maximals['x'].data.reshape(num_voids,1), 
                               maximals['y'].data.reshape(num_voids,1), 
                               maximals['z'].data.reshape(num_voids,1)), 
                              axis=1)
maximals_radii = np.array([2.5, 1.5])
maximals_flags = np.array([0, 1])

num_holes = 4

holes = Table()
holes['x'] = [24., 10.5]
holes['y'] = [7.9, 3.2]
holes['z'] = [0.1, -0.5]
holes = vstack([holes, maximals])

holes_xyz = np.concatenate((holes['x'].data.reshape(num_holes,1), 
                            holes['y'].data.reshape(num_holes,1), 
                            holes['z'].data.reshape(num_holes,1)), 
                           axis=1)
holes_radii = np.concatenate((np.array([2., 0.5]), maximals_radii))
holes_flags = np.concatenate((np.array([0, 1]), maximals_flags))
#-------------------------------------------------------------------------------
# Galaxies
#-------------------------------------------------------------------------------
ra_range = np.arange(10, 30, 0.5)
dec_range = np.arange(-10, 10, 0.5)
redshift_range = np.arange(0, 0.011, 0.0005)

RA, DEC, REDSHIFT = np.meshgrid(ra_range, dec_range, redshift_range)

galaxies_table = Table()

galaxies_table['ra'] = np.ravel(RA)
galaxies_table['dec'] = np.ravel(DEC)
galaxies_table['redshift'] = np.ravel(REDSHIFT)

#galaxies_table['Rgal'] = c*galaxies_table['redshift']/100.

# All of the galaxies will be brighter than the magnitude limit, so that none of them are removed.
galaxies_table['rabsmag'] = 5*np.random.rand(len(galaxies_table)) - 25.1

w_galaxies_xyz, f_galaxies_xyz = filter_galaxies(galaxies_table, 
                                                 'test_', 
                                                 '', 
                                                 dist_metric='redshift',
                                                 write_table=False)

remove_boolean = np.zeros(len(w_galaxies_xyz), dtype=bool)

for i in range(len(holes)):
    
    d = (holes['x'][i] - w_galaxies_xyz[:,0])**2 + (holes['y'][i] - w_galaxies_xyz[:,1])**2 + (holes['z'][i] - w_galaxies_xyz[:,2])**2
    
    remove_boolean = remove_boolean | (d < holes_radii[i]**2)
    
wall_galaxies_xyz = w_galaxies_xyz[~remove_boolean]
field_galaxies_xyz = np.concatenate([f_galaxies_xyz, w_galaxies_xyz[remove_boolean]])
#-------------------------------------------------------------------------------
################################################################################




################################################################################
# Visualize voids
#-------------------------------------------------------------------------------
true_void_color = np.array([0, 0, 1, 0.8], dtype=np.float32) # blue
old_void_color  = np.array([0, 1, 0, 0.8], dtype=np.float32) # green
new_void_color  = np.array([1, 0, 1, 0.8], dtype=np.float32) # magenta

# Concatenate all the void catalogs together, using the colors given above
all_holes_xyz = np.concatenate((holes_xyz, old_holes_xyz, new_holes_xyz))
all_holes_raii = np.concatenate((holes_radii, old_holes_radii, new_holes_radii))
all_holes_IDs = np.concatenate((np.tile(1, (len(holes),)), 
                                np.tile(2, (len(old_holes_radii),)), 
                                np.tile(3, (len(new_holes_radii),))))

# Set void coloring
all_holes_colors = np.concatenate((np.tile(true_void_color, (len(holes),1)), 
                                   np.tile(old_void_color, (len(old_holes_radii),1)), 
                                   np.tile(new_void_color, (len(new_holes_radii),1))))

# Draw voids
viz = VoidRender(holes_xyz=all_holes_xyz, 
                 holes_radii=all_holes_raii, 
                 holes_group_IDs=all_holes_IDs, 
                 galaxy_xyz=field_galaxies_xyz, 
                 galaxy_color=np.array([1, 0, 0, 1], dtype=np.float32), 
                 wall_galaxy_xyz=wall_galaxies_xyz, 
                 wall_distance=None, 
                 wall_galaxy_color=np.array([0, 0, 0, 1], dtype=np.float32), 
                 galaxy_display_radius=10.0, 
                 remove_void_intersects=2, 
                 filter_for_degenerate=False,
                 void_hole_color=all_holes_colors, 
                 canvas_size=(1600, 1200))
viz.run()
################################################################################






