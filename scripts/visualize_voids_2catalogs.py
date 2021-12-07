'''
Visually inspect two different void catalogs from the same galaxy catalog.
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np

from scipy.spatial import KDTree

from vast.voidfinder.viz import VoidRender, load_void_data, load_galaxy_data

from vispy.color import Colormap
################################################################################



################################################################################
# Load data
#-------------------------------------------------------------------------------
#maximals1_filename = '../../data/SDSS/vollim_dr7_cbp_102709_all_cells_comoving_maximal.txt'
#maximals2_filename = '../../data/SDSS/vollim_dr7_cbp_102709_empty_cells_only_comoving_maximal.txt'
#maximals1_filename = '../../data/SDSS/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_Voronoi_CMD_vflag_comoving_comoving_maximal.txt'
#maximals2_filename = '../../data/SDSS/vollim_dr7_cbp_102709_comoving_maximal.txt'
maximals1_filename = '../../void_catalogs/SDSS/python_implementation/nsa_v1_0_1_main_comoving_maximal.txt'
maximals2_filename = '../../void_catalogs/SDSS/python_implementation/kias1033_5_MPAJHU_ZdustOS_main_comoving_maximal.txt'

maximals1_xyz, maximals1_radii, maximals1_flags = load_void_data(maximals1_filename)
maximals2_xyz, maximals2_radii, maximals2_flags = load_void_data(maximals2_filename)


#void_catalog1_filename = '../../data/SDSS/vollim_dr7_cbp_102709_all_cells_comoving_holes.txt'
#void_catalog2_filename = '../../data/SDSS/vollim_dr7_cbp_102709_empty_cells_only_comoving_holes.txt'
#void_catalog1_filename = '../../data/SDSS/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_Voronoi_CMD_vflag_comoving_comoving_holes.txt'
#void_catalog2_filename = '../../data/SDSS/vollim_dr7_cbp_102709_comoving_holes.txt'
void_catalog1_filename = '../../void_catalogs/SDSS/python_implementation/nsa_v1_0_1_main_comoving_holes.txt'
void_catalog2_filename = '../../void_catalogs/SDSS/python_implementation/kias1033_5_MPAJHU_ZdustOS_main_comoving_holes.txt'

holes1_xyz, holes1_radii, holes1_flags = load_void_data(void_catalog1_filename)
holes2_xyz, holes2_radii, holes2_flags = load_void_data(void_catalog2_filename)


#galaxy_catalog_filename = '/Users/kellydouglass/Documents/Drexel/Research/Data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_Voronoi_CMD_vflag_comoving.txt'
galaxy_catalog_filename = '/Users/kellydouglass/Documents/Research/data/SDSS/dr7/nsa_v1_0_1_main.txt'

galaxy_data = load_galaxy_data(galaxy_catalog_filename)
################################################################################



################################################################################
# Find nearest void between the two catalogs
#-------------------------------------------------------------------------------
maximals2_tree = KDTree(maximals2_xyz)

dist, idx = maximals2_tree.query(maximals1_xyz)
################################################################################



################################################################################
# Visualize each void pair from the two catalogs
#-------------------------------------------------------------------------------
v1_color = np.array([0,1,0,1])
v2_color = np.array([0.5,0.5,1,1])

for i in range(len(maximals1_flags)):

    #i = np.where(maximals1_flags == 815)[0][0]

    if dist[i] > maximals1_radii[i]:

        v1_flag = maximals1_flags[i]
        v2_flag = maximals2_flags[idx[i]]

        #---------------------------------------------------------------------------
        # Extract holes corresponding to these two voids
        #---------------------------------------------------------------------------
        holes1_flag_boolean = holes1_flags == v1_flag
        holes2_flag_boolean = holes2_flags == v2_flag

        N1 = sum(holes1_flag_boolean)
        N2 = sum(holes2_flag_boolean)

        holes_xyz = np.concatenate((holes1_xyz[holes1_flag_boolean], 
                                  holes2_xyz[holes2_flag_boolean]))
        holes_radii = np.concatenate((holes1_radii[holes1_flag_boolean], 
                                    holes2_radii[holes2_flag_boolean]))
        holes_group_IDs = np.concatenate((np.tile(1, (N1,)),
                                        np.tile(2, (N2,))))
        #---------------------------------------------------------------------------


        #---------------------------------------------------------------------------
        # Set void coloring
        #---------------------------------------------------------------------------
        hole_colors = np.concatenate((np.tile(v1_color, (N1,1)), 
                                    np.tile(v2_color, (N2,1))))
        #---------------------------------------------------------------------------


        #---------------------------------------------------------------------------
        # Draw voids
        #---------------------------------------------------------------------------
        viz = VoidRender(holes_xyz=holes_xyz,
                       holes_radii=holes_radii,
                       holes_group_IDs=holes_group_IDs,
                       galaxy_xyz=galaxy_data,
                       galaxy_display_radius=5,
                       remove_void_intersects=2,
                       void_hole_color=hole_colors,
                       SPHERE_TRIANGULARIZATION_DEPTH=4,
                       canvas_size=(1600,1200),
                       filter_for_degenerate=False)

        viz.run()
        #---------------------------------------------------------------------------
################################################################################
