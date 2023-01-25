'''
This is a copy of the VF_debug_compare.ipynb notebook, but with the ability to 
run VoidRender on the resulting comparisons.
'''

################################################################################
# Load modules
#-------------------------------------------------------------------------------
import numpy as np

from astropy.table import Table, hstack

import pickle

from vast.voidfinder.viz import VoidRender
################################################################################



"""
################################################################################
# Output files
#
# Read and parse the output files generated in _voidfinder_cython.pyx to convert 
# to astropy tables (for easier comparisons).
#-------------------------------------------------------------------------------
def parse_debug_file_with_radii(filename):
    '''
    Open and parse the output file, converting its contents into an astropy 
    table.
    
    
    PARAMETERS
    ==========
    
    filename : string
        path to the output file
        
    
    RETURNS
    =======
    
    output_table : astropy table
        Table containing the contents of the given file.  Columns include:
          - "cell" : contains the cell coordinates from which the hole was grown
          - "hole" : contains the center hole coordinates
          - "R_hole" : radius of the hole
          - "R1" : radius of the hole after the first bounding galaxy is found
          - "R2" : radius of the hole after the second bounding galaxy is found
          - "R3" : radius of the hole after the third bounding galaxy is found
          - "gal1" : coodinates of the first bounding galaxy
          - "gal2" : coordinates of the second bounding galaxy
          - "gal3" : coordinates of the third bounding galaxy
          - "gal4" : coordinates of the fourth bounding galaxy
    '''
    
    # Read in file contents
    DEBUG_OUTPUT_FILE = open(filename, 'r')
    DEBUG_OUTPUT = DEBUG_OUTPUT_FILE.read()
    DEBUG_OUTPUT_FILE.close()
    
    # Split up the file contents by new line marker
    DEBUG_OUTPUT_lines = DEBUG_OUTPUT.split('\n')
    
    # Number of attributes for each cell
    N = 10
    
    # Number of cells
    n_cells = int(len(DEBUG_OUTPUT_lines)/N)
    
    output_table = Table()
    output_table['cell'] = np.zeros((n_cells, 3), dtype=int)
    output_table['hole'] = np.zeros((n_cells, 3), dtype=np.float32)
    output_table['R_hole'] = np.zeros(n_cells, dtype=np.float32)
    output_table['R1'] = np.zeros(n_cells, dtype=np.float32)
    output_table['R2'] = np.zeros(n_cells, dtype=np.float32)
    output_table['R3'] = np.zeros(n_cells, dtype=np.float32)
    output_table['gal1'] = np.zeros((n_cells, 3), dtype=np.float32)
    output_table['gal2'] = np.zeros((n_cells, 3), dtype=np.float32)
    output_table['gal3'] = np.zeros((n_cells, 3), dtype=np.float32)
    output_table['gal4'] = np.zeros((n_cells, 3), dtype=np.float32)
    
    for i in range(n_cells):
        
        output_table['cell'][i] = np.fromstring(DEBUG_OUTPUT_lines[N*i][19:-1], 
                                                dtype=int, 
                                                sep=', ')
        output_table['hole'][i] = np.fromstring(DEBUG_OUTPUT_lines[N*i+1][14:-1], 
                                                sep=', ')
        output_table['R_hole'][i] = float(DEBUG_OUTPUT_lines[N*i+2][13:])
        output_table['R1'][i] = float(DEBUG_OUTPUT_lines[N*i+3][15:])
        output_table['R2'][i] = float(DEBUG_OUTPUT_lines[N*i+4][15:])
        output_table['R3'][i] = float(DEBUG_OUTPUT_lines[N*i+5][15:])
        output_table['gal1'][i] = np.fromstring(DEBUG_OUTPUT_lines[N*i+6][12:-1], 
                                                sep=', ')
        output_table['gal2'][i] = np.fromstring(DEBUG_OUTPUT_lines[N*i+7][12:-1], 
                                                sep=', ')
        output_table['gal3'][i] = np.fromstring(DEBUG_OUTPUT_lines[N*i+8][12:-1], 
                                                sep=', ')
        output_table['gal4'][i] = np.fromstring(DEBUG_OUTPUT_lines[N*i+9][12:-1], 
                                                sep=', ')
        
    return output_table


def parse_debug_file(filename):
    '''
    Open and parse the output file, converting its contents into an astropy 
    table.
    
    
    PARAMETERS
    ==========
    
    filename : string
        path to the output file
        
    
    RETURNS
    =======
    
    output_table : astropy table
        Table containing the contents of the given file.  Columns include:
          - "cell" : contains the cell coordinates from which the hole was grown
          - "hole" : contains the center hole coordinates
          - "R_hole" : radius of the hole
          - "gal1" : coodinates of the first bounding galaxy
          - "gal2" : coordinates of the second bounding galaxy
          - "gal3" : coordinates of the third bounding galaxy
          - "gal4" : coordinates of the fourth bounding galaxy
    '''
    
    # Read in file contents
    DEBUG_OUTPUT_FILE = open(filename, 'r')
    DEBUG_OUTPUT = DEBUG_OUTPUT_FILE.read()
    DEBUG_OUTPUT_FILE.close()
    
    # Split up the file contents by new line marker
    DEBUG_OUTPUT_lines = DEBUG_OUTPUT.split('\n')
    
    # Number of attributes for each cell
    N = 7
    
    # Number of cells
    n_cells = int(len(DEBUG_OUTPUT_lines)/N)
    
    output_table = Table()
    output_table['cell'] = np.zeros((n_cells, 3), dtype=int)
    output_table['hole'] = np.zeros((n_cells, 3), dtype=np.float32)
    output_table['R_hole'] = np.zeros(n_cells, dtype=np.float32)
    output_table['gal1'] = np.zeros((n_cells, 3), dtype=np.float32)
    output_table['gal2'] = np.zeros((n_cells, 3), dtype=np.float32)
    output_table['gal3'] = np.zeros((n_cells, 3), dtype=np.float32)
    output_table['gal4'] = np.zeros((n_cells, 3), dtype=np.float32)
    
    for i in range(n_cells):
        
        output_table['cell'][i] = np.fromstring(DEBUG_OUTPUT_lines[N*i][19:-1], 
                                                dtype=int, 
                                                sep=', ')
        output_table['hole'][i] = np.fromstring(DEBUG_OUTPUT_lines[N*i+1][14:-1], 
                                                sep=', ')
        output_table['R_hole'][i] = float(DEBUG_OUTPUT_lines[N*i+2][13:])
        output_table['gal1'][i] = np.fromstring(DEBUG_OUTPUT_lines[N*i+3][12:-1], 
                                                sep=', ')
        output_table['gal2'][i] = np.fromstring(DEBUG_OUTPUT_lines[N*i+4][12:-1], 
                                                sep=', ')
        output_table['gal3'][i] = np.fromstring(DEBUG_OUTPUT_lines[N*i+5][12:-1], 
                                                sep=', ')
        output_table['gal4'][i] = np.fromstring(DEBUG_OUTPUT_lines[N*i+6][12:-1], 
                                                sep=', ')
        
    return output_table


print('Reading in files')

Jan_output = parse_debug_file('/Users/kellydouglass/Desktop/Voids_old/Void_analysis/scripts/VF_DEBUG.txt')

June_output = parse_debug_file_with_radii('/Users/kellydouglass/Documents/Research/Voids/code/scripts/VF_DEBUG.txt')
################################################################################




################################################################################
# Compare outputs
# 
# For the cells that result in holes in both versions, are they the same holes?
#-------------------------------------------------------------------------------
print('Comparing files')

keep_rows = []

for i in range(len(Jan_output)):
    
    matched_row = np.all(Jan_output['cell'][i] == June_output['cell'], axis=1)
    
    if np.any(matched_row):
        
        j = np.argwhere(matched_row)[0][0]
        
        gal1_comp = np.all(Jan_output['gal1'][i] == June_output['gal1'][j])
        gal2_comp = np.all(Jan_output['gal2'][i] == June_output['gal2'][j])
        gal3_comp = np.all(Jan_output['gal3'][i] == June_output['gal3'][j])
        gal4_comp = np.all(Jan_output['gal4'][i] == June_output['gal4'][j])
        
        if not np.all([gal1_comp, gal2_comp, gal3_comp, gal4_comp]):
            
            keep_rows.append([i, j])
            
keep_rows = np.array(keep_rows)

diff_gals = hstack([Jan_output[keep_rows[:,0]], June_output[keep_rows[:,1]]])

temp_outfile = open('Jan_June_same_cells.pickle', 'wb')
pickle.dump((diff_gals), temp_outfile)
temp_outfile.close()
################################################################################
"""



temp_infile = open('Jan_June_same_cells.pickle', 'rb')
diff_gals = pickle.load(temp_infile)
temp_infile.close()



################################################################################
# How do the holes compare?
#-------------------------------------------------------------------------------
print('Calculating hole center with first three galaxies')

coord_min = np.array([-331.75566142, -306.9135296, -21.11837496]) # Copied from terminal output

dl = 5.

hole_center_start = (diff_gals['cell_2'] + 0.5)*dl + coord_min

# Find radius of hole after finding first galaxy
hole_radius_1 = np.linalg.norm(hole_center_start - diff_gals['gal1_2'], axis=1)

# Find unit vector pointing from the first galaxy to the center
# (this defines the direction that the hole center moves)
unit_vector_1 = (hole_center_start - diff_gals['gal1_2'])/hole_radius_1[:, np.newaxis]

# Find radius of hole after finding second galaxy
d12 = np.sum((diff_gals['gal1_2'] - diff_gals['gal2_2'])**2, axis=1)
b = np.sum((diff_gals['gal2_2'] - diff_gals['gal1_2'])*unit_vector_1, axis=1)

hole_radius_2 = 0.5*d12/b

# Find new hole center (after second galaxy)
hole_center_2 = diff_gals['gal1_2'] + unit_vector_1*hole_radius_2[:, np.newaxis]

# Find unit vector pointing from the midpoint between the first two galaxies to the center
# (this defines the direction that the hole center moves)
midpoint = 0.5*(diff_gals['gal1_2'] + diff_gals['gal2_2'])

distance_from_center = np.linalg.norm(hole_center_2 - midpoint, axis=1)

unit_vector_2 = (hole_center_2 - midpoint)/distance_from_center[:, np.newaxis]

# Find new hole center (after third galaxy)
x_num = np.sum(diff_gals['gal1_2']**2, axis=1) - np.sum(diff_gals['gal3_2']**2, axis=1) + 2*np.sum(hole_center_2*(diff_gals['gal3_2'] - diff_gals['gal1_2']), axis=1)
x_denom = 2*np.sum(unit_vector_2*(diff_gals['gal1_2'] - diff_gals['gal3_2']), axis=1)

x = x_num/x_denom

hole_center_3 = hole_center_2 + x[:, np.newaxis]*unit_vector_2

# Find radius of hole after finding third galaxy
hole_radius_3 = np.linalg.norm(hole_center_3 - diff_gals['gal1_2'], axis=1)

# Find unit vector pointing from plane of first three galaxies to center
# (this defines the direction that the hole center moves)
v3 = np.cross(diff_gals['gal1_2'] - diff_gals['gal2_2'], 
              diff_gals['gal3_2'] - diff_gals['gal2_2'], 
              axis=1)

unit_vector_3 = v3/np.linalg.norm(v3, axis=1)[:, np.newaxis]

# Make sure that the unit vector is pointing from the plane to the center
plane_to_hole_center = hole_center_3 - unit_vector_3

unit_vector_3[np.sum(plane_to_hole_center*unit_vector_3, axis=1) < 0] *= -1
################################################################################




################################################################################
# Visualize current hole and both fourth galaxies with VoidRender
#-------------------------------------------------------------------------------
viz = VoidRender(holes_xyz=np.array(hole_center_3[3:4]), 
                 holes_radii=np.array(hole_radius_3[3:4]), 
                 holes_group_IDs=np.zeros(1),
                 galaxy_xyz=np.array([diff_gals['gal1_2'][3:4][0], 
                                      diff_gals['gal2_2'][3:4][0], 
                                      diff_gals['gal3_2'][3:4][0]]), 
                 wall_galaxy_xyz=np.array([#diff_gals['gal4_2'][3:4][0], 
                                           diff_gals['gal4_1'][3:4][0]]), 
                 wall_distance=None, 
                 galaxy_display_radius=10
                )
viz.run()
################################################################################




