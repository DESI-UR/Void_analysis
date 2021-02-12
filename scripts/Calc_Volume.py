import sys

sys.path.insert(1, '/home/ddunham7/Voids/VoidFinder/python/')


import numpy as np
import argparse
from astropy.table import Table

from voidfinder.table_functions import to_array
from voidfinder.volume import calculate_void_volume

from time import time


start_time = time()

################################################################################
# Import data
#-------------------------------------------------------------------------------
#holes_filename = '../../void_catalogs/SDSS/kias1033_5_16grid_comoving_holes.txt'
#maximals_filename = '../../void_catalogs/SDSS/kias1033_5_16grid_comoving_maximal.txt'
parser = argparse.ArgumentParser()
parser.add_argument('num')
number = parser.parse_args()


#num = 8
file_name = 'dr7Data_comoving_'
file_name_in = 'holes_edge_length_'
file_name_out = 'maximal_edge_length_'

holes_filename = file_name + file_name_in + number.num + '.txt'
maximals_filename = file_name + file_name_out + number.num + '.txt'


holes_table = Table.read(holes_filename, format='ascii.commented_header')
maximals_table = Table.read(maximals_filename, format='ascii.commented_header')
################################################################################


################################################################################
# Initialize volume for each void
#-------------------------------------------------------------------------------
maximals_table['volume'] = np.zeros(len(maximals_table))
################################################################################


################################################################################
# Build a dictionary to easily find all of the holes in each void
#-------------------------------------------------------------------------------
index_dict = {}

for idx, flag in enumerate(holes_table['flag']):
    
    if flag not in index_dict:
        index_dict[flag] = []
        
    index_dict[flag].append(idx)
################################################################################


################################################################################
# Calculate volume for each void
#-------------------------------------------------------------------------------
for void in range(len(maximals_table)):
    
    void_table = holes_table[index_dict[void]]
    
    if len(void_table) == 1:
        void_volume = (4/3) * np.pi * void_table['radius'][0]**3
        
    elif len(void_table) == 2:
        void_volume = (4/3) * np.pi * void_table['radius'][0]**3
        void_volume += (4/3) * np.pi * void_table['radius'][1]**3
        
        diffx = void_table['x'][0] - void_table['x'][1]
        diffy = void_table['y'][0] - void_table['y'][1]
        diffz = void_table['z'][0] - void_table['z'][1]
        
        d = np.sqrt(diffx**2 + diffy**2 + diffz**2)
        
        A = np.pi / 12
        overlap_height2 = (void_table['radius'][0] + void_table['radius'][1] - d)**2
        B = d**2 + 2 * d * (void_table['radius'][0] + void_table['radius'][1]) - 3 * (void_table['radius'][0] - void_table['radius'][1])**2
        overlap = A * overlap_height2 * B / d
        
        void_volume -= overlap
        
    else:
        t = time()
        
        void_holes = to_array(void_table)
    
        void_volume = calculate_void_volume(void_holes.astype(np.float32), 
                                            np.array(void_table['radius'], dtype=np.float32), 
                                            0.05)
        
        '''
        x_max = np.max(void_table['x'] + void_table['radius'])
        x_min = np.min(void_table['x'] - void_table['radius'])
        y_max = np.max(void_table['y'] + void_table['radius'])
        y_min = np.min(void_table['y'] - void_table['radius'])
        z_max = np.max(void_table['z'] + void_table['radius'])
        z_min = np.min(void_table['z'] - void_table['radius'])
        
        
        x_size = int((x_max - x_min) / 0.05)
        y_size = int((y_max - y_min) / 0.05)
        z_size = int((z_max - z_min) / 0.05)
        
        
        points_in = 0
        
        for x in np.arange(x_min, x_max, 0.05):
            for y in np.arange(y_min, y_max, 0.05):
                for z in np.arange(z_min, z_max, 0.05):
                    
                    diffx = x - void_table['x']
                    diffy = y - void_table['y']
                    diffz = z - void_table['z']
    
                    d = diffx**2 + diffy**2 + diffz**2
                    
                    if np.any(d < void_table['radius']):
                        points_in += 1
                        
                        
        void_volume = points_in * 0.05**3
        '''
        
        
        print('Hole with', len(void_table), 'holes:', time()-t)
        
    maximals_table['volume'][void] = void_volume
################################################################################


################################################################################
# Save volumes
#-------------------------------------------------------------------------------
maximals_table.write(maximals_filename, 
                     format='ascii.commented_header', 
                     overwrite=True)
################################################################################

print('Total time:', time() - start_time)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
