################################################################################
# VoidFinder - Hoyle & Vogeley (2002)
#
# This is a working example script for running VoidFinder on a simulated data 
# set with periodic boundary conditions.
################################################################################




################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np
import struct
import array as arr
from vast.voidfinder import find_voids, wall_field_separation

from vast.voidfinder.preprocessing import load_data_to_Table
################################################################################



################################################################################
# User inputs
#-------------------------------------------------------------------------------
# Input data file name
sim_filename = "TNG300-3-Dark-mask-Nm=512-th=0.65-sig=2.4.fvol"

# "Survey" name - this will be used as the prefix for all output files
survey_name = "TNG300_"

# File name for maximal spheres
out1_filename = survey_name + "maximals.txt"

# File name for void holes
out2_filename = survey_name + "holes.txt"

# Number of CPUs available for analysis.
# A value of None will use one less than all available CPUs.
num_cpus = 249
################################################################################




################################################################################
# Read in data
#-------------------------------------------------------------------------------
# Read in the simulated data
def load_fvolume_to_table(filename):
    
    F = open(filename,'rb')

    #--- Read header
    #head = F.read(256)
    head = arr.array('b')
    head.fromfile(F, 256)    
    (sizeX,) = struct.unpack('i',head[12:16])
    (sizeY,) = struct.unpack('i',head[16:20])
    (sizeZ,) = struct.unpack('i',head[20:24])
    print('>>> Reading volume of size:', sizeX,sizeY,sizeZ)
    
    den = arr.array('f')
    den.fromfile(F,sizeX*sizeY*sizeZ)    
    F.close()
    den = np.array(den).reshape((sizeX,sizeY,sizeZ)).astype(np.float32)    
    X,Y,Z=np.meshgrid(np.arange(sizeX),np.arange(sizeY),np.arange(sizeZ))
    wall = den!=0
    X = X [wall]
    Y = Y [wall]
    Z = Z [wall]
    
    return X.astype(float), Y.astype(float), Z.astype(float)

x, y, z = load_fvolume_to_table(sim_filename)

#-------------------------------------------------------------------------------
# Restructure the data for the find_voids function
#-------------------------------------------------------------------------------

num_gal = x.shape[0]

wall_coords_xyz = np.concatenate((x.reshape(num_gal,1),
                             y.reshape(num_gal,1),
                             z.reshape(num_gal,1)), axis=1)

wall_coords_xyz += 0.5 #displace points into middle of cells

wall_coords_xyz *= 205/512 #rescale to physical size of simulation

#-------------------------------------------------------------------------------

# Coordinate limits of the simulation
xyz_limits = np.array([[0.,0.,0.],[205.,205.,205.]])
#xyz_limits = np.array([[0.,0.,0.],[512.,512.,512.]])

# Size of a single grid cell
hole_grid_edge_length = 205/512
#hole_grid_edge_length = 10

################################################################################
# Find voids
#-------------------------------------------------------------------------------
print("Finding voids")

find_voids(wall_coords_xyz,
           survey_name,
           mask_type='periodic',
           xyz_limits=xyz_limits,
           #save_after=50000,
           #use_start_checkpoint=True,
           hole_grid_edge_length=hole_grid_edge_length,
           maximal_spheres_filename=out1_filename,
           void_table_filename=out2_filename,
           potential_voids_filename=survey_name + 'potential_voids_list.txt',
           num_cpus=num_cpus,
           verbose=1,
           batch_size=100)
################################################################################

#add verbose, change void size