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
import pandas as pd


mask_file_name = "/Users/lorenzomendoza/Desktop/Research/Function/NSA_main_mask.pickle"


temp_infile = open(mask_file_name, "rb")
mask, mask_resolution = pickle.load(temp_infile)
temp_infile.close()


V2_galzones = Table.read(
    "/Users/lorenzomendoza/Desktop/Research/Function/V2_REVOLVER-nsa_v1_0_1_galzones.dat", format='ascii.commented_header')
V2_zonevoids = Table.read(
    "/Users/lorenzomendoza/Desktop/Research/Function/V2_REVOLVER-nsa_v1_0_1_zonevoids.dat", format='ascii.commented_header')

V2_galzones2 = Table.read(
    "/Users/lorenzomendoza/Desktop/Research/Function/V2_REVOLVER-nsa_v1_0_1_galzones.dat", format='ascii.commented_header')
V2_zonevoids2 = Table.read(
    "/Users/lorenzomendoza/Desktop/Research/Function/V2_REVOLVER-nsa_v1_0_1_zonevoids.dat", format='ascii.commented_header')


# V2_gz = np.zeros(len(V2_zonevoids['zone']),dtype=int)
# V2_gz[V2_zonevoids['zone'] > -1] = 1
'''
for i in range(len(V2_gz)):
    if V2_galzones['zone'][i] > -1:
        #V2_gz[i] = V2_zonevoids['void1'][V2_galzones['zone'][i]]
        V2_gz[i] = 1
'''


V2_gz = np.zeros(len(V2_galzones['zone']), dtype=int)

for i in range(len(V2_gz)):

    if V2_zonevoids['void1'][V2_galzones['zone'][i]] > -1:
        V2_gz[i] = 1


V2_gz2 = np.zeros(len(V2_galzones2['zone']), dtype=int)

for i in range(len(V2_gz2)):

    if V2_zonevoids2['void1'][V2_galzones2['zone'][i]] > -1:
        V2_gz2[i] = 1


file_name = "/Users/lorenzomendoza/Desktop/Research/Function/V2_nsa_v1_0_1_gal.txt"
file_name2 = "/Users/lorenzomendoza/Desktop/Research/Function/V2_nsa_v1_0_1_gal.txt"


data_table_vl = Table.read(file_name, format="ascii.commented_header")

data_table_vl2 = Table.read(file_name2, format="ascii.commented_header")


omega_M = np.float32(0.3)
h = np.float32(1.0)


Rgal = z_to_comoving_dist(
    data_table_vl['redshift'].astype(np.float32), omega_M, h)
data_table_vl['Rgal'] = Rgal

Rgal2 = z_to_comoving_dist(
    data_table_vl2['redshift'].astype(np.float32), omega_M, h)
data_table_vl2['Rgal'] = Rgal2


# Edge Case: 513626 = [[-0.  0.  0.]]
z_boolean = data_table_vl['redshift'] > 0
data_table_vl = data_table_vl[z_boolean]


# Edge Case: 513626 = [[-0.  0.  0.]]
z_boolean2 = data_table_vl2['redshift'] > 0
data_table_vl2 = data_table_vl2[z_boolean]


galaxies_xyz = ra_dec_to_xyz(data_table_vl)
galaxies_xyz2 = ra_dec_to_xyz(data_table_vl2)


data_table_vl['x'] = galaxies_xyz[:, 0]
data_table_vl['y'] = galaxies_xyz[:, 1]
data_table_vl['z'] = galaxies_xyz[:, 2]

data_table_vl2['x'] = galaxies_xyz2[:, 0]
data_table_vl2['y'] = galaxies_xyz2[:, 1]
data_table_vl2['z'] = galaxies_xyz2[:, 2]


# create boolean mask
boolmask = np.isin(data_table_vl['index'], V2_galzones['gal'])

# assign values using boolean indexing
V2_galzones['x'] = data_table_vl['x'][boolmask]
V2_galzones['y'] = data_table_vl['y'][boolmask]
V2_galzones['z'] = data_table_vl['z'][boolmask]

# create boolean mask
boolmask2 = np.isin(data_table_vl['index'], V2_galzones['gal'])

# assign values using boolean indexing
V2_galzones2['x'] = data_table_vl2['x'][boolmask2]
V2_galzones2['y'] = data_table_vl2['y'][boolmask2]
V2_galzones2['z'] = data_table_vl2['z'][boolmask2]


x = V2_galzones2['x']
y = V2_galzones2['y']
z = V2_galzones2['z']

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection='3d')

ax.scatter(x, y, z,
           color='maroon', s=0.1, alpha=0.5,
           label='holes')

ax.set(xlabel='X [Mpc/h]', ylabel='Y [Mpc/h]', zlabel='Z [Mpc/h]')

ax.legend(loc='upper right', fontsize=10)

plt.title("Maximal Sphere and Holes of NSA")

plt.show()
