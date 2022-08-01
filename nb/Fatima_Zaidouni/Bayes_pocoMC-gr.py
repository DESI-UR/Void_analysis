'''
Calculate the Bayes factor for the color (g-r) using the pocoMC package to 
determine whether the void and wall samples are drawn from the same or different 
parent distributions.

This is based on the jupyter notebook Bayes_factors-g-r.ipynb
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table
from astropy.io import fits

import numpy as np

import sys

import os
os.environ['OMP_NUM_THREADS'] = '1'

import pocomc as pc

from multiprocessing import Pool

import matplotlib
import matplotlib.pyplot as plt

from functions import log_prior, bin_data, logLjoint1_skew, logLjoint2_skew

np.set_printoptions(threshold=sys.maxsize)

matplotlib.rc('font', size=14)
matplotlib.rc('font', family='DejaVu Sans')
################################################################################




################################################################################
# Data
#-------------------------------------------------------------------------------
#data_directory = '../../../../data/'
data_directory = '../../../../Data/NSA/'
data_filename = data_directory + 'NSA_v1_0_1_VAGC_vflag-V2-VF.fits'

hdu = fits.open(data_filename)
data = Table(hdu[1].data)
hdu.close()
#-------------------------------------------------------------------------------
# Just keep the main SDSS DR7 footprint
#-------------------------------------------------------------------------------
catalog_SDSS = data[data['IN_DR7_LSS'] == 1]
del data

ra_boolean = np.logical_and(catalog_SDSS['RA'] > 110, catalog_SDSS['RA'] < 270)
catalog_north = catalog_SDSS[ra_boolean]
del catalog_SDSS

strip_boolean = np.logical_and.reduce([catalog_north['RA'] > 250, 
                                       catalog_north['RA'] < 269, 
                                       catalog_north['DEC'] > 51, 
                                       catalog_north['DEC'] < 67])
catalog_main = catalog_north[~strip_boolean]
del catalog_north
#-------------------------------------------------------------------------------
# Define array of parameter to be fit
#-------------------------------------------------------------------------------
gr_NSA = np.array(catalog_main['g_r'])
#-------------------------------------------------------------------------------
# Separate galaxies by their LSS classifications
#-------------------------------------------------------------------------------
# V2
wall_v2 = catalog_main['vflag_V2'] == 0
void_v2 = catalog_main['vflag_V2'] == 1
#edge_v2 = catalog_main['vflag_V2'] == 2
#out_v2 = catalog_main['vflag_V2'] == 9

# VoidFinder
wall_vf = catalog_main['vflag_VF'] == 0
void_vf = catalog_main['vflag_VF'] == 1
#edge_vf = catalog_main['vflag_VF'] == 2
#out_vf = catalog_main['vflag_VF'] == 9

del catalog_main
################################################################################




################################################################################
# General parameters and properties
#-------------------------------------------------------------------------------
# Bins
gr_bins = np.arange(0, 1.25, 0.02)

# Parameter labels
labels1_bi = ['s', 'a', r'$\mu_a$', r'$\sigma_a$', 'skew$_a$', 
                   'b', r'$\mu_b$', r'$\sigma_b$', 'skew$_b$']
labels2_bi = ['$a_1$', r'$\mu_{1a}$', r'$\sigma_{1a}$', 'skew$_{1a}$', 
              '$b_1$', r'$\mu_{1b}$', r'$\sigma_{1b}$', 'skew$_{1b}$', 
              '$a_2$', r'$\mu_{2a}$', r'$\sigma_{2a}$', 'skew$_{2a}$', 
              '$b_2$', r'$\mu_{2b}$', r'$\sigma_{2b}$', 'skew$_{2b}$']
labels1_tri = ['s', 'a', r'$\mu_a$', r'$\sigma_a$', 'skew$_a$', 
                    'b', r'$\mu_b$', r'$\sigma_b$', 'skew$_b$', 
                    'c', r'$\mu_c$', r'$\sigma_c$', 'skew$_c$']
labels2_tri = ['$a_1$', r'$\mu_{1a}$', r'$\sigma_{1a}$', 'skew$_{1a}$', 
               '$b_1$', r'$\mu_{1b}$', r'$\sigma_{1b}$', 'skew$_{1b}$', 
               '$c_2$', r'$\mu_{1c}$', r'$\sigma_{1c}$', 'skew$_{1c}$', 
               '$a_2$', r'$\mu_{2a}$', r'$\sigma_{2a}$', 'skew$_{2a}$', 
               '$b_2$', r'$\mu_{2b}$', r'$\sigma_{2b}$', 'skew$_{2b}$', 
               '$c_2$', r'$\mu_{2c}$', r'$\sigma_{2c}$', 'skew$_{2c}$']

# Number of particles to use
n_particles = 1000

# Number of parameters in M1
n_dim1 = len(labels1_bi)

# Number of parameters in M2
n_dim2 = len(labels2_bi)

# Number of CPUs
n_cpus = 10
################################################################################





################################################################################
# Fit the color distributions with skewnormal distributions for V2
#
# Both one- and two-parent models
# 
# This is a bimodal distribution, but we are fitting it with a sum of two/three 
# skew normals to account for the extra bumps in the distributions.
#-------------------------------------------------------------------------------
# Bin data
x, n1, n2, dn1, dn2 = bin_data(gr_NSA[wall_v2], 
                               gr_NSA[void_v2], 
                               gr_bins)
#-------------------------------------------------------------------------------
# 1-parent model
#-------------------------------------------------------------------------------
V2_fit_bounds1 = [[0.1, 1],     # s ........ Gaussian a to b scale factor
                  [500, 5000],  # a ........ Gaussian a amplitude
                  [0, 0.75],    # mu_a ..... Gaussian a location
                  [0.01, 3],    # sigma_a .. Gaussian a scale
                  [-5, 5],      # skew_a ... Gaussian a skew
                  [100, 5000],  # b ........ Gaussian b amplitude
                  [0.75, 1.25], # mu_b ..... Gaussian b location
                  [0.01, 3],    # sigma_b .. Gaussian b scale
                  [-5, 0]]      # skew_b ... Gaussian b skew

# Prior samples for M1
V2_prior_samples1 = np.random.uniform(low=np.array(V2_fit_bounds1).T[0], 
                                      high=np.array(V2_fit_bounds1).T[1], 
                                      size=(n_particles, n_dim1))

# pocoMC sampler (parallel)
if __name__ == '__main__':

    with Pool(n_cpus) as pool:

        # Initialize sampler for M1
        V2_sampler1 = pc.Sampler(n_particles=n_particles, 
                                 n_dim=n_dim1, 
                                 log_likelihood=logLjoint1_skew, 
                                 log_prior=log_prior, 
                                 bounds=np.array(V2_fit_bounds1), 
                                 log_likelihood_args=[n1, n2, x, 2], 
                                 log_prior_args=[np.array(V2_fit_bounds1)], 
                                 pool=pool)

        # Run sampler
        V2_sampler1.run(V2_prior_samples1)
        
# Get results
V2_results1 = V2_sampler1.results

# Corner plot of V2 M1
pc.plotting.corner(V2_results1, 
                   labels=labels1_bi, 
                   dims=range(len(labels1_bi)), 
                   show_titles=True, 
                   quantiles=[0.16, 0.5, 0.84])
plt.show()

# V2 log(z)
lnzM1_V2 = V2_results1['logz'][-1]
#-------------------------------------------------------------------------------
# 2-parent model
#-------------------------------------------------------------------------------
V2_fit_bounds2 = [[100, 2500],  # a1 ........ Gaussian A amplitude
                  [0.2, 0.7],   # mu_a1 ..... Gaussian A location
                  [0.01, 1],    # sigma_a1 .. Gaussian A scale
                  [0, 5],       # skew_a1 ... Gaussian A skew
                  [500, 5000],  # b1 ........ Gaussian B amplitude
                  [0.7, 1.2],   # mu_b1 ..... Gaussian B location
                  [0.01, 1],    # sigma_b1 .. Gaussian B scale
                  [-5, 0],      # skew_b1 ... Gaussian B skew
                  [1000, 5000], # a2 ........ Gaussian A amplitude
                  [0.2, 0.7],   # mu_a2 ..... Gaussian A location
                  [0.01, 1],    # sigma_a2 .. Gaussian A scale
                  [0, 5],       # skew_a2 ... Gaussian A skew
                  [1000, 5000], # b2 ........ Gaussian B amplitude
                  [0.7, 1.2],   # mu_b2 ..... Gaussian B location
                  [0.01, 1],    # sigma_b2 .. Gaussian B scale
                  [-5, 0]]      # skew_b2 ... Gaussian B skew

# Prior samples for M2
V2_prior_samples2 = np.random.uniform(low=np.array(V2_fit_bounds2).T[0], 
                                      high=np.array(V2_fit_bounds2).T[1], 
                                      size=(n_particles, n_dim2))

# pocoMC sampler (parallel)
if __name__ == '__main__':

    with Pool(n_cpus) as pool:

        # Initialize sampler for M1
        V2_sampler2 = pc.Sampler(n_particles=n_particles, 
                                 n_dim=n_dim2, 
                                 log_likelihood=logLjoint2_skew, 
                                 log_prior=log_prior, 
                                 bounds=np.array(V2_fit_bounds2), 
                                 log_likelihood_args=[n1, n2, x, 2], 
                                 log_prior_args=[np.array(V2_fit_bounds2)], 
                                 pool=pool)

        # Run sampler
        V2_sampler2.run(V2_prior_samples2)
        
# Get results
V2_results2 = V2_sampler2.results

# Corner plot of V2 M2
pc.plotting.corner(V2_results2, 
                   labels=labels2_bi, 
                   dims=range(len(labels2_bi)), 
                   show_titles=True, 
                   quantiles=[0.16, 0.5, 0.84])
plt.show()

# V2 log(z)
lnzM2_V2 = V2_results2['logz'][-1]
#-------------------------------------------------------------------------------
# Calculate Bayes factor
#-------------------------------------------------------------------------------
lnB12_V2 = lnzM1_V2 - lnzM2_V2

B12_V2 = np.exp(lnB12_V2)

print('V2 g-r: B12 = {:.3g}; log(B12) = {:.3f}'.format(B12_V2, lnB12_V2*np.log10(np.exp(1))))
#-------------------------------------------------------------------------------
################################################################################






################################################################################
# Fit the color distributions with skewnormal distributions for VoidFinder
#
# Both one- and two-parent models
# 
# This is a bimodal distribution, but we are fitting it with a sum of two/three 
# skew normals to account for the extra bumps in the distributions.
#-------------------------------------------------------------------------------
# Bin data
x, n1, n2, dn1, dn2 = bin_data(gr_NSA[wall_vf], 
                               gr_NSA[void_vf], 
                               gr_bins)
#-------------------------------------------------------------------------------
# 1-parent model
#-------------------------------------------------------------------------------
VF_fit_bounds1 = [[1, 5],       # s ........ Gaussian a to b scale factor
                  [500, 10000], # a ........ Gaussian a amplitude
                  [0, 0.75],    # mu_a ..... Gaussian a location
                  [0.01, 3],    # sigma_a .. Gaussian a scale
                  [0, 5],       # skew_a ... Gaussian a skew
                  [500, 10000], # b ........ Gaussian b amplitude
                  [0.75, 1.25], # mu_b ..... Gaussian b location
                  [0.01, 2],    # sigma_b .. Gaussian b scale
                  [-5, 0]]      # skew_b ... Gaussian b skew

# Prior samples for M1
VF_prior_samples1 = np.random.uniform(low=np.array(VF_fit_bounds1).T[0], 
                                      high=np.array(VF_fit_bounds1).T[1], 
                                      size=(n_particles, n_dim1))

# pocoMC sampler (parallel)
if __name__ == '__main__':

    with Pool(n_cpus) as pool:

        # Initialize sampler for M1
        VF_sampler1 = pc.Sampler(n_particles=n_particles, 
                                 n_dim=n_dim1, 
                                 log_likelihood=logLjoint1_skew, 
                                 log_prior=log_prior, 
                                 bounds=np.array(VF_fit_bounds1), 
                                 log_likelihood_args=[n1, n2, x, 2], 
                                 log_prior_args=[np.array(VF_fit_bounds1)], 
                                 pool=pool)

        # Run sampler
        VF_sampler1.run(VF_prior_samples1)
        
# Get results
VF_results1 = VF_sampler1.results

# Corner plot of VoidFinder's M1
pc.plotting.corner(VF_results1, 
                   labels=labels1_bi, 
                   dims=range(len(labels1_bi)), 
                   show_titles=True, 
                   quantiles=[0.16, 0.5, 0.84])
plt.show()

# VoidFinder log(z)
lnzM1_VF = VF_results1['logz'][-1]
#-------------------------------------------------------------------------------
# 2-parent model
#-------------------------------------------------------------------------------
VF_fit_bounds2 = [[500, 5000], # a1 ........ Gaussian A amplitude
                  [0.2, 0.7],  # mu_a1 ..... Gaussian A location
                  [0.01, 1],   # sigma_a1 .. Gaussian A scale
                  [0, 5],      # skew_a1 ... Gaussian A skew
                  [500, 5000], # b1 ........ Gaussian B amplitude
                  [0.7, 1.2],  # mu_b1 ..... Gaussian B location
                  [0.01, 1],   # sigma_b1 .. Gaussian B scale
                  [-5, 0],     # skew_b1 ... Gaussian B skew
                  [100, 5000], # a2 ........ Gaussian A amplitude
                  [0.2, 0.7],  # mu_a2 ..... Gaussian A location
                  [0.01, 1],   # sigma_a2 .. Gaussian A scale
                  [0, 5],      # skew_a2 ... Gaussian A skew
                  [100, 1000], # b2 ........ Gaussian B amplitude
                  [0.7, 1.2],  # mu_b2 ..... Gaussian B location
                  [0.01, 1],   # sigma_b2 .. Gaussian B scale
                  [-5, 0]]     # skew_b2 ... Gaussian B skew

# Prior samples for M2
VF_prior_samples2 = np.random.uniform(low=np.array(VF_fit_bounds2).T[0], 
                                      high=np.array(VF_fit_bounds2).T[1], 
                                      size=(n_particles, n_dim2))

# pocoMC sampler (parallel)
if __name__ == '__main__':

    with Pool(n_cpus) as pool:

        # Initialize sampler for M1
        VF_sampler2 = pc.Sampler(n_particles=n_particles, 
                                 n_dim=n_dim2, 
                                 log_likelihood=logLjoint2_skew, 
                                 log_prior=log_prior, 
                                 bounds=np.array(VF_fit_bounds2), 
                                 log_likelihood_args=[n1, n2, x, 2], 
                                 log_prior_args=[np.array(VF_fit_bounds2)], 
                                 pool=pool)

        # Run sampler
        VF_sampler2.run(VF_prior_samples2)
        
# Get results
VF_results2 = VF_sampler2.results

# Corner plot of VoidFinder M2
pc.plotting.corner(VF_results2, 
                   labels=labels2_bi, 
                   dims=range(len(labels2_bi)), 
                   show_titles=True, 
                   quantiles=[0.16, 0.5, 0.84])
plt.show()

# VoidFinder log(z)
lnzM2_VF = VF_results2['logz'][-1]
#-------------------------------------------------------------------------------
# Calculate Bayes factor
#-------------------------------------------------------------------------------
lnB12_VF = lnzM1_VF - lnzM2_VF

B12_VF = np.exp(lnB12_VF)

print('VoidFinder g-r: B12 = {:.3g}; log(B12) = {:.3f}'.format(B12_VF, lnB12_VF*np.log10(np.exp(1))))
#-------------------------------------------------------------------------------
################################################################################





