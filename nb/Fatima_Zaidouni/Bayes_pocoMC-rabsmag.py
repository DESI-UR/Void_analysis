'''
Calculate the Bayes factor for the absolute magntidue using the pocoMC package 
to determine whether the void and wall samples are drawn from the same or 
different parent distributions.

This is a copy of the jupyter notebook Bayes_pocoMC-rabsmag.ipynb
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table
from astropy.io import fits

import numpy as np

import sys

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
data_filename = '../../../../data/NSA_v1_0_1_VAGC_vflag-V2-VF.fits'

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
rabsmag_NSA = np.array(catalog_main['ELPETRO_ABSMAG'][:,4])
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
# Fit the absolute magnitude distributions with skewnormal distributions
#
# Both one- and two-parent models
# 
# This is a unimodal distribution, but we are fitting it with a sum of two skew 
# normals to account for the extra bumps in the distributions.
#-------------------------------------------------------------------------------
rabsmag_bins = np.arange(-24, -15, 0.1)
#-------------------------------------------------------------------------------
# 1-parent model
#-------------------------------------------------------------------------------
V2_fit_bounds1 = [[1, 4],        # s ........ Gaussian a to b scale factor
                  [2000, 8000],  # a ........ Gaussian a amplitude
                  [-22, -20.25], # mu_a ..... Gaussian a location
                  [1, 3],        # sigma_a .. Gaussian a scale
                  [0, 5],        # skew_a ... Gaussian a skew
                  [2000, 8000],  # b ........ Gaussian b amplitude
                  [-20.25, -18], # mu_b ..... Gaussian b location
                  [0.1, 2],      # sigma_b .. Gaussian b scale
                  [-4, 0]]       # skew_b ... Gaussian b skew

VF_fit_bounds1 = [[0.01, 2],     # s ........ Gaussian a to b scale factor
                  [5000, 20000], # a ........ Gaussian a amplitude
                  [-22, -20],    # mu_a ..... Gaussian a location
                  [0.1, 3],      # sigma_a .. Gaussian a scale
                  [-5, 5],       # skew_ a .. Gaussian a skew
                  [5000, 30000], # b ........ Gaussian b amplitude
                  [-20, -18],    # mu_b ..... Gaussian b location
                  [0.01, 3],     # sigma_b .. Gaussian b scale
                  [-5, 5]]       # skew_b ... Gaussian b skew
#-------------------------------------------------------------------------------
# pocoMC sampling of Likelihood and Priors
#-------------------------------------------------------------------------------
# Number of particles to use
n_particles = 1000

# Number of parameters in M1
n_dim1 = len(V2_fit_bounds1)

# Prior samples for M1
V2_prior_samples1 = np.random.uniform(low=np.array(V2_fit_bounds1).T[0], 
                                      high=np.array(V2_fit_bounds1).T[1], 
                                      size=(n_particles, n_dim1))

# Bin data
x, n1, n2, dn1, dn2 = bin_data(rabsmag_NSA[wall_v2], 
                               rabsmag_NSA[void_v2], 
                               rabsmag_bins)

# Number of CPUs
n_cpus = 4

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
#-------------------------------------------------------------------------------





