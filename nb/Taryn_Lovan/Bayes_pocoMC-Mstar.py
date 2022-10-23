'''
Calculate the Bayes factor for the stellar mass using the pocoMC package to 
determine whether the Kirshner87 and SDSS DR7 samples are drawn from the same or 
different parent distributions.
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

import pickle

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
data_directory = '../../../../Data/'
SDSS_filename = data_directory + 'NSA/NSA_v1_0_1_VAGC_vflag-V2-VF.fits'
Kirshner_filename = data_directory + 'KNSA.txt'

hdu = fits.open(SDSS_filename)
SDSS = Table(hdu[1].data)
hdu.close()

Kirshner = Table.read(Kirshner_filename, format='ascii.commented_header')
#-------------------------------------------------------------------------------
# Just keep the main SDSS DR7 footprint
#-------------------------------------------------------------------------------
catalog_SDSS = SDSS[SDSS['IN_DR7_LSS'] == 1]
del SDSS

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
Mstar_SDSS = np.log10(catalog_main['ELPETRO_MASS'])
Mstar_Kirshner = np.log10(Kirshner['MASS'])

del catalog_main, Kirshner
################################################################################




################################################################################
# General parameters and properties
#-------------------------------------------------------------------------------
# Bins
SDSS_bins = np.arange(7.5, 11.5, 0.1)
Kirshner_bins = np.arange(7.5, 11.5, 0.2)

# Parameter labels
labels1_bi = ['s', 'a', r'$\mu_a$', r'$\sigma_a$', 'skew$_a$', 
                   'b', r'$\mu_b$', r'$\sigma_b$', 'skew$_b$']
labels2_bi = ['$a_1$', r'$\mu_{1a}$', r'$\sigma_{1a}$', 'skew$_{1a}$', 
              '$b_1$', r'$\mu_{1b}$', r'$\sigma_{1b}$', 'skew$_{1b}$', 
              '$a_2$', r'$\mu_{2a}$', r'$\sigma_{2a}$', 'skew$_{2a}$', 
              '$b_2$', r'$\mu_{2b}$', r'$\sigma_{2b}$', 'skew$_{2b}$']

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
# Fit the stellar mass distributions with skewnormal distributions
#
# Both one- and two-parent models
# 
# This is a unimodal distribution, but we are fitting it with a sum of two skew 
# normals to account for the extra bumps in the distributions.
#-------------------------------------------------------------------------------
# Bin data
x_SDSS, x_Kirshner, n_SDSS, n_Kirshner, dn_SDSS, dn_Kirshner = bin_data(Mstar_SDSS, 
                                                                        Mstar_Kirshner, 
                                                                        SDSS_bins, 
                                                                        Kirshner_bins)
#-------------------------------------------------------------------------------
# 1-parent model
#-------------------------------------------------------------------------------
'''
fit_bounds1 = [[0.00001, 1],   # s ........ Gaussian a to b scale factor
               [10000, 50000], # a ........ Gaussian a amplitude
               [7.5, 10.45],   # mu_a ..... Gaussian a location
               [0.1, 2],       # sigma_a .. Gaussian a scale
               [-4, 0],        # skew_a ... Gaussian a skew
               [5000, 50000],  # b ........ Gaussian b amplitude
               [10.45, 11],    # mu_b ..... Gaussian b location
               [0.1, 2],       # sigma_b .. Gaussian b scale
               [-4, 0]]        # skew_b ... Gaussian b skew

# Prior samples for M1
prior_samples1 = np.random.uniform(low=np.array(fit_bounds1).T[0], 
                                   high=np.array(fit_bounds1).T[1], 
                                   size=(n_particles, n_dim1))

# pocoMC sampler (parallel)
if __name__ == '__main__':

    with Pool(n_cpus) as pool:

        # Initialize sampler for M1
        sampler1 = pc.Sampler(n_particles=n_particles, 
                              n_dim=n_dim1, 
                              log_likelihood=logLjoint1_skew, 
                              log_prior=log_prior, 
                              bounds=np.array(fit_bounds1), 
                              log_likelihood_args=[n_SDSS, n_Kirshner, x_SDSS, x_Kirshner, 2], 
                              log_prior_args=[np.array(fit_bounds1)], 
                              pool=pool)

        # Run sampler
        sampler1.run(prior_samples1)
        
# Get results
results1 = sampler1.results

# Pickle results
temp_outfile = open('pocoMC_results/sampler_results_Mstar_M1.pickle', 
                    'wb')
pickle.dump((results1), temp_outfile)
temp_outfile.close()

os.system('play -nq -t alsa synth {} sine {}'.format(0.5, 400))

exit()
'''
#-------------------------------------------------------------------------------
# 2-parent model
#-------------------------------------------------------------------------------
fit_bounds2 = [[20000, 50000], # a1 ........ Gaussian A amplitude
               [9.5, 10.6],    # mu_a1 ..... Gaussian A location
               [0.01, 2],      # sigma_a1 .. Gaussian A scale
               [-5, 0],        # skew_a1 ... Gaussian A skew
               [20000, 50000], # b1 ........ Gaussian B amplitude
               [10.6, 11],     # mu_b1 ..... Gaussian B location
               [0.01, 2],      # sigma_b1 .. Gaussian B scale
               [-10, 0],       # skew_b1 ... Gaussian B skew
               [1, 50],        # a2 ........ Gaussian A amplitude
               [8, 10.25],     # mu_a2 ..... Gaussian A location
               [0.1, 2],       # sigma_a2 .. Gaussian A scale
               [-2.5, 5],      # skew_a2 ... Gaussian A skew
               [10, 100],      # b2 ........ Gaussian B amplitude
               [10.25, 11],    # mu_b2 ..... Gaussian B location
               [0.01, 2],      # sigma_b2 .. Gaussian B scale
               [-5, 5]]        # skew_b2 ... Gaussian B skew

# Prior samples for M2
prior_samples2 = np.random.uniform(low=np.array(fit_bounds2).T[0], 
                                   high=np.array(fit_bounds2).T[1], 
                                   size=(n_particles, n_dim2))

# pocoMC sampler (parallel)
if __name__ == '__main__':

    with Pool(n_cpus) as pool:

        # Initialize sampler for M2
        sampler2 = pc.Sampler(n_particles=n_particles, 
                              n_dim=n_dim2, 
                              log_likelihood=logLjoint2_skew, 
                              log_prior=log_prior, 
                              bounds=np.array(fit_bounds2), 
                              log_likelihood_args=[n_SDSS, n_Kirshner, x_SDSS, x_Kirshner, [2,2]], 
                              log_prior_args=[np.array(fit_bounds2)], 
                              pool=pool)

        # Run sampler
        sampler2.run(prior_samples2)
        
# Get results
results2 = sampler2.results

# Pickle results
temp_outfile = open('pocoMC_results/sampler_results_Mstar_M2.pickle', 
                    'wb')
pickle.dump((results2), temp_outfile)
temp_outfile.close()

os.system('play -nq -t alsa synth {} sine {}'.format(0.5, 350))
################################################################################





