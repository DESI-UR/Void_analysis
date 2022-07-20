import numpy as np

from scipy.optimize import minimize

import dynesty

import pickle

from functions import bin_data, model_skew, mixturemodel_skew




################################################################################
#-------------------------------------------------------------------------------
def nlogLjoint1_gauss_skew(params, m, n, x, peaks, dm, dn):
    """
    Gaussian joint log-likelihood of the two data sets.
    

    Parameters
    ----------
    params : list or ndarray
        List of 9 parameters: 2x4 skew normal pars + scale factor.
    m : ndarray
        Binned counts in data set 1.
    n : ndarray
        Binned counts in data set 2.
    x : ndarray
        Bin centers used to construct the histogrammed counts m and n.
    peaks : integer
        Number of peaks (skew normals) in the model.
    dm : ndarray
        Uncertainty of binned counts in data set 1.
    dn : ndarray
        Uncertainty of binned counts in data set 2.
        

    Returns
    -------
    logL : float
        Log likelihood of sets m and n given model parameters.
    """

    N = len(n)
    M = len(m)

    s, pars = params[0], params[1:]
    
    if peaks == 1:
        lambda1 = model_skew(pars, x)
    elif peaks == 2:
        lambda1 = mixturemodel_skew(pars, x)
    else:
        print('The mixture model for this many skew normals is not yet defined.')
        exit()

    lambda1[lambda1<=0] = np.finfo(dtype=np.float64).tiny
    
    lambda2 = s*lambda1

    return 0.5*M*np.log(2*np.pi) + np.sum(0.5*((m - lambda1)/dm)**2 + np.log(dm)) + \
           0.5*N*np.log(2*np.pi) + np.sum(0.5*((n - lambda2)/dn)**2 + np.log(dn))
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def logLjoint1_gauss_skew(params, m, n, x, peaks, dm, dn):
    return -nlogLjoint1_gauss_skew(params, m, n, x, peaks, dm, dn)
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def nlogLjoint2_gauss_skew(params, m, n, x, peaks, dm, dn):
    """
    Gaussian joint log-likelihood of the two data sets.
    
    Parameters
    ----------
    params : list or ndarray
        List of 16 parameters: 2x4x2 Gaussian components.
    m : ndarray
        Binned counts in data set 1.
    n : ndarray
        Binned counts in data set 2.
    x : ndarray
        Bin centers used to construct the histogrammed counts m and n.
    dm : ndarray
        Uncertainty of binned counts in data set 1.
    dn : ndarray
        Uncertainty of binned counts in data set 2.

        
    Returns
    -------
    logL : float
        Log likelihood of sets m and n given model parameters.
    """

    N = len(n)
    M = len(m)

    if peaks == 1:
        lambda1 = model_skew(params[:4], x)
        lambda2 = model_skew(params[4:], x)
    elif peaks == 2:
        lambda1 = mixturemodel_skew(params[:8], x)
        lambda2 = mixturemodel_skew(params[8:], x)
    else:
        print('The mixture model for this many skew normals is not yet defined.')
        exit()

    lambda1[lambda1<=0] = np.finfo(dtype=np.float64).tiny
    lambda2[lambda2<=0] = np.finfo(dtype=np.float64).tiny
    
    return 0.5*M*np.log(2*np.pi) + np.sum(0.5*((m - lambda1)/dm)**2 + np.log(dm)) + \
           0.5*N*np.log(2*np.pi) + np.sum(0.5*((n - lambda2)/dn)**2 + np.log(dn))
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def logLjoint2_gauss_skew(params, m, n, x, peaks, dm, dn):
    return -nlogLjoint2_gauss_skew(params, m, n, x, peaks, dm, dn)
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def Model_1_fit(bounds1, data1, data2, bins_, peaks, p0=None):
    '''
    Find the maximum likelihood parameters of model M1 (one-parent model) by 
    minimizing -ln(L).  Performs a bounded fit using the L-BFGS-B algorithm by 
    generating 30 random seeds for the minimizer.


    PARAMETERS
    ==========

    bounds1 : list of length-2 lists
        Parameter bounds.  The length of bounds1 should equal the number of free 
        parameters in the fit.  Each element of the list is [min, max] of that 
        parameter's space.

    data1, data2 : ndarrays of length (N,), (M,)
        data to be fit

    bins_ : ndarray of length (n,)
        Bin edges for binning data

    peaks : integer
        Number of peaks in the parent model

    p0 : list of floats
        Initial guesses at which to start minimize.  If none are given, then 
        minimize will be seeded with random values uniformly generated from 
        within the bounds.


    RETURNS
    =======

    bestfit1 : scipy.optimize.minimize result object
        Contains the best-fit parameters and other output from the 
        scipy.optimize.minimize routine.
    '''

    x, n1, n2, dn1, dn2 = bin_data(data1, data2, bins_)

    # Set all uncertainties equal to 1, so that they do not affect the fits
    dn1 = np.ones_like(dn1, dtype=float)
    dn2 = np.ones_like(dn2, dtype=float)
    
    print("running minimizer...this might take a few minutes...")

    if p0 is not None:
        bestfit1 = minimize(nlogLjoint1_gauss_skew, 
                            p0, 
                            method='L-BFGS-B', 
                            args=(n1, n2, x, peaks, dn1, dn2), 
                            bounds=bounds1)

    else:
        bestfit1 = None

        for i in range(30):

            p0 = [np.random.uniform(b[0], b[1]) for b in bounds1]

            result = minimize(nlogLjoint1_gauss_skew, 
                              p0, 
                              method='L-BFGS-B', 
                              args=(n1, n2, x, peaks, dn1, dn2), 
                              bounds=bounds1)

            if result.success:

                if bestfit1 is None:
                    bestfit1 = result

                else:
                    if result.fun < bestfit1.fun:
                        bestfit1 = result

    return bestfit1
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def Model_1_sampler(prior_, data1, data2, bins_, peaks, fname_suffix=''):
    '''
    Run the dynesty nested sample to generate samples from the posterior PDF of 
    the M1 model.

    The results of the dynesty sampler are pickled for later import.


    PARAMETERS
    ==========

    prior_ : function
        Function to generate priors for the M1 model.  Required by the dynesty 
        sampler.

    data1, data2 : ndarrays of length (N,), (M,)
        data to be fit

    bins_ : ndarray of length (n,)
        Bin edges for binning data

    peaks : integer
        Number of peaks in the parent model
    '''

    x, n1, n2, dn1, dn2 = bin_data(data1, data2, bins_)

    # Set all uncertainties equal to 1, so that they do not affect the fits
    dn1 = np.ones_like(dn1, dtype=float)
    dn2 = np.ones_like(dn2, dtype=float)

    print("running the nested sampler... this might take from minutes to hours...")

    if peaks == 1:
        dsampler = dynesty.DynamicNestedSampler(logLjoint1_gauss_skew, 
                                                prior_, 
                                                ndim=5,
                                                logl_args=(n1, n2, x, peaks, dn1, dn2),
                                                nlive=1000,
                                                bound='multi',
                                                sample='auto')
    elif peaks == 2:
        dsampler = dynesty.DynamicNestedSampler(logLjoint1_gauss_skew, 
                                                prior_, 
                                                ndim=9,
                                                logl_args=(n1, n2, x, peaks, dn1, dn2),
                                                nlive=1000,
                                                bound='multi',
                                                sample='auto')
    else:
        print('The mixture model for this many skew normals is not yet defined.')
        exit()

    dsampler.run_nested()
    dres1 = dsampler.results
    
    with open('dynesty_output/sampler_results_model1_gauss'+fname_suffix+'.pickle', 'wb') as dres1_file:
        pickle.dump(dres1, dres1_file)

    print("sampler output saved as pickle file 'dynesty_output/sampler_results_model1_gauss"+fname_suffix+"'")
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def Model_2_fit(bounds2, data1, data2, bins_, peaks, p0=None):
    '''
    Find the maximum likelihood parameters of model M2 (two-parent model) by 
    minimizing -ln(L).  Performs a bounded fit using the L-BFGS-B algorithm by 
    generating 30 random seeds for the minimizer.


    PARAMETERS
    ==========

    bounds2 : list of length-2 lists
        Parameter bounds.  The length of bounds2 should equal the number of free 
        parameters in the fit.  Each element of the list is [min, max] of that 
        parameter's space.

    data1, data2 : ndarrays of length (N,), (M,)
        data to be fit

    bins_ : ndarray of length (n,)
        Bin edges for binning data

    peaks : integer
        Number of peaks in the parent model

    p0 : list of floats
        Initial guesses at which to start minimize.  If none are given, then 
        minimize will be seeded with random values uniformly generated from 
        within the bounds.


    RETURNS
    =======

    bestfit2 : scipy.optimize.minimize result object
        Contains the best-fit parameters and other output from the 
        scipy.optimize.minimize routine.
    '''

    x, n1, n2, dn1, dn2 = bin_data(data1, data2, bins_)

    # Set all uncertainties equal to 1, so that they do not affect the fits
    dn1 = np.ones_like(dn1, dtype=float)
    dn2 = np.ones_like(dn2, dtype=float)

    print("running minimizer...this might take a few minutes...")

    if p0 is not None:
        bestfit2 = minimize(nlogLjoint2_gauss_skew, 
                            p0, 
                            method='L-BFGS-B', 
                            args=(n1, n2, x, peaks, dn1, dn2), 
                            bounds=bounds2)

    else:
        bestfit2 = None

        for i in range(30):

            p0 = [np.random.uniform(b[0], b[1]) for b in bounds2]

            result = minimize(nlogLjoint2_gauss_skew, 
                              p0, 
                              method='L-BFGS-B', 
                              args=(n1, n2, x, peaks, dn1, dn2), 
                              bounds=bounds2)

            if result.success:

                if bestfit2 is None:
                    bestfit2 = result
                else:
                    if result.fun < bestfit2.fun:
                        bestfit2 = result
    
    return bestfit2
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def Model_2_sampler(prior_, data1, data2, bins_, peaks, fname_suffix=''):
    '''
    Run the dynesty nested sample to generate samples from the posterior PDF of 
    the M2 model.

    The results of the dynesty sampler are pickled for later import.


    PARAMETERS
    ==========

    prior_ : function
        Function to generate priors for the M2 model.  Required for the dynesty 
        sampler.

    data1, data2 : ndarrays of length (N,), (M,)
        data to be fit

    bins_ : ndarray of length (n,)
        Bin edges for binning data

    peaks : integer
        Number of peaks in the parent model
    '''

    x, n1, n2, dn1, dn2 = bin_data(data1, data2, bins_)

    # Set all uncertainties equal to 1, so that they do not affect the fits
    dn1 = np.ones_like(dn1, dtype=float)
    dn2 = np.ones_like(dn2, dtype=float)

    print("running the nested sampler... this might take from minutes to hours...")

    if peaks == 1:
        dsampler = dynesty.DynamicNestedSampler(logLjoint2_gauss_skew, 
                                                prior_, 
                                                ndim=8,
                                                logl_args=(n1, n2, x, peaks, dn1, dn2),
                                                nlive=1000,
                                                bound='multi',
                                                sample='auto')
    elif peaks == 2:
        dsampler = dynesty.DynamicNestedSampler(logLjoint2_gauss_skew, 
                                                prior_, 
                                                ndim=16,
                                                logl_args=(n1, n2, x, peaks, dn1, dn2),
                                                nlive=1000,
                                                bound='multi',
                                                sample='auto')
    else:
        print('The mixture model for this many skew normals is not yet defined.')
        exit()

    dsampler.run_nested()
    dres2 = dsampler.results
    
    with open('dynesty_output/sampler_results_model2_gauss'+fname_suffix+'.pickle', 'wb') as dres2_file:
        pickle.dump(dres2, dres2_file)

    print("sampler output saved as pickle file 'dynesty_output/sampler_results_model2_gauss"+fname_suffix+"'")
################################################################################





