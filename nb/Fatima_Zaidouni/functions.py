################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import norm, skewnorm, ks_2samp
from scipy.integrate import trapz
from scipy.special import gammaln
from scipy.optimize import minimize

import dynesty
from dynesty import plotting as dyplot

import pickle
################################################################################





################################################################################
# General plot formatting
#-------------------------------------------------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
################################################################################





################################################################################
#-------------------------------------------------------------------------------
def depth_plot(VF, VF_err, 
               full, full_err, 
               V2, V2_err, 
               r_limits, 
               fillbetween=False, 
               y_label='', 
               filename=None):
    '''
    Plot the galaxy property trend as a function of depth within a V2 void.
    
    
    PARAMETERS
    ==========
    
    VF, VF_err : floats
        The value (and uncertainty) of the property for VoidFinder void 
        galaxies.
        
    full, full_err : floats
        The value (and uncertainty) of the property for all the galaxies in the 
        sample.
    
    V2, V2_err : lists or ndarrays of shape (n,)
        The values (and uncertainties) of the property for V2 void galaxies as 
        a function of distance from the edge of the void.
        
    r_limits : list or ndarray of shape (n,)
        The values of the depth limit for each slice of the V2 void galaxies.
        
    fillbetween : boolean
        Determines whether to plot the uncertainties for V2 as a filled-in 
        region around the data points (True) or as individual error bars 
        (False).  Default value is False.
        
    y_label : string
        Label for the y-axis.  Default is nothing.
        
    filename : string
        Name of the image file if it is to be saved.  Default is None (do not 
        save).
    '''
    
    plt.figure(figsize=(10,6), tight_layout=True)

    # VoidFinder
    plt.fill_between([-0.1, 0.55], 
                     [VF + VF_err], 
                     [VF - VF_err], 
                     color='mistyrose')
    plt.hlines(VF, -0.1, 0.55, colors='r', linestyle='dotted', linewidth=2, label='VoidFinder void')

    # All
    plt.fill_between([-0.1, 0.55], 
                     [full + full_err], 
                     [full - full_err], 
                     color='gainsboro')
    plt.hlines(full, -0.1, 0.55, colors='k', linewidth=2, label='SDSS DR7')

    # V2
    if fillbetween:
        plt.plot(r_limits, V2, '.', color='crimson', markersize=10, label='V$^2$ void')
        plt.fill_between([0.55, *r_limits, -0.05],  
                         np.array([V2[0], *V2, V2[-1]]) + np.array([V2_err[0], *V2_err, V2_err[-1]]), 
                         np.array([V2[0], *V2, V2[-1]]) - np.array([V2_err[0], *V2_err, V2_err[-1]]), 
                         color='crimson', 
                         alpha=0.1)
    else:
        plt.errorbar(r_limits, 
                     V2, 
                     yerr=V2_err, 
                     fmt='x', 
                     color='crimson', 
                     markersize=10, 
                     label='V$^2$ void')
    
    plt.xlabel('Normalized distance to the void boundary', fontsize=24)
    plt.ylabel(y_label, fontsize=24)

    plt.xlim(0.55, -0.05)
    #plt.ylim(0.8, 1.7)

    plt.legend(fontsize=18)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    if filename is not None:
        plt.savefig('figures/' + filename + '.png', format='png', dpi=300)
################################################################################



    
def remove_nan(array):
    return array[np.logical_not(np.isnan(array))]

def KS(data1,data2):
    ks, p = ks_2samp(data1, data2)
    print("KS test (test statistic, p-value): ", (ks, p))
    return ks,p




################################################################################
#-------------------------------------------------------------------------------
def uniform(a, b, u):
    """Given u in [0,1], return a uniform number in [a,b]."""
    return a + (b-a)*u
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def jeffreys(a, b, u):
    """Given u in [0,1], return a Jeffreys random number in [a,b]."""
    return a**(1-u) * b**u
################################################################################



"""
################################################################################
#-------------------------------------------------------------------------------
def prior_xform1_uni(u):
    '''
    Priors for the 5 parameters of unimodel M1.  Required by the dynesty 
    sampler.


    PARAMETERS
    ==========

    u : ndarray
        Array of uniform random numbers between 0 and 1.


    RETURNS
    =======
    
    priors : ndarray
        Transformed random numbers giving prior ranges on model parameters.
    '''

    s = uniform(0.1, 10, u[0])
    a = jeffreys(500, 100000, u[1])
    mu = uniform(8, 12, u[2])
    sigma = uniform(0.1, 3, u[3])
    skew = uniform(-5, 5, u[4])

    return s, a, mu, sigma, skew
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def prior_xform2_uni(u):
    '''
    Priors for the 8 parameters of unimodel M2.  Required by the dynesty 
    sampler.


    PARAMETERS
    ==========

    u : ndarray
        Array of uniform random numbers between 0 and 1.


    RETURNS
    =======
    
    priors : ndarray
        Transformed random numbers giving prior ranges on model parameters.
    '''

    a = jeffreys(500, 100000, u[0])
    mu_a = uniform(8, 12, u[1])
    sigma_a = uniform(0.1, 3, u[2])
    skew_a = uniform(-5, 5, u[3])

    b = jeffreys(500, 100000, u[4])
    mu_b = uniform(8, 12, u[5])
    sigma_b = uniform(0.1, 3, u[6])
    skew_b = uniform(-5, 5, u[7])

    return a, mu_a, sigma_a, skew_a, b, mu_b, sigma_b, skew_b
################################################################################
"""



def plot_hist(data1,data2,bins_,label="property"):
    '''
    Histogram the data (left) and show normalized histograms (right).
    '''
    print("plotting histograms of data...")
    fig, axes = plt.subplots(1,2, figsize=(11,4), sharex=True, tight_layout=True)

    ax = axes[0]
    ax.hist(data1, bins=bins_, alpha=0.5,label="data1")
    ax.hist(data2, bins=bins_, alpha=0.5, label="data2")
    
    ax.set(xlabel=label)
    ax.set_title("data histogram")
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    ax.hist(data1, bins=bins_, alpha=0.5, density=True, label="data1")
    ax.hist(data2, bins=bins_, alpha=0.5, density=True, label="data2")
    
    ax.set(xlabel=label)
    ax.set_title("Normalized data histogram")
    ax.legend()
    
    ax.grid(True)
    
    fig.savefig('data_plot_hist_'+label+'.png', dpi=100)

    plt.show()





################################################################################
#-------------------------------------------------------------------------------
def bin_data(data1, data2, bins_, label='', plot=False, density=False):
    '''
    Histogram the given distributions.


    PARAMETERS
    ==========

    data1, data2 : ndarrays of length (N,), (M,)
        Contains the values to be binned

    bins_ : ndarray of length (n,)
        The bin edges

    label : string
        The x-label used for the plot (if plot == True)

    plot : boolean
        Whether or not to plot the distribution.  Default is False (no plot 
        produced)

    density : boolean
        Whether or not to normalize the count in each bin by the total number of 
        elements.  Default is False (return raw counts).
    '''

    n1, edges1 = np.histogram(data1, bins=bins_)

    dn1 = np.sqrt(n1)

    x = 0.5*(edges1[1:] + edges1[:-1])

    '''
    if density:
        N1 = np.trapz(n1, x1)
        n1, dn1 = n1/N1, dn1/N1
    '''

    n2, edges2 = np.histogram(data2, bins=bins_)

    dn2 = np.sqrt(n2)

    #x = 0.5*(edges2[1:] + edges2[:-1])

    '''
    if density:
        N2 = np.trapz(n2, x2)
        n2, dn2 = n2/N2, dn2/N2
    '''

    if plot == True:
        plt.errorbar(x, n1, yerr=dn1, fmt='.')        
        plt.errorbar(x, n2, yerr=dn2, fmt='.')
        plt.xlabel(label)
        plt.ylabel('count') 
        plt.title("Binned Data")
        plt.show()
        
    return x,n1,n2,dn1,dn2
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def model_skew(params, x):
    """
    One skew normal distribution
    

    Parameters
    ----------

    params : list or ndarray
        List of parameters (expect 1x3).

    x : float or ndarray
        Values to calculate the model.
    

    Returns
    -------
    model : float or ndarray
        Model evaluated at x.
    """

    a, mu, sg, skew = params

    return a*skewnorm.pdf(x, skew, loc=mu, scale=sg)
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def mixturemodel_skew(params, x):
    """
    Mixture of two skew normal distributions.
    

    Parameters
    ----------

    params : list or ndarray
        List of parameters (expect 2x3).

    x : float or ndarray
        Values to calculate the model.
    

    Returns
    -------
    model : float or ndarray
        Mixture model evaluated at x.
    """

    a, mua, sga, askew = params[:4]

    b, mub, sgb, bskew = params[4:]

    return a*skewnorm.pdf(x, askew, loc=mua, scale=sga) + \
           b*skewnorm.pdf(x, bskew, loc=mub, scale=sgb)
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def mixturemodel3_skew(params, x):
    """
    Mixture of three skew normal distributions.
    

    Parameters
    ----------

    params : list or ndarray
        List of parameters (expect 3x4).

    x : float or ndarray
        Values to calculate the model.
    

    Returns
    -------
    model : float or ndarray
        Mixture model evaluated at x.
    """

    a, mua, sga, askew = params[:4]

    b, mub, sgb, bskew = params[4:8]
    
    c, muc, sgc, cskew = params[8:]

    return a*skewnorm.pdf(x, askew, loc=mua, scale=sga) + \
           b*skewnorm.pdf(x, bskew, loc=mub, scale=sgb) + \
           c*skewnorm.pdf(x, cskew, loc=muc, scale=sgc)
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def logLjoint1_skew(params, m, n, x, peaks):
    """
    Joint log-likelihood of the two data sets.
    

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
        

    Returns
    -------
    logL : float
        Log likelihood of sets m and n given model parameters.
    """

    s, pars = params[0], params[1:]
    
    if peaks == 1:
        lambda1 = model_skew(pars, x)
    elif peaks == 2:
        lambda1 = mixturemodel_skew(pars, x)
    elif peaks == 3:
        lambda1 = mixturemodel3_skew(pars, x)
    else:
        print('The mixture model for this many skew normals is not yet defined.')
        exit()

    lambda1[lambda1<=0] = np.finfo(dtype=np.float64).tiny
    
    lambda2 = s*lambda1
    
    return np.sum(m*np.log(lambda1) - lambda1 - gammaln(m+1) + \
                  n*np.log(lambda2) - lambda2 - gammaln(n+1))
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def nlogLjoint1_skew(params, m, n, x, peaks):
    """Negative log-likelihood, for minimizers."""

    return -logLjoint1_skew(params, m, n, x, peaks)
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def logLjoint2_skew(params, m, n, x, peaks):
    """Joint log-likelihood of the two data sets.
    
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
        
    Returns
    -------
    logL : float
        Log likelihood of sets m and n given model parameters.
    """

    if peaks == 1:
        lambda1 = model_skew(params[:4], x)
        lambda2 = model_skew(params[4:], x)
    elif peaks == 2:
        lambda1 = mixturemodel_skew(params[:8], x)
        lambda2 = mixturemodel_skew(params[8:], x)
    elif peaks == 3:
        lambda1 = mixturemodel3_skew(params[:12], x)
        lambda2 = mixturemodel3_skew(params[12:], x)
    else:
        print('The mixture model for this many skew normals is not yet defined.')
        exit()

    lambda1[lambda1<=0] = np.finfo(dtype=np.float64).tiny
    lambda2[lambda2<=0] = np.finfo(dtype=np.float64).tiny
    
    return np.sum(m*np.log(lambda1) - lambda1 - gammaln(m+1) + \
                  n*np.log(lambda2) - lambda2 - gammaln(n+1))
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def nlogLjoint2_skew(params, m, n, x, peaks):
    """Negative log likelihood, for minimizers."""

    return -logLjoint2_skew(params, m, n, x, peaks)
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
    
    print("running minimizer...this might take a few minutes...")

    if p0 is not None:
        bestfit1 = minimize(nlogLjoint1_skew, 
                            p0, 
                            method='L-BFGS-B', 
                            args=(n1, n2, x, peaks), 
                            bounds=bounds1)

    else:
        bestfit1 = None

        for i in range(30):

            p0 = [np.random.uniform(b[0], b[1]) for b in bounds1]

            result = minimize(nlogLjoint1_skew, 
                              p0, 
                              method='L-BFGS-B', 
                              args=(n1, n2, x, peaks), 
                              bounds=bounds1)

            if result.success:
    #             print(p0)
    #             print('  {:.2f}'.format(result.fun))

                if bestfit1 is None:
                    bestfit1 = result

                else:
                    if result.fun < bestfit1.fun:
                        bestfit1 = result
    '''
    print("best fit parameters",bestfit1)
    fig, axes = plt.subplots(2,2, figsize=(10,5), sharex=True,
                             gridspec_kw={'height_ratios':[3,1], 'hspace':0},
                             tight_layout=True)

    ax = axes[0,0]
    ep = ax.errorbar(x, n1, yerr=dn1, fmt='.', alpha=0.5)
    ax.plot(x, mixturemodel_skew(bestfit1.x[1:], x), color=ep[0].get_color(), label='data set 1')
    ax.set(ylabel='count',
           title=r'$\mathcal{M}_1$ (Skew normal model)')
    ax.grid(ls=':')
    ax.legend(fontsize=10)

    ax = axes[1,0]
    ax.errorbar(x, n1 - mixturemodel_skew(bestfit1.x[1:], x), yerr=dn1, fmt='.')
    ax.grid(ls=':')

    ax.set(#xlim=(0, 4),
           xlabel=label,)
           #ylim=(-750,750))

    ax = axes[0,1]
    ep = ax.errorbar(x, n2, yerr=dn1, fmt='.', color='#ff7f0e', alpha=0.5);
    ax.plot(x, bestfit1.x[0]*mixturemodel_skew(bestfit1.x[1:], x), color=ep[0].get_color(), label='data set 2')
    ax.grid(ls=':')
    ax.set(title=r'$-\ln{{\mathcal{{L}}_\mathrm{{max}}}}={{{:.1f}}}$'.format(bestfit1.fun))
    ax.legend(fontsize=10)

    ax = axes[1,1]
    ax.errorbar(x, n2 - bestfit1.x[0]*mixturemodel_skew(bestfit1.x[1:], x), yerr=dn2, fmt='.', color='#ff7f0e')
    ax.grid(ls=':')

    ax.set(#xlim=(0, 4),
           xlabel=label,)
           #ylim=(-275,275))

    fig.savefig('model1_fit_'+label+'.png', dpi=100)
    '''
    return bestfit1
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def Model_1_plot(params, data1, data2, bins_, peaks, xlabel_text='', title_text=''):
    '''
    Plot the binned data and best-fit for the one-parent model.


    PARAMETERS
    ==========
    
    params : parameter values for best-fit
        best-fit values for one-parent model

    data1, data2 : ndarrays of length (N,), (M,)
        data to be fit

    bins_ : ndarray of length (n,)
        Bin edges for binning data

    peaks : integer
        Number of peaks in the parent model

    xlabel_text : string
        Label for x-axis of plot

    title_text : string
        Title for plot
    '''

    x, n1, n2, dn1, dn2 = bin_data(data1, data2, bins_)

    s, pars = params[0], params[1:]

    if peaks == 1:
        m1 = model_skew(pars, x)
        m2 = s*m1
    elif peaks == 2:
        m1 = mixturemodel_skew(pars, x)
        m2 = s*m1
    elif peaks == 3:
        m1 = mixturemodel3_skew(pars, x)
        m2 = s*m1
    else:
        print('The mixture model for this many skew normals is not yet defined.')
        exit()

    ############################################################################
    # Plot distributions and best fits
    #---------------------------------------------------------------------------
    fig, ax = plt.subplots(1,1, figsize=(6,4), tight_layout=True)

    ep = ax.errorbar(x, n1, yerr=dn1, fmt='k.')
    ax.plot(x, m1, color=ep[0].get_color(), label='Wall')

    ep = ax.errorbar(x, n2, yerr=dn2, fmt='r.')
    ax.plot(x, m2, color=ep[0].get_color(), label='Void')

    ax.set_ylabel('count')
    ax.set_xlabel(xlabel_text)
    ax.set_title(title_text)

    ax.legend()

    plt.show()
    ############################################################################
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

    print("running the nested sampler... this might take from minutes to hours...")
    
    dsampler = dynesty.DynamicNestedSampler(logLjoint1_skew, 
                                            prior_, 
                                            ndim=1 + 4*peaks,
                                            logl_args=(n1, n2, x, peaks),
                                            nlive=1000,
                                            bound='multi',
                                            sample='auto'
                                           )

    dsampler.run_nested()
    dres1 = dsampler.results
    
    with open('dynesty_output/sampler_results_model1'+fname_suffix+'.pickle', 'wb') as dres1_file:
        pickle.dump(dres1, dres1_file)

    print("sampler output saved as pickle file 'dynesty_output/sampler_results_model1"+fname_suffix+"'")
################################################################################





def Model_1_output(data1,data2,bins_,label,sampler_results='sampler_results_model1_'): 
    '''
    sampler_results: path of pickle file where sampler results are saved
    '''
    x,n1,n2,dn1,dn2 = bin_data(data1,data2,bins_,label)

    with open(sampler_results+label, 'rb') as dres1_file:
        dres1 = pickle.load(dres1_file)
        
    print("plotting corner plots...")
    labels = ['$s$', r'$\alpha$', r'$\mu_\alpha$', r'$\sigma_\alpha$', r'$\xi_\alpha$',
                 r'$\beta$',  r'$\mu_\beta$',  r'$\sigma_\beta$',  r'$\xi_\beta$']

    fig, axes = dyplot.cornerplot(dres1, smooth=0.03,
                                  labels=labels,
                                  show_titles=True,
                                  quantiles_2d=[1-np.exp(-0.5*r**2) for r in [1.,2.,3]],
                                  quantiles=[0.16, 0.5, 0.84],
                                  fig=plt.subplots(9, 9, figsize=(2.5*9,2.6*9)),
                                  color='#1f77d4')

    fig.savefig('corner_model1_'+label+'.png', dpi=100)
    
    mapvals1 = np.zeros(9, dtype=float)
    for i in range(9):
        x16, x50, x84 = dynesty.utils.quantile(dres1.samples[:,i],
                                               np.asarray([0.16, 0.5, 0.84]))
        mapvals1[i] = x50
        
    print("The maximum a posteriori (MAP) values of the parameters: ",mapvals1)
    print("Best fit results: ")
    fig, axes = plt.subplots(2,2, figsize=(10,5), sharex=True,
                         gridspec_kw={'height_ratios':[3,1], 'hspace':0},
                         tight_layout=True)

    ax = axes[0,0]
    ep = ax.errorbar(x, n1, yerr=dn1, fmt='.', alpha=0.5)
    ax.plot(x, mixturemodel_skew(mapvals1[1:], x), color=ep[0].get_color(), label='data set 1')
    ax.set(ylabel='count',
           title=r'$\mathcal{M}_1$ (Skew normal model)')
    ax.grid(ls=':')
    ax.legend(fontsize=10)

    ax = axes[1,0]
    ax.errorbar(x, n1 - mixturemodel_skew(mapvals1[1:], x), yerr=dn1, fmt='.')
    ax.grid(ls=':')

    ax.set(#xlim=(0, 4),
           xlabel=label,)
           #ylim=(-750,750))

    ax = axes[0,1]
    ep = ax.errorbar(x, n2, yerr=dn1, fmt='.', color='#ff7f0e', alpha=0.5);
    ax.plot(x, mapvals1[0]*mixturemodel_skew(mapvals1[1:], x), color=ep[0].get_color(), label='data set 2')
    ax.grid(ls=':')
    ax.set(title='Parameter MAP values')
    ax.legend(fontsize=10)

    ax = axes[1,1]
    ax.errorbar(x, n2 - mapvals1[0]*mixturemodel_skew(mapvals1[1:], x), yerr=dn2, fmt='.', color='#ff7f0e')
    ax.grid(ls=':')

    ax.set(#xlim=(0, 4),
           xlabel=label,)
           #ylim=(-275,275))

    fig.savefig('map_model1_'+label+'.png', dpi=100)
    
    lnZ1 = dres1.logz[-1]
    print("Bayesian Evidence for model 1 : ", lnZ1)
    
    return lnZ1



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

    print("running minimizer...this might take a few minutes...")

    if p0 is not None:
        bestfit2 = minimize(nlogLjoint2_skew, 
                            p0, 
                            method='L-BFGS-B', 
                            args=(n1, n2, x, peaks), 
                            bounds=bounds2)

    else:
        bestfit2 = None

        for i in range(30):

            p0 = [np.random.uniform(b[0], b[1]) for b in bounds2]

            result = minimize(nlogLjoint2_skew, 
                              p0, 
                              method='L-BFGS-B', 
                              args=(n1, n2, x, peaks), 
                              bounds=bounds2)

            if result.success:
    #             print(p0)
    #             print('  {:.2f}'.format(result.fun))
                if bestfit2 is None:
                    bestfit2 = result
                else:
                    if result.fun < bestfit2.fun:
                        bestfit2 = result
    '''
    print(result.hess_inv)
    print("plotting best fit results...")
    fig, axes = plt.subplots(2,2, figsize=(10,5), sharex=True,
                             gridspec_kw={'height_ratios':[3,1], 'hspace':0},
                             tight_layout=True)

    ax = axes[0,0]
    ep = ax.errorbar(x, n1, yerr=dn1, fmt='.')
    ax.plot(x, mixturemodel_skew(bestfit2.x[:8], x), color=ep[0].get_color(), label='data set 1')
    ax.set(ylabel='count',
           title=r'$\mathcal{M}_2$ (Skew normal model)')
    ax.grid(ls=':')
    ax.legend(fontsize=10)

    ax = axes[1,0]
    ax.errorbar(x, n1 - mixturemodel_skew(bestfit2.x[:8], x), yerr=dn1, fmt='.')
    ax.grid(ls=':')

    ax.set(#xlim=(0, 4),
           xlabel=label,)
           #ylim=(-750,750))

    ax = axes[0,1]
    ep = ax.errorbar(x, n2, yerr=dn1, fmt='.', color='#ff7f0e');
    ax.plot(x, mixturemodel_skew(bestfit2.x[8:], x), color=ep[0].get_color(), label='data set 2')
    ax.grid(ls=':')
    ax.set(title=r'$-\ln{{\mathcal{{L}}_\mathrm{{max}}}}={{{:.1f}}}$'.format(bestfit2.fun))
    ax.legend(fontsize=10)

    ax = axes[1,1]
    ax.errorbar(x, n2 - mixturemodel_skew(bestfit2.x[8:], x), yerr=dn2, fmt='.', color='#ff7f0e')
    ax.grid(ls=':')

    ax.set(#xlim=(0, 4),
           xlabel=label,)
           #ylim=(-275,275))

    fig.savefig('model2_fit_'+label+'.png', dpi=100)
    '''
    return bestfit2
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def Model_2_plot(params, data1, data2, bins_, peaks, xlabel_text='', title_text=''):
    '''
    Plot the binned data and best-fit for the two-parent model.


    PARAMETERS
    ==========
    
    params : parameter values for best-fit
        best-fit values for two-parent model

    data1, data2 : ndarrays of length (N,), (M,)
        data to be fit

    bins_ : ndarray of length (n,)
        Bin edges for binning data

    peaks : integer
        Number of peaks in the parent model

    xlabel_text : string
        Label for x-axis of plot

    title_text : string
        Title for plot
    '''

    x, n1, n2, dn1, dn2 = bin_data(data1, data2, bins_)
    
    n_params = len(params)
    
    pars1 = params[:int(0.5*n_params)]
    pars2 = params[int(0.5*n_params):]

    if peaks == 1:
        m1 = model_skew(pars1, x)
        m2 = model_skew(pars2, x)
    elif peaks == 2:
        m1 = mixturemodel_skew(pars1, x)
        m2 = mixturemodel_skew(pars2, x)
    elif peaks == 3:
        m1 = mixturemodel3_skew(pars1, x)
        m2 = mixturemodel3_skew(pars2, x)
    else:
        print('The mixture model for this many skew normals is not yet defined.')
        exit()

    ############################################################################
    # Plot distributions and best fits
    #---------------------------------------------------------------------------
    fig, ax = plt.subplots(1,1, figsize=(6,4), tight_layout=True)

    ep = ax.errorbar(x, n1, yerr=dn1, fmt='k.')
    ax.plot(x, m1, color=ep[0].get_color(), label='Wall')

    ep = ax.errorbar(x, n2, yerr=dn2, fmt='r.')
    ax.plot(x, m2, color=ep[0].get_color(), label='Void')

    ax.set_ylabel('count')
    ax.set_xlabel(xlabel_text)
    ax.set_title(title_text)

    ax.legend()

    plt.show()
    ############################################################################
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

    print("running the nested sampler... this might take from minutes to hours...")
    
    dsampler = dynesty.DynamicNestedSampler(logLjoint2_skew, 
                                            prior_, 
                                            ndim=8*peaks,
                                            logl_args=(n1, n2, x, peaks),
                                            nlive=1000,
                                            bound='multi',
                                            sample='auto')

    dsampler.run_nested()
    dres2 = dsampler.results
    
    with open('dynesty_output/sampler_results_model2'+fname_suffix+'.pickle', 'wb') as dres2_file:
        pickle.dump(dres2, dres2_file)

    print("sampler output saved as pickle file 'dynesty_output/sampler_results_model2"+fname_suffix+"'")
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def estimate_Jacobian(f, x, params):
    '''
    Numerically estimate the Jacobian array (array of partial first derivatives) of the 
    function f at x for each of the function's parameters.
    
    (Apparently, I need to write this myself because it is not included in any of the 
    normal Python packages...)
    
    
    PARAMETERS
    ==========
    
    f : executable function
        Function of which to calculate partial first derivatives.  Should take x as the 
        first argument and params as the remaining ones.
        
    x : float
        Position in f(x) at which to estimate first partial derivatives.
        
    params : list or ndarray of shape (N,) or (,N)
        Values for each of the parameters of f around which to estimate the partial 
        derivatives.
        
    
    RETURNS
    =======
    
    J : ndarray of shape (N,)
        Values of the partial first derivatives for each param at f(x).
    '''
    
    # Number of parameters
    N = len(params)
    
    # Initialize the Jacobian array
    J = np.zeros(N, dtype=float)
    
    # Define epsilon
    eps = np.sqrt(np.finfo(float).eps)
    
    for i in range(N):
        
        # Adjust the ith parameter by epsilon
        eps_adjust = np.zeros(N, dtype=float)
        eps_adjust[i] = eps
        params_eps = params + eps_adjust
        
        # Estimate the first derivative for the ith parameter in f(x)
        J[i] = (f(x, *params_eps) - f(x, *params))/eps
        
    return J
################################################################################




