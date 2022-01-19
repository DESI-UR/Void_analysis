import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm, ks_2samp
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import PchipInterpolator
from scipy.special import gammaln
from scipy.optimize import minimize
# import numdifftools as ndt
from astropy.io import ascii
# import dynesty
# from dynesty import plotting as dyplot
import pickle
    
def remove_nan(array):
    return array[np.logical_not(np.isnan(array))]

def KS(data1,data2):
    ks, p = ks_2samp(data1, data2)
    print("KS test (test statistic, p-value): ", (ks, p))
    return ks,p

def uniform(a, b, u):
    """Given u in [0,1], return a uniform number in [a,b]."""
    return a + (b-a)*u

def jeffreys(a, b, u):
    """Given u in [0,1], return a Jeffreys random number in [a,b]."""
    return a**(1-u) * b**u

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

def bin_data(data1,data2,bins_,label, plot=False): 
    density = False
    n1, edges1 = np.histogram(data1, bins=bins_)
    dn1 = np.sqrt(n1)
    x = 0.5*(edges1[1:] + edges1[:-1])
    if density:
        N1 = np.trapz(n1, x1)
        n1, dn1 = n1/N1, dn1/N1

    n2, edges2 = np.histogram(data2, bins=bins_)
    dn2 = np.sqrt(n2)
    # x = 0.5*(edges2[1:] + edges2[:-1])
    if density:
        N2 = np.trapz(n2, x2)
        n2, dn2 = n2/N2, dn2/N2
        
    if plot == True:
        plt.errorbar(x, n1, yerr=dn1, fmt='.')        
        plt.errorbar(x, n2, yerr=dn2, fmt='.')
        plt.xlabel(label)
        plt.ylabel('count') 
        plt.title("Binned Data")
        plt.show()
        
    return  x,n1,n2,dn1,dn2


def mixturemodel_skew(params, x):
    """Mixture of two skew normal distributions.
    
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

def logLjoint1_skew(params, m, n, x):
    """Joint log-likelihood of the two data sets.
    
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
        
    Returns
    -------
    logL : float
        Log likelihood of sets m and n given model parameters.
    """
    s, pars = params[0], params[1:]
    
    lambda1 = mixturemodel_skew(pars, x)
    lambda1[lambda1<=0] = np.finfo(dtype=np.float64).tiny
    
    lambda2 = s*lambda1
    
    return np.sum(m*np.log(lambda1) - lambda1 - gammaln(m+1) + n*np.log(lambda2) - lambda2 - gammaln(n+1))

def nlogLjoint1_skew(params, m, n, x):
    """Negative log-likelihood, for minimizers."""
    return -logLjoint1_skew(params, m, n, x)

def logLjoint2_skew(params, m, n, x):
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
    lambda1 = mixturemodel_skew(params[:8], x)
    lambda1[lambda1<=0] = np.finfo(dtype=np.float64).tiny
    
    lambda2 = mixturemodel_skew(params[8:], x)
    lambda2[lambda2<=0] = np.finfo(dtype=np.float64).tiny
    
    return np.sum(m*np.log(lambda1) - lambda1 - gammaln(m+1) + n*np.log(lambda2) - lambda2 - gammaln(n+1))

def nlogLjoint2_skew(params, m, n, x):
    """Negative log likelihood, for minimizers."""
    return -logLjoint2_skew(params, m, n, x)

def Model_1_fit(bounds1,data1,data2,bins_,label):
    x,n1,n2,dn1,dn2 = bin_data(data1,data2,bins_,label)
    
    # Generate 30 random seeds for the minimizer.
    # Store the result with the lowest -ln(L) in bestfit.
    print("running minimizer...this might take a few minutes...")

    bestfit1 = None

    for i in range(30):
        p0 = [np.random.uniform(b[0], b[1]) for b in bounds1]
        result = minimize(nlogLjoint1_skew, p0, method='L-BFGS-B', args=(n1, n2, x), bounds=bounds1)

        if result.success:
#             print(p0)
#             print('  {:.2f}'.format(result.fun))
            if bestfit1 is None:
                bestfit1 = result
            else:
                if result.fun < bestfit1.fun:
                    bestfit1 = result
                    
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
        
    
def Model_1_sampler(prior_xform1,data1,data2,bins_,label):
    x,n1,n2,dn1,dn2 = bin_data(data1,data2,bins_,label)
    print("running the nested sampler... this might take from minutes to hours...")
    dsampler = dynesty.DynamicNestedSampler(logLjoint1_skew, prior_xform1, ndim=9,
                                        logl_args=(n1, n2, x),
                                        nlive=2000,
                                        bound='multi',
                                        sample='auto')

    dsampler.run_nested()
    dres1 = dsampler.results
    
    with open('sampler_results_model1_'+label, 'wb') as dres1_file:
        pickle.dump(dres1, dres1_file)
    print("sampler output saved as pickle file 'sampler_results_model1_"+label+"'")
    
def Model1_output(data1,data2,bins_,label,sampler_results='sampler_results_model1_'): 
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



def Model_2_fit(bounds2,data1,data2,bins_,label):
    x,n1,n2,dn1,dn2 = bin_data(data1,data2,bins_,label)
    print("running minimizer...this might take a few minutes...")
    bestfit2 = None

    for i in range(20):
        p0 = [np.random.uniform(b[0], b[1]) for b in bounds2]
        result = minimize(nlogLjoint2_skew, p0, method='L-BFGS-B', args=(n1, n2, x), bounds=bounds2)

        if result.success:
#             print(p0)
#             print('  {:.2f}'.format(result.fun))
            if bestfit2 is None:
                bestfit2 = result
            else:
                if result.fun < bestfit2.fun:
                    bestfit2 = result
                    
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
    
    return result
    
def Model_2_sampler(prior_xform2,data1,data2,bins_,label):
    x,n1,n2,dn1,dn2 = bin_data(data1,data2,bins_,label)
    print("running the nested sampler... this might take from minutes to hours...")
    dsampler = dynesty.DynamicNestedSampler(logLjoint2_skew, prior_xform2, ndim=16,
                                            logl_args=(n1, n2, x),
                                            nlive=2000,
                                            bound='multi',
                                            sample='auto')

    dsampler.run_nested()
    dres2 = dsampler.results
    
    with open('sampler_results_model2_'+label, 'wb') as dres2_file:
        pickle.dump(dres2, dres2_file)
    print("sampler output saved as pickle file 'sampler_results_model2_"+label+"'")

def Model2_output(data1,data2,bins_,label,sampler_results='sampler_results_model2_'):
    '''
    sampler_results: path of pickle file where sampler results are saved
    '''
    x,n1,n2,dn1,dn2 = bin_data(data1,data2,bins_,label)

    with open(sampler_results+label, 'rb') as dres2_file:
        dres2 = pickle.load(dres2_file)
        
    print("plotting corner plots...")
    labels = [r'$\alpha$', r'$\mu_\alpha$', r'$\sigma_\alpha$', r'$\xi_\alpha$',
              r'$\beta$',  r'$\mu_\beta$',  r'$\sigma_\beta$',  r'$\xi_\beta$',
              r'$\gamma$',  r'$\mu_\gamma$',  r'$\sigma_\gamma$',  r'$\xi_\gamma$',
              r'$\delta$',  r'$\mu_\delta$',  r'$\sigma_\delta$',  r'$\xi_\delta$']

    fig, axes = dyplot.cornerplot(dres2, smooth=0.03,
                                  labels=labels,
                                  show_titles=True,
                                  quantiles_2d=[1-np.exp(-0.5*r**2) for r in [1.,2.,3]],
                                  quantiles=[0.16, 0.5, 0.84],
                                  fig=plt.subplots(16, 16, figsize=(2.5*16,2.6*16)),
                                  color='#1f77d4')

    fig.savefig('corner_model2_'+label+'.png', dpi=100)
    
    
    mapvals2 = np.zeros(16, dtype=float)
    for i in range(16):
        x16, x50, x84 = dynesty.utils.quantile(dres2.samples[:,i],
                                               np.asarray([0.16, 0.5, 0.84]))
        mapvals2[i] = x50
    print("The maximum a posteriori (MAP) values of the parameters: ",mapvals2)
    
    print("Best fit results: ")
    fig, axes = plt.subplots(2,2, figsize=(10,5), sharex=True,
                             gridspec_kw={'height_ratios':[3,1], 'hspace':0},
                             tight_layout=True)

    ax = axes[0,0]
    ep = ax.errorbar(x, n1, yerr=dn1, fmt='.', alpha=0.5)
    ax.plot(x, mixturemodel_skew(mapvals2[:8], x), color=ep[0].get_color(), label='data set 1')
    ax.set(ylabel='count',
           title=r'$\mathcal{M}_2$ (Skew normal model)')
    ax.grid(ls=':')
    ax.legend(fontsize=10)

    ax = axes[1,0]
    ax.errorbar(x, n1 - mixturemodel_skew(mapvals2[:8], x), yerr=dn1, fmt='.')
    ax.grid(ls=':')

    ax.set(#xlim=(0, 4),
           xlabel=label,)
           #ylim=(-750,750))

    ax = axes[0,1]
    ep = ax.errorbar(x, n2, yerr=dn1, fmt='.', color='#ff7f0e', alpha=0.5);
    ax.plot(x, mixturemodel_skew(mapvals2[8:], x), color=ep[0].get_color(), label='data set 2')
    ax.grid(ls=':')
    ax.set(title='Parameter MAP values')
    ax.legend(fontsize=10)

    ax = axes[1,1]
    ax.errorbar(x, n2 - mixturemodel_skew(mapvals2[8:], x), yerr=dn2, fmt='.', color='#ff7f0e')
    ax.grid(ls=':')

    ax.set(#xlim=(0, 4),
           xlabel=label,)
           #ylim=(-275,275))

    fig.savefig('map_model2_'+label+'.png', dpi=100)
    
    lnZ2 = dres2.logz[-1]
    print("Bayesian Evidence for model 2 : ", lnZ2)
    
    return lnZ2