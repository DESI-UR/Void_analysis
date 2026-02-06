from sunbird.inference.pocomc import PocoMCSampler
from sunbird.inference import priors as sunbird_priors
from sunbird.cosmology.model_params import get_model_params
from sunbird import setup_logging

#import acm.observables.emc as emc
import acm.observables.voidfinder.voidfinder_voids as void_lib
from acm.observables import CombinedObservable

from cosmoprimo import fiducial

from pathlib import Path
import numpy as np
import argparse


def get_priors(cosmo=True, hod=True):
    """
    Return a dictionary of prior distributions, hard limits (ranges),
    and labels for cosmological and HOD parameters in a format
    that is readable by PocoMCSampler.
    """
    stats_module = 'scipy.stats'
    priors, ranges, labels = {}, {}, {}
    if cosmo:
        priors.update(sunbird_priors.AbacusSummit(stats_module).priors)
        ranges.update(sunbird_priors.AbacusSummit(stats_module).ranges)
        labels.update(sunbird_priors.AbacusSummit(stats_module).labels)
    if hod:
        priors.update(sunbird_priors.Yuan23(stats_module).priors)
        ranges.update(sunbird_priors.Yuan23(stats_module).ranges)
        labels.update(sunbird_priors.Yuan23(stats_module).labels)
    return priors, ranges, labels


def get_fixed_params(cosmo_model, hod_model):
    """
    Return a list of fixed parameter names based on the cosmological and HOD models.
    This function checks which parameters are free in the specified models.
    """
    free = []
    # cosmology
    if 'base' in cosmo_model:
        free += ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s']
    if 'w0' in cosmo_model:
        free += ['w0_fld']
    if 'wa' in cosmo_model:
        free += ['wa_fld']
    if 'Nur' in cosmo_model:
        free += ['N_ur']
    if 'nrun' in cosmo_model:
        free += ['nrun']
    if 'fixed-ns' in cosmo_model:
        free.remove('n_s')
    # HOD
    if 'base' in hod_model:
        free += ['logM_cut', 'logM_1', 'sigma', 'alpha', 'kappa']
    if 'AB' in hod_model:
        free += ['B_cen', 'B_sat']
    if "CB" in hod_model:
        free += ["A_cen", "A_sat"]
    if 'VB' in hod_model:
        free += ['alpha_c', 'alpha_s']
    if '_s' in hod_model or '-s' in hod_model:
        free += ['s']
    fixed = [par for par in priors.keys() if par not in free]
    return fixed

def get_filters(observable_name):
    """
    Get the select and slice coordinates for the observable.
    This function returns dictionaries that specify which coordinates to select
    and which to slice for the given observable.
    """
    # select_filters = {'cosmo_hod_idx': args.cosmo_idx, 'hod_idx': args.hod_idx}
    select_filters = {'cosmo_hod_idx': args.cosmo_hod_idx}
    slice_filters = {}
    """Get the select and slice coordinates for the observable."""
    if observable_name == 'GalaxyCorrelationFunctionMultipoles':
        select_filters.update({'multipoles': [0, 2]})
        slice_filters.update({'s': [0.0, 150]})
    elif observable_name == 'GalaxyPowerSpectrumMultipoles':
        select_filters.update({'multipoles': [0, 2, 4]})
    elif observable_name == 'GalaxyBispectrumMultipoles':
        select_filters.update({'multipoles': [0, 2]})
        slice_filters.update({'k': [0.0, 0.7]})
    elif observable_name == 'ReconstructedGalaxyPowerSpectrumMultipoles':
        select_filters.update({'multipoles': [0, 2, 4]})
        slice_filters.update({'k': [0.0, 0.7]})
    elif observable_name == 'DensitySplitPowerSpectrumMultipoles':
        select_filters.update({'statistics': ['quantile_data_power']})
    """elif observable_name == 'VoidFinderVGCCMultipoles':
        select_filters.update({'multipoles': [0, 2]})
        slice_filters.update({'s': [3.5, 150.5]})
    elif observable_name == 'VoidFinderVVACMultipoles':
        select_filters.update({'multipoles': [0, 2]})
        slice_filters.update({'s': [23.5, 100.5]})"""
    if observable_name == 'VoidFinderVGCCMultipoles':
        slice_filters.update({'s': [0, 295]})
    elif observable_name == 'VoidFinderVVACMultipoles':
        slice_filters.update({'s': [0, 155]})
    
    return select_filters, slice_filters

def get_observable(observable_names):
    """Get the observable class by name."""
    paths = {
        'data_dir': '/global/homes/h/hrincon/ACM/compressed/data/',
        'covariance_dir': '/global/homes/h/hrincon/ACM/compressed/cov/',
    }
    if isinstance(observable_names, str):
        observable_names = [observable_names]
    observables = []
    for observable_name in observable_names:
        select_filters, slice_filters = get_filters(observable_name)
        #obs = getattr(emc, observable_name)(
        obs = getattr(void_lib, observable_name)(
            paths=paths, numpy_output=True,
            select_filters=select_filters,
            slice_filters=slice_filters,
        )
        observables.append(obs)
    return obs if len(observables) == 1 else CombinedObservable(observables)

def fit_abacus():
    """
    Fit the AbacusSummit data using the PocoMCSampler.
    This function loads the data, covariance matrix, and model,
    prepares the precision matrix, and samples the posterior distribution.
    It also saves the results, including plots and chain data.
    """
    statistics = observable.stat_name
    print(f'Fitting {statistics} with cosmo_idx={args.cosmo_idx} and hod_idx={args.hod_idx}')
    # load the data
    data_x = observable.x[0]
    #data_x = observable.denormalize(data_x)
    data_x_names = observable.x_names
    data_y = observable.y[0]
    print(f'Loaded data_x with shape: {data_x.shape}')
    print(f'Loaded data_y with shape {data_y.shape}')

    # load the covariance matrix
    covariance_matrix = observable.get_covariance_matrix(volume_factor=64)
    print(f'Loaded covariance matrix with shape: {covariance_matrix.shape}')

    # load the model
    model = observable.get_model_prediction


    # load emulator error matrix
    if add_emulator_error:
        emulator_cov = observable.get_emulator_covariance_matrix(
            method='median', diag=False,
        )
        covariance_matrix += emulator_cov

    # get the debiased inverse
    from acm.utils.tools import get_covariance_correction
    correction = get_covariance_correction(
        n_s=len(observable.covariance_y),
        n_d=len(covariance_matrix),
        n_theta=len(data_x_names) - len(fixed_param_names),
        method='percival',
    )
    precision_matrix = np.linalg.inv(correction * covariance_matrix)

    # a dictionary containing the values of the parameters we want to fix
    fixed_params = {key: data_x[data_x_names.index(key)] for key in fixed_param_names}

    # sample the posterior
    sampler = PocoMCSampler(
        observation=data_y.real,
        precision_matrix=precision_matrix,
        theory_model=model,
        fixed_parameters=fixed_params,
        priors=priors,
        ranges=ranges,
        labels=labels,
        ellipsoid=True,
    )
    sampler(vectorize=True, n_total=4096)

    # plot and save results
    #bestfit = get_bestfit(sampler)
    markers = {key: data_x[data_x_names.index(key)] for key in data_x_names if key not in fixed_params}
    #bestfit.update(fixed_params)
    cosmo = fiducial.AbacusSummit(args.cosmo_idx)
    markers.update({'Omega_m': cosmo['Omega_m'], 'h': cosmo['h']})

    # statistics = '+'.join(statistics)
    if identifier is not None:
        statistics += f'_{identifier}'
    save_dir = '/global/homes/h/hrincon/ACM/inference/'
    save_dir = Path(save_dir) / f'c{args.cosmo_idx:03}_hod{args.hod_idx:03}/cosmo-{cosmo_model}_hod-{hod_model}/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    """Save the chain data and plots to the specified directory."""
    sampler.plot_triangle(save_fn=save_dir / f'chain_{statistics}_triangle.pdf', thin=128,
                        markers=markers, title_limit=1)
    sampler.plot_trace(save_fn=save_dir / f'chain_{statistics}_trace.pdf', thin=128)
    sampler.save_chain(save_fn=save_dir / f'chain_{statistics}.npy', metadata={'markers': markers, 'zeff': 0.5})
    sampler.save_table(save_fn=save_dir / f'chain_{statistics}_stats.txt')
    # observable.plot_observable(model_params=bestfit, save_fn=save_dir / f'chain_{statistics}_bestfit.pdf')
    # sampler.plot_bestfit(save_fn=save_dir / f'chain_{statistics}_mean.png', model='mean')

def get_bestfit(sampler):
    """
    Get the maximum a posterior point from a chain.
    """
    chain = sampler.get_chain(flat=True, thin=1)
    maxp = chain['samples'][chain['log_posterior'].argmax()]
    names = [param for param in sampler.priors.keys() if param not in sampler.fixed_parameters]
    return {key: val for key, val in zip(names, maxp)}


if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument("--cosmo_idx", type=int, default=0)
    parser.add_argument("--hod_idx", type=int, default=0)
    parser.add_argument("--cosmo_hod_idx", type=int, default=0)

    args = parser.parse_args()
    setup_logging()

    todo_cosmo_models = [
        # 'base-Nur'
        'base',
        'base-w0',
        'base-w0-wa',
        'base-Nur-nrun-w0-wa' ,
    ]
    todo_hod_models = ['base-VB-AB-CB-s'] * len(todo_cosmo_models)

    for cosmo_model, hod_model in zip(todo_cosmo_models, todo_hod_models):
        print(f'Running inference for cosmo model: {cosmo_model}, hod model: {hod_model}')

        # set up the inference
        priors, ranges, labels = get_priors(cosmo=True, hod=True)

        identifier = 'Cemu-median'
        fixed_param_names = get_fixed_params(cosmo_model, hod_model)

        # TODO: set back to True once emulator cov added
        add_emulator_error = False

        obs_todo = [
            #'MinkowskiFunctionals',
            # ['ProjectedGalaxyCorrelationFunction', 'GalaxyPowerSpectrumMultipoles'],
            # 'GalaxyCorrelationFunctionMultipoles',
            # 'GalaxyPowerSpectrumMultipoles',
            # 'ReconstructedGalaxyPowerSpectrumMultipoles',
            # ['GalaxyBispectrumMultipoles']
            # 'DensitySplitPowerSpectrumMultipoles',
            # 'WaveletScatteringTransform',
            # 'MinimumSpanningTree',
            # 'DTVoidGalaxyCorrelationFunctionMultipoles',
            # 'VoxelVoidGalaxyCorrelationFunctionMultipoles',
            # # 'VIDEVoidGalaxyDensityProfile',
            # 'VIDEVoidGalaxyCorrelationFunctionMultipoles',
            # 'VIDEVoidSizeFunction',
            # 'CumulantGeneratingFunction',
            # 'GalaxyOverdensityPDF',
            'VoidFinderVGCCMultipoles',
            #'VoidFinderVVACMultipoles'
        ]

        for observable_name in obs_todo:
            observable = get_observable(observable_name)
            fit_abacus()
