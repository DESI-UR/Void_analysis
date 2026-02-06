import pandas as pd
import numpy as np
import os
from astropy.table import Table
from astropy.io import fits
import sys
sys.path.insert(1, '/global/homes/h/hrincon/python_tools')
import Util as util
from pycorr import TwoPointCorrelationFunction
from itertools import product as itertools_product

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from read_voids import get_cosmo_hod_voids

p = ArgumentParser(description='Configuration options for cubic box mocks.',
                   formatter_class=ArgumentDefaultsHelpFormatter)


p.add_argument('-c', '--cosmology', type=str, default='c000',
               help='Abacus simulation cosmology (e.g. c000)')
p.add_argument('-p', '--phase', type=str, default='ph000',
               help='Abacus simulation phase (e.g. ph000)')
p.add_argument('-pl', '--pipeline', type=str, default='ACM',
              help='Mock pipeline (ACM for ACM group mocks, ACM_phase for covariance)')
"""p.add_argument('-t', '--tracer', type=str, default='LRG',
               help='Tracer (e.g. LRG, ELG)')
p.add_argument('-z', '--redshift', type=str, default='z0.500',
               help='redshift snapshot (used for cubic sims only)')
p.add_argument('-s', '--seed', type=str, default='seed0',
               help='HOD populating random seed')
p.add_argument('-d', '--hod', type=str, default='hod000',
               help='hod number (0 for best fit values, higher numbers for matched pairs')
p.add_argument('-st', '--sim_type', type=str, default='Cubic',
              help='Simulation type (Cubic or Cutsky)')
"""
p.add_argument('-nc', '--num_cpus', type=int, default=10,
              help='Number of cpus for multiprocessing. Setting to None uses one less than all available cpus')
args = p.parse_args()


cosm = args.cosmology#'c000'
phase = args.phase#'ph000'
pipeline = args.pipeline#'ACM'
"""
tracer = args.tracer#"LRG"
redshift = args.redshift#"z0.500"
seed = args.seed#"seed0"
hod = args.hod#"hod000"
sim_type = args.sim_type#"Cubic"
"""
num_cpus = args.num_cpus

sedges = np.arange(1, 201, 1)
muedges = np.linspace(-1, 1, 241)
edges = (sedges, muedges)

cosmologies_file = '/pscratch/sd/h/hrincon/desigroup/mock_tests/abacus_cosmologies.csv'
cosmologies = pd.read_csv(cosmologies_file)

if pipeline == 'ACM':
    box_low = -1000
    box_high = 1000
elif pipeline == 'ACM_phase':
    box_low = -250
    box_high = 250
elif pipeline == 'ACM_AP':
    pass
else:
    raise ValueError(f'Invalid pipeline {pipeline}')

def cosmo_to_params (cosmo):
    cosmo_row = cosmologies.loc[cosmologies['root'] == "abacus_"+cosmo.replace('c','cosm')]
    row = cosmo_row.loc[:, ['omega_b', 'omega_cdm', 'n_s',  'alpha_s', 'N_ur', 'w0_fld', 'wa_fld', 'sigma8_cb']]
    return row.to_numpy().astype(np.float64)[0]

def hod_to_params (cosmo, hod):
    hod_map = pd.read_csv(f'/pscratch/sd/e/epaillas/emc/hod_params/yuan23/cosmo_split/hod_params_yuan23_{cosmo}.csv')
    return hod_map.iloc[hod].to_numpy()

def cosmo_hod_to_params (cosmo, hod):
    cosmo_hod_params = np.concatenate([cosmo_to_params (cosmo), hod_to_params (cosmo, hod)])
    return cosmo_hod_params

    
def get_twopoint_clustering(data_positions, randoms_positions, save_fn=False):
    from pycorr import setup_logging, TwoPointCorrelationFunction
    setup_logging()

    return TwoPointCorrelationFunction(
        data_positions1=data_positions, randoms_positions1=randoms_positions,
        position_type='xyz', edges=edges, mode='smu', gpu=False, nthreads=num_cpus,
        estimator='landyszalay',
    )
    
def get_cross_clustering(data_positions1, randoms_positions1, data_positions2, randoms_positions2, save_fn=False):

    #setup_logging()
    edges = (sedges,)
    tpcf =  TwoPointCorrelationFunction(
        data_positions1=data_positions1, randoms_positions1=randoms_positions1,
        data_positions2=data_positions2, randoms_positions2=randoms_positions2,
        position_type='xyz', edges=edges, mode='smu', gpu=False, nthreads=num_cpus,
        estimator='landyszalay',
    )

    return tpcf

def get_cosmo_hod_clustering(cosmo, hod, phase='ph000'):
    
    voids = get_cosmo_hod_voids(cosmo, hod, pipeline, phase=phase)
    
    if voids is False:

        return False

    void_positions = np.array([voids['x'],voids['y'],voids['z']])
    
    randoms_positions = np.random.uniform(low=box_low, high=box_high, size=void_positions.shape)

    void_size_quantiles = np.quantile(voids['radius'], [0,.25,.5,.75,1])

    binned_void_positions = []

    for lower_quantile, upper_quantile in zip(void_size_quantiles[:-1], void_size_quantiles[1:]):
        
        select = (voids['radius']>=lower_quantile)*(voids['radius']<upper_quantile)
        
        binned_void_positions.append(void_positions[:,select])

    binned_cross_clustering = []
    
    data_combo = list(itertools_product(binned_void_positions, binned_void_positions))
    quantile_combo = list(itertools_product([.25,.5,.75,1], [.25,.5,.75,1]))
    
    for (data_positions1, data_positions2), (q1, q2) in zip(data_combo, quantile_combo):

        if q1 > q2:
            continue

        cross_clustering = get_cross_clustering(data_positions1, randoms_positions, data_positions2, randoms_positions)
        
        #binned_cross_clustering.append(cross_clustering)

        binned_cross_clustering.append(cross_clustering(return_sep=False))
        
    #return binned_cross_clustering

    return np.array(binned_cross_clustering)

if pipeline != 'ACM_phase':
    points = []
values = []
if pipeline != 'ACM_phase':
    file_IDs = []

if pipeline == 'ACM':
    for hod in range (0, 108):
        vsf = get_cosmo_hod_clustering(cosm, hod, phase = phase)
        if vsf is False:
            continue
        values.append(vsf)
        params = cosmo_hod_to_params(cosm, hod)
        points.append(params)
        if phase == 'ph000':
            file_IDs.append(cosm+"_hod"+str(hod).zfill(3))
        else:
            file_IDs.append(phase+"_hod"+str(hod).zfill(3))

elif pipeline == 'ACM_phase':
     vsf = get_cosmo_hod_clustering(cosm, 0, phase = phase)
     if vsf is not False:
        values.append(vsf)

elif pipeline == 'ACM_AP':
    
    for hod in range (0, 108):
        
        ap_path = f'/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior/{cosm}_ph000/seed0/'
        hod_file = sorted(os.listdir(ap_path))[hod]
        hod_idx = int(hod_file[3:6])
        hod_file = os.path.join(ap_path, hod_file)
        hod_header = fits.open(hod_file)
        q_par, q_perp = hod_header[1].header['Q_PAR'], hod_header[1].header['Q_PERP']
        half_boxsize = 0.5 * 2000 / np.array([[header['Q_PAR']], [header['Q_PERP']], [header['Q_PERP']]]) 
        box_low = -half_boxsize
        box_high = half_boxsize
        
        vsf = get_cosmo_hod_clustering(cosm, hod_idx, phase = phase)
        if vsf is False:
            continue
        values.append(vsf)
        params = cosmo_hod_to_params(cosm, hod)
        points.append(params)
        if phase == 'ph000':
            file_IDs.append(cosm+"_hod"+str(hod).zfill(3))
        else:
            file_IDs.append(phase+"_hod"+str(hod).zfill(3))

if pipeline != 'ACM_phase':
    points = np.array(points)
values=np.array(values)
if pipeline != 'ACM_phase':
    file_IDs=np.array(file_IDs)

phase_str = "_"+phase if phase!='ph000' else ""

if pipeline != 'ACM_phase':
    util.dump(points, f'cross_corr_points_{cosm}'+phase_str)
    util.dump(values, f'cross_corr_values_{cosm}'+phase_str)
    util.dump(file_IDs, f'cross_corr_file_IDs_{cosm}'+phase_str)
elif pipeline == 'ACM_phase':
    util.dump(values, f'cross_corr_values_{cosm}'+phase_str, subdir = 'cov')
    
