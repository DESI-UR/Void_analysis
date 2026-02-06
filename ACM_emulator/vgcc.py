import pandas as pd
import numpy as np
import os
from astropy.table import Table
from astropy.io import fits

import sys
sys.path.insert(1, '/global/homes/h/hrincon/python_tools')
import Util as util
from pycorr import TwoPointCorrelationFunction, setup_logging
from itertools import product as itertools_product

from read_voids import get_cosmo_hod_voids, get_cosmo_hod_galaxies

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

p = ArgumentParser(description='Configuration options for cubic box mocks.',
                   formatter_class=ArgumentDefaultsHelpFormatter)


p.add_argument('-c', '--cosmology', type=str, default='c000',
               help='Abacus simulation cosmology (e.g. c000)')
p.add_argument('-p', '--phase', type=str, default='ph000',
               help='Abacus simulation phase (e.g. ph000)')
p.add_argument('-pl', '--pipeline', type=str, default='ACM',
              help='Mock pipeline (ACM for ACM group mocks, ACM_phase for covariance)')
p.add_argument('-t', '--tracer', type=str, default='LRG',
               help='Tracer (e.g. LRG, ELG)')
p.add_argument('-z', '--redshift', type=str, default='z0.500',
               help='redshift snapshot (used for cubic sims only)')
p.add_argument('-s', '--seed', type=str, default='seed0',
               help='HOD populating random seed')
"""
p.add_argument('-d', '--hod', type=str, default='hod000',
               help='hod number (0 for best fit values, higher numbers for matched pairs')
p.add_argument('-st', '--sim_type', type=str, default='Cubic',
              help='Simulation type (Cubic or Cutsky)')
"""
p.add_argument('-nc', '--num_cpus', type=int, default=10,
              help='Number of cpus for multiprocessing. Setting to None uses one less than all available cpus')
p.add_argument('-g', '--use_gpu', type=bool, default=False,
              help='Number of cpus for multiprocessing. Setting to None uses one less than all available cpus')
args = p.parse_args()


cosm = args.cosmology#'c000'
phase = args.phase#'ph000'
pipeline = args.pipeline#'ACM'
tracer = args.tracer#"LRG"
redshift = args.redshift#"z0.500"
seed = args.seed#"seed0"
"""

hod = args.hod#"hod000"
sim_type = args.sim_type#"Cubic"
"""
num_cpus = args.num_cpus
use_gpu = args.use_gpu

if use_gpu:
    # only 4 gpus per node (num_cpus is now really num_gpus)
    num_cpus = 4

sedges = np.arange(1, 150, 1)
muedges = np.linspace(-1, 1, 241)
edges = (sedges, muedges)

cosmo_index_lookup = {'c000': 0, 'c001': 1, 'c002': 2, 'c003': 3, 'c004': 4, 'c013': 5, 'c100': 6, 'c101': 7, 'c102': 8, 'c103': 9, 'c104': 10, 'c105': 11, 'c106': 12, 'c107': 13, 'cnum_hods': 14, 'c109': 15, 'c110': 16, 'c111': 17, 'c112': 18, 'c113': 19, 'c114': 20, 'c115': 21, 'c116': 22, 'c117': 23, 'c118': 24, 'c119': 25, 'c120': 26, 'c121': 27, 'c122': 28, 'c123': 29, 'c124': 30, 'c125': 31, 'c126': 32, 'c130': 33, 'c131': 34, 'c132': 35, 'c133': 36, 'c134': 37, 'c135': 38, 'c136': 39, 'c137': 40, 'c138': 41, 'c139': 42, 'c140': 43, 'c141': 44, 'c142': 45, 'c143': 46, 'c144': 47, 'c145': 48, 'c146': 49, 'c147': 50, 'c148': 51, 'c149': 52, 'c150': 53, 'c151': 54, 'c152': 55, 'c153': 56, 'c154': 57, 'c155': 58, 'c156': 59, 'c157': 60, 'c158': 61, 'c159': 62, 'c160': 63, 'c161': 64, 'c162': 65, 'c163': 66, 'c164': 67, 'c165': 68, 'c166': 69, 'c167': 70, 'c168': 71, 'c169': 72, 'c170': 73, 'c171': 74, 'c172': 75, 'c173': 76, 'c174': 77, 'c175': 78, 'c176': 79, 'c177': 80, 'c178': 81, 'c179': 82, 'c180': 83, 'c181': 84}

num_cosmo = len(cosmo_index_lookup)
num_r_bins = 5
num_hods = 108
num_s_bins = len(sedges) - 1

cosmologies_file = '/pscratch/sd/h/hrincon/desigroup/mock_tests/abacus_cosmologies.csv'
cosmologies = pd.read_csv(cosmologies_file)

is_phase = (pipeline == 'ACM_phase' or pipeline == 'ACM_AP_phase')

if pipeline == 'ACM':
    box_low = -1000
    box_high = 1000
elif is_phase:
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

    
def get_clustering(data_positions, randoms_positions, los, save_fn=False):
    setup_logging()

    return TwoPointCorrelationFunction(
        data_positions1=data_positions, randoms_positions1=randoms_positions,
        position_type='xyz', edges=edges, mode='smu', gpu=use_gpu, nthreads=num_cpus,
        estimator='landyszalay', los=los,
    )
    
def get_cross_clustering(data_positions1, randoms_positions1, data_positions2, randoms_positions2, los, save_fn=False):
    setup_logging()
    
    tpcf =  TwoPointCorrelationFunction(
        data_positions1=data_positions1, randoms_positions1=randoms_positions1,
        data_positions2=data_positions2, randoms_positions2=randoms_positions2,
        position_type='xyz', edges=edges, mode='smu', gpu=use_gpu, nthreads=num_cpus,
        estimator='landyszalay', los = los,
    )

    return tpcf

def get_vgcc(cosmo, hod, los, phase='ph000'):
    voids = get_cosmo_hod_voids(cosmo, hod, pipeline, phase=phase)
    
    if voids is False:

        return False

    galaxies_positions = get_cosmo_hod_galaxies(cosmo, hod, seed, redshift, pipeline, phase=phase)
    randoms_positions = np.random.uniform(low=box_low, high=box_high, size=galaxies_positions.shape)
    
    r_bins = np.quantile(voids['radius'],np.linspace(0,1,num_r_bins+1))
    
    radial_ccfs = []
    
    for r_low, r_high in zip(r_bins[:-1], r_bins[1:]):
        select_voids = (voids['radius']>r_low) * (voids['radius'] <= r_high)
        void_positions = np.array([voids[select_voids]['x'],voids[select_voids]['y'],voids[select_voids]['z']])
        
        cross_clustering = get_cross_clustering(void_positions, randoms_positions, galaxies_positions, randoms_positions, los)
        
        ccf = cross_clustering(ells=(0, 2, 4), return_sep=False)
    
        radial_ccfs.append(ccf)
        
    radial_ccfs = np.array(radial_ccfs) 
    
    """
    void_positions = np.array([voids['x'],voids['y'],voids['z']])
    cross_clustering = get_cross_clustering(void_positions, randoms_positions, galaxies_positions, randoms_positions, los)
    radial_ccfs = cross_clustering(ells=(0, 2, 4), return_sep=False)
    """

    return radial_ccfs

"""
if pipeline == 'ACM_AP':
    # load savefile for cosmological/HOD paramters
    if not os.path.exists('/global/homes/h/hrincon/ACM/vgcc_x.array'):
        # shape = (num cosmo, num HOD, num emulator grid parameters)
        vgcc_array_x = np.memmap('vgcc_x.array', dtype='float32', mode='w+', shape=(num_cosmo, num_hods, 20))
        vgcc_array_x[:] = 9999.0
        vgcc_array_x.flush()
    else:
        vgcc_array_x = np.memmap('vgcc_x.array', dtype='float32', mode='r+', shape=(num_cosmo, num_hods, 20))
    
    # load savefile for vgcc
    if not os.path.exists('/global/homes/h/hrincon/ACM/vgcc_y.array'):
        # shape = (num cosmo, num HOD, num radial bins, num multipoles, num s bins)
        vgcc_array_y = np.memmap('vgcc_y.array', dtype='float32', mode='w+', shape=(num_cosmo, num_hods, num_r_bins, 3, num_s_bins))
        vgcc_array_y[:] = 9999.0
        vgcc_array_y.flush()
    else:
        vgcc_array_y = np.memmap('vgcc_y.array', dtype='float32', mode='r+', shape=(num_cosmo, num_hods, num_r_bins, 3, num_s_bins))
"""

if not is_phase:
    points = []
values = []
if not is_phase:
    file_IDs = []

if pipeline == 'ACM':
    for hod in range (0, num_hods):
        statistic = get_vgcc(cosm, hod, 'x', phase = phase)
        if statistic is False:
            continue
        values.append(statistic)
        params = cosmo_hod_to_params(cosm, hod)
        points.append(params)
        if phase == 'ph000':
            file_IDs.append(cosm+"_hod"+str(hod).zfill(3))
        else:
            file_IDs.append(phase+"_hod"+str(hod).zfill(3))

elif is_phase:
     statistic = get_vgcc(cosm, 0, 'x', phase = phase)
     if statistic is not False:
        values.append(statistic)

elif pipeline == 'ACM_AP':
    
    for hod in range (0, num_hods):

        print(f'Starting HOD: {hod} | cosm:{cosm}')
        
        ap_path = f'/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior/{cosm}_ph000/seed0/'
        hod_file = sorted(os.listdir(ap_path))[hod]
        hod_idx = int(hod_file[3:6])
        hod_file = os.path.join(ap_path, hod_file)
        hod_header = fits.open(hod_file)
        q_par, q_perp = hod_header[1].header['Q_PAR'], hod_header[1].header['Q_PERP']
        half_boxsize = 0.5 * 2000 / np.array([[q_par], [q_perp], [q_perp]]) 
        box_low = -half_boxsize
        box_high = half_boxsize

        statistic = get_vgcc(cosm, hod_idx, 'x', phase = phase)
        if statistic is False:
            continue
        #cosm_out_idx = cosmo_index_lookup[cosm]
        #vgcc_array_y[cosm_out_idx, hod] = statistic
        values.append(statistic)
        params = cosmo_hod_to_params(cosm, hod_idx)
        #vgcc_array_x[cosm_out_idx, hod] = params
        points.append(params)
        if phase == 'ph000':
            file_IDs.append(cosm+"_hod"+str(hod_idx).zfill(3))
        else:
            file_IDs.append(phase+"_hod"+str(hod_idx).zfill(3))
    #vgcc_array_y.flush()
    #vgcc_array_x.flush()

if not is_phase:
    points = np.array(points)
values=np.array(values)
if not is_phase:
    file_IDs=np.array(file_IDs)

phase_str = "_"+phase if phase!='ph000' else ""

if not is_phase:
    util.dump(points, pipeline+f'_vgcc_points_{cosm}'+phase_str)
    util.dump(values, pipeline+f'_vgcc_values_{cosm}'+phase_str)
    util.dump(file_IDs, pipeline+f'_vgcc_file_IDs_{cosm}'+phase_str)
else:
    if pipeline == 'ACM_phase':
        subdir = 'cov'
    elif pipeline == 'ACM_AP_phase':
        subdir = 'cov_AP'
    else:
        raise ValueError('Invalid pipeline for phase mode:',pipeline)
    util.dump(values, f'vgcc_values_{cosm}'+phase_str, subdir = subdir)
    
