import numpy as np
import vast.catalog.void_slice_plots as vsp
from vast.voidfinder import find_voids, wall_field_separation
from vast.voidfinder.preprocessing import load_data_to_Table
import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

p = ArgumentParser(description='Configuration options for cubic box mocks.',
                   formatter_class=ArgumentDefaultsHelpFormatter)


p.add_argument('-c', '--cosmology', type=str, default='c000',
               help='Abacus simulation cosmology (e.g. c000)')
p.add_argument('-p', '--phase', type=str, default='ph000',
               help='Abacus simulation phase (e.g. ph000)')
p.add_argument('-t', '--tracer', type=str, default='LRG',
               help='Tracer (e.g. LRG, ELG)')
p.add_argument('-z', '--redshift', type=str, default='z0.500',
               help='redshift snapshot (used for cubic sims only)')
p.add_argument('-s', '--seed', type=str, default='seed0',
               help='HOD populating random seed')
p.add_argument('-d', '--hod', type=str, default='hod000',
               help='hod number (0 for best fit values, higher numbers for matched pairs')
p.add_argument('-pl', '--pipeline', type=str, default='ACM',
              help='Mock pipeline (ACM for ACM group mocks, DCM for DESI Cutsky Mocks)')
p.add_argument('-st', '--sim_type', type=str, default='Cubic',
              help='Simulation type (Cubic or Cutsky)')
p.add_argument('-nc', '--num_cpus', type=int, default=6,
              help='Number of cpus for multiprocessing. Setting to None uses one less than all available cpus')
args = p.parse_args()

cosm = args.cosmology#'c000'
phase = args.phase#'ph000'
tracer = args.tracer#"LRG"
redshift = args.redshift#"z0.500"
seed = args.seed#"seed0"
hod = args.hod#"hod000"
pipeline = args.pipeline#"ACM"
sim_type = args.sim_type#"Cubic"
num_cpus = args.num_cpus

# for number density cut
################################################################################
# User inputs
#-------------------------------------------------------------------------------

print('starting', cosm, phase, hod)

if pipeline == "ACM_AP": 
    path=f'/pscratch/sd/h/hrincon/desigroup/VoidFinder/ACM_AP/Cubic/LRG/{cosm}/z0.500/Abacus_{hod}_VoidFinder_Output.fits'
    if os.path.exists(path):
        print('Catalog exists. Exiting.')
        sys.exit()

is_phase = (pipeline == "ACM_phase"  or pipeline == 'ACM_AP_phase')

# "Survey" name - this will be used as the prefix for all output files
survey_name = f"Abacus_{phase}_" if is_phase else f"Abacus_{hod}_"

# Change this directory paths to where you want the output to be saved.
out_directory = f'/pscratch/sd/h/hrincon/desigroup/VoidFinder/{pipeline}/{sim_type}/{tracer}/{cosm}/'
if is_phase: out_directory += f'covariance/'
elif phase != "ph000": out_directory += f'{phase}/'
    
if seed != "seed0": out_directory += f'{seed}/'
if sim_type == "Cubic":  out_directory += f'{redshift}/'
os.makedirs(os.path.dirname(out_directory), exist_ok=True)
# Coordinate limits of the simulation (in Mpc)

box_size = 500. if pipeline == 'ACM_phase' or pipeline == 'ACM_AP_phase' else 2000. # Mpc/h 2000
corner = -250. if pipeline == 'ACM_phase' or pipeline == 'ACM_AP_phase' else -1000.
xyz_limits = np.array([[corner,corner,corner],[corner+box_size,corner+box_size,corner+box_size]])

# Size of a single grid cell
hole_grid_edge_length = 5.0

################################################################################

print("Reading data")
if sim_type == "Cubic":
    if pipeline == "DCM": # DESI Cutsky Mock
        import pandas as pd
        wall_coords_xyz = pd.read_csv(f'/pscratch/sd/h/hrincon/desigroup/mock_tests/ACMmocks/Cubic/{tracer}/AbacusSummit_base_{cosm}_{phase}/{redshift}/galaxies/{tracer}s.dat',
                              comment='#', 
                              sep="\s+", 
                                  usecols = [0, 1, 2]
                                 )
        wall_coords_xyz = wall_coords_xyz.to_numpy()

        socket_path=f"/tmp/voidfinder_{hod}.sock"
    
    elif pipeline == "ACM":
        import fitsio
        redshift = redshift[:4] #remove trailing 0s
        
        wall_coords_xyz = fitsio.read(f'/pscratch/sd/e/epaillas/emc/hods/cosmo+hod/{redshift}/yuan23_prior/{cosm}_{phase}/{seed}/{hod}.fits', columns=['X', 'Y', 'Z'])
        wall_coords_xyz = np.array([wall_coords_xyz['X'], wall_coords_xyz['Y'], wall_coords_xyz['Z']]).T

        socket_path=f"/tmp/voidfinder_{hod}.sock"
        
    elif pipeline == "ACM_phase":
        import fitsio
        wall_coords_xyz = fitsio.read(f'/pscratch/sd/e/epaillas/emc/hods/z0.5/yuan23_prior/small/hod466/{phase}_hod466.fits', columns=['X', 'Y', 'Z'])
        wall_coords_xyz = np.array([wall_coords_xyz['X'], wall_coords_xyz['Y'], wall_coords_xyz['Z']]).T

        socket_path=f"/tmp/voidfinder_{phase}.sock"

    elif pipeline == "ACM_AP":
        import fitsio
        redshift = redshift[:4] #remove trailing 0s
        wall_coords_xyz, header = fitsio.read(f'/pscratch/sd/n/ntbfin/emulator/hods/{redshift}/yuan23_prior/{cosm}_{phase}/{seed}/{hod}.fits', 
                                      columns=['X_RSD', 'Y_PERP', 'Z_PERP'], header=True)
        wall_coords_xyz = np.array([wall_coords_xyz['X_RSD'], wall_coords_xyz['Y_PERP'], wall_coords_xyz['Z_PERP']]).T

        half_boxsize = 0.5 * 2000 / np.array([header['Q_PAR'], header['Q_PERP'], header['Q_PERP']])  # for first column as LOS
        xyz_limits = np.array([-half_boxsize, half_boxsize])

        socket_path=f"/tmp/voidfinder_{hod}.sock"
        
    elif pipeline == "ACM_AP_phase":
        
        from cosmoprimo.fiducial import AbacusSummit
        fid_cosmo = AbacusSummit(0)
        hubble = 100 * fid_cosmo.efunc(float(redshift[1:]))
        scale_factor = 1 / (1 + float(redshift[1:]))

        import fitsio
        redshift = redshift[:4] #remove trailing 0s
        wall_coords = fitsio.read(f'/pscratch/sd/e/epaillas/emc/hods/{redshift}/yuan23_prior/small/hod466/{phase}_hod466.fits', columns=['X', 'Y', 'Z', 'VX', 'VY', 'VZ'])
        
        x_coord = wall_coords['X'] + wall_coords['VX'] / (hubble * scale_factor) # RSD in x direction
        x_coord = ((x_coord - corner) % box_size ) + corner # periodic wrapping
        wall_coords_xyz = np.array([x_coord, wall_coords['Y'], wall_coords['Z']]).T

        socket_path=f"/tmp/voidfinder_{phase}.sock"        
    else:
        raise ValueError (f'"{pipeline}" is not a valid pipeline')
else:
    raise ValueError (f'"{sim_type}" is not a valid simulation type')

# get coords in xyz limits (coords should already be in xyz limits for all pipeliens so far added)
#wall_coords_xyz = wall_coords_xyz[np.all((wall_coords_xyz > xyz_limits[0])*(wall_coords_xyz < xyz_limits[1]), axis=1)]
print(wall_coords_xyz.shape)

print("Finding voids")
find_voids(wall_coords_xyz,
           survey_name,
           out_directory,
           mask_type='xyz',
           xyz_limits=xyz_limits,
           #save_after=50000,
           #use_start_checkpoint=True,
           num_cpus=num_cpus,
           batch_size=10000,
           verbose=1,
           save_missing_galaxies=False,
           maximal_spheres_only=True,
          SOCKET_PATH=socket_path);