#!/bin/bash
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --array=0-4,13,100-126,130-181

# select above from 0-4,13,100-126,130-181

# for alt cosmo
printf -v task_padded "%03g" $SLURM_ARRAY_TASK_ID
cosm="c${task_padded}"


source /global/common/software/desi/desi_environment.sh main

python vgcc.py -c $cosm -pl ACM_AP -g True