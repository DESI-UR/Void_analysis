#!/bin/bash
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH -t 1:30:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH -c 256
#SBATCH --array=1-24

# select above from 1-24
# 0 done seperately in main DESIVoSS script

source /global/common/software/desi/desi_environment.sh main

# 36 threads, each will take a max of 9 hods to reach 324 hods total
# Start though taking 3 hods to reach 108 total
# 10 GB per thread to reach 360 GB total
# Each thread used 7 cpus (w/ 6 cpus given to find_voids) to reach 252
# 12 minutes to make a void catalog (rough guess, can be much longer if high # of tracers in HOD/cosmo)

printf -v task_padded "%03g" $SLURM_ARRAY_TASK_ID
phase="ph${task_padded}"

min_hod=0 
#324
max_hod=107

hod_array=() 

for hod in $(seq $min_hod $max_hod); do 
  printf -v hod_padded "%03g" $hod
  hod_array+=("hod${hod_padded}")
done

parallel --jobs 36 python DESIVoSS.py -c "c000" -p "$phase" -d {1} ::: "${hod_array[@]}"
