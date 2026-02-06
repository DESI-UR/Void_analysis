#!/bin/bash
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH -t 1:30:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH -c 256
#SBATCH --array=107-126,130-181

# select above from 0-4,13,100-126,130-181
# 000 done seperately in sbash

source /global/common/software/desi/desi_environment.sh main

# 36 threads, each will take a max of 9 hods to reach 324 hods total
# Start though taking 3 hods to reach 108 total
# 10 GB per thread to reach 360 GB total
# Each thread used 7 cpus (w/ 6 cpus given to find_voids) to reach 252
# 12 minutes to make a void catalog (rough guess, can be much longer if high # of tracers in HOD/cosmo)

# for c000
#cosm="c000"

# for alt cosmo
printf -v task_padded "%03g" $SLURM_ARRAY_TASK_ID
cosm="c${task_padded}"

files=(/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior/${cosm}_ph000/seed0/*)


min_hod=0 
#324 107
max_hod=107

hod_array=() 

#for hod in $(seq $min_hod $max_hod); do 
for (( hod=$max_hod; hod>=$min_hod; hod-- )); do
  file=${files[hod]}
  file=${file##*/}
  file=${file%.*} 
  hod_array+=("${file}")
done
#echo "${hod_array[@]}"
parallel --jobs 36 python DESIVoSS.py -c "$cosm" -p "ph000" -d {1}  -pl "ACM_AP" ::: "${hod_array[@]}"