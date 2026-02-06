#!/bin/bash

# run this in an interactive job: bash corrfunc_alt_phase.sh


source /global/common/software/desi/desi_environment.sh main



phase_array=("ph001" "ph001" "ph002" "ph003" "ph004" "ph005" "ph006" "ph007" "ph008" "ph009" "ph010" "ph011" "ph012" "ph013" "ph014" "ph015" "ph016" "ph017" "ph018" "ph019" "ph020" "ph021" "ph022" "ph023" "ph024") 


parallel --jobs 25 python corrfunc.py -c "c000" -p {1} ::: "${phase_array[@]}"