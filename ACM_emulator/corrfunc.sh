#!/bin/bash

# run this in an interactive job: bash corrfunc.sh

source /global/common/software/desi/desi_environment.sh main

#cosm_array=("c100" "c101" "c102" "c103" "c104" "c105" "c106" "c107" "c108" "c109" "c110" "c111" "c112" "c113" "c114" "c115" "c116" "c117" "c118" "c119" "c120" "c121" "c122" "c123" "c124" "c125" "c126" "c130" "c131" "c132" "c133" "c134" "c135" "c136" "c137" "c138" "c139" "c140" "c141" "c142" "c143" "c144" "c145" "c146" "c147" "c148" "c149" "c150" "c151" "c152" "c153" "c154" "c155" "c156" "c157" "c158" "c159" "c160" "c161" "c162" "c163" "c164" "c165" "c166" "c167" "c168" "c169" "c170" "c171" "c172" "c173" "c174" "c175" "c176" "c177" "c178" "c179" "c180" "c181") 


parallel --jobs 25 python corrfunc.py -c {1} ::: "${cosm_array[@]}"