#!/bin/bash

#$ -cwd

#$ -pe gpu-titanx 1
#$ -l h_vmem=30G

. /etc/profile.d/modules.sh
module load anaconda/5.0.1
source activate tf14n

python CC.py --dataset bibsonomy-clean --C 100 --gamma 100 --kfold -1 --marking_id CC