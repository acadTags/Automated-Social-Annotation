#!/bin/bash

#$ -cwd

#$ -pe gpu 1
#$ -l h_vmem=30G

. /etc/profile.d/modules.sh
module load anaconda/5.0.1
source activate tf14n

python PLST.py --dataset bibsonomy-clean --C 100 --gamma 100 --kfold -1