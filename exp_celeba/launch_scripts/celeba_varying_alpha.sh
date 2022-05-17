#!/bin/bash

module load cuda/10.2      
source $HOME/pytenv3/bin/activate

mkdir -p $SLURM_TMPDIR/data/celeba
rsync /home/mila/g/georgeth/projects/celeba/* $SLURM_TMPDIR/data/celeba
python setup_datasets.py --download --data_path $SLURM_TMPDIR/data

python train.py --save /network/projects/g/georgeth/linvsnonlin/celeba/recorder_longer5.pkl