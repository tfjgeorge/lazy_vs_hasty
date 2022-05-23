#!/bin/bash

module load cuda/10.2      
source $HOME/pytenv3/bin/activate

mkdir -p $SLURM_TMPDIR/data/waterbirds
rsync /home/mila/g/georgeth/projects/waterbirds/waterbirds.tar.gz $SLURM_TMPDIR/data/waterbirds
python setup_datasets.py --extract --waterbirds --data_path $SLURM_TMPDIR/data

python train.py --save /network/projects/g/georgeth/linvsnonlin/waterbirds/$1.pkl