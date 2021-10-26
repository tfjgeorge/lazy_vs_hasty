#!/bin/bash

module load cuda/10.2      
source $HOME/pytenv3/bin/activate

python train.py --lr 0.02 --mom 0 --epochs 200 --task cifar10_resnet18 --alpha 1 --track_accs