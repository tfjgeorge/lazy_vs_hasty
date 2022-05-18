#!/bin/bash

module load cuda/10.2      
source $HOME/pytenv3/bin/activate

LR_BASE=0.01
LR=0.01
N_EPOCHS_BASE=3
MODEL=resnet18
# python train.py --lr $LR --mom 0.9 --epochs $N_EPOCHS_BASE --task cifar10_$MODEL --alpha 1 --track_accs --diff 0.15 --diff-type random --base_path /home/mila/g/georgeth/projects/linvsnonlin/cifar10_noisy/

# python train.py --lr $LR --mom 0.9 --epochs 250 --task cifar10_$MODEL --alpha 1 --track_accs --diff 0.15 --diff-type random --track_lin --fork_from /network/projects/g/georgeth/linvsnonlin/cifar10_noisy/alpha\=1.0\,batch_size\=125\,depth\=0\,diff\=0.15\,diff_type\=random\,epochs\=$N_EPOCHS_BASE\,l2\=0.0\,lr\=$LR_BASE\,mom\=0.9\,task\=cifar10_$MODEL\,track_accs\=True\,width\=0/checkpoint_10_0.pt
python train.py --lr $LR --mom 0.9 --epochs 2000 --task cifar10_$MODEL --alpha 100 --track_accs --diff 0.15 --diff-type random --track_lin --fork_from /network/projects/g/georgeth/linvsnonlin/cifar10_noisy/alpha\=1.0\,batch_size\=125\,depth\=0\,diff\=0.15\,diff_type\=random\,epochs\=$N_EPOCHS_BASE\,l2\=0.0\,lr\=$LR_BASE\,mom\=0.9\,task\=cifar10_$MODEL\,track_accs\=True\,width\=0/checkpoint_10_0.pt
