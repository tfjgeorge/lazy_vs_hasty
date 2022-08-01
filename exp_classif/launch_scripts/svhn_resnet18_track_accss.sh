#!/bin/bash

module load cuda/10.2      
source $HOME/pytenv3/bin/activate

MODEL=resnet18
BASE_EPOCHS=3
# python train.py --lr 0.01 --mom 0.9 --epochs $BASE_EPOCHS --task svhn_$MODEL --alpha 1 --base_path $SCRATCH/linvsnonlin/svhn/

# python train.py --lr 0.01 --mom 0.9 --epochs 200 --task svhn_$MODEL --alpha 1 --track_all_accs --track_lin --fork_from $SCRATCH/linvsnonlin/svhn/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=$BASE_EPOCHS,l2=0.0,lr=0.01,mom=0.9,task=svhn_$MODEL,width=0/checkpoint_0_0.pt
python train.py --lr 0.01 --mom 0.9 --epochs 2500 --task svhn_$MODEL --alpha 100 --track_all_accs --track_lin --fork_from $SCRATCH/linvsnonlin/svhn/alpha=1.0,batch_size=125,depth=0,diff=0.0,diff_type=random,epochs=$BASE_EPOCHS,l2=0.0,lr=0.01,mom=0.9,task=svhn_$MODEL,width=0/checkpoint_0_0.pt