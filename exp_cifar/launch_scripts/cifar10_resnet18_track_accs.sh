#!/bin/bash

module load cuda/10.2      
source $HOME/pytenv3/bin/activate

# python train.py --lr 0.02 --mom 0 --epochs 200 --task cifar10_resnet18 --alpha 1 --track_accs
python train.py --lr 0.02 --mom 0 --epochs 10000 --task cifar10_resnet18 --alpha 10000 --track_accs --fork_from results/alpha\=1.0\,batch_size\=125\,depth\=0\,diff\=0.0\,diff_type\=random\,epochs\=200\,l2\=0.0\,lr\=0.02\,mom\=0.0\,task\=cifar10_resnet18\,track_accs\=True\,width\=0/checkpoint_20_400.pt