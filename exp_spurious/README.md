Prepare Celeb A dataset

SLURM_TMPDIR=/Tmp/slurm.1852024.0
mkdir -p $SLURM_TMPDIR/data/celeba
rsync /home/mila/g/georgeth/projects/celeba/* $SLURM_TMPDIR/data/celeba
python setup_datasets.py --extract --celeba --data_path $SLURM_TMPDIR/data

X_NODES=kepler3,kepler4,kepler5,rtx5,rtx4,rtx1,mila01,mila02,mila03,rtx7
sbatch --gres=gpu:1 -c 4 -x $X_NODES -t 5:00:00 --partition=main launch_scripts/celeba_varying_alpha.sh

Prepare Waterbirds dataset

SLURM_TMPDIR=/Tmp/slurm.1849021.0
mkdir -p $SLURM_TMPDIR/data/waterbirds
rsync /home/mila/g/georgeth/projects/waterbirds/waterbirds.tar.gz $SLURM_TMPDIR/data/waterbirds
python setup_datasets.py --extract --waterbirds --data_path $SLURM_TMPDIR/data
