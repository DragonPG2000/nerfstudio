#!/bin/bash
#SBATCH --job-name=hyperspectral-splatfacto
#SBATCH --output=baseline.out
#SBATCH --error=baseline.err
#SBATCH --partition="hays-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node="a40:2"
#SBATCH --qos="short"
#SBATCH --exclude="clippy"
#SBATCH --mem-per-gpu="40"

export PYTHONUNBUFFERED=TRUE
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
source ~/.bashrc
conda init
conda activate hyper_3d

cd /nethome/skumar704/flash/hyperspectral_3d/nerfstudio
wandb login 6ea385b99c7c031e9e9f9a735469b234c40b6039

srun -u ns-train hyperspectral-splatfacto --data /nethome/skumar704/flash/FAKEPLANT1_2 --vis wandb
echo "done"