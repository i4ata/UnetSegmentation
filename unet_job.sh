#!/bin/bash
#SBATCH --time=00:50:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=unet_run
#SBATCH --mem=8000
#SBARCH --output=training.log

module purge
module load Python/3.9.6-GCCcore-11.2.0

source ~/env/bin/activate

python custom_unet.py
