#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=unet
#SBATCH --mem=8000

source ~/env/bin/activate .

python -m src.evaluate

