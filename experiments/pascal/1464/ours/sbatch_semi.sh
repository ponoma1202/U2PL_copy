#!/bin/bash
# **************************
# MODIFY THESE OPTIONS

#SBATCH --partition=isg
#SBATCH --nodelist=quebec
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=u2pl_semi
#SBATCH -o log-%N.%j.out
#SBATCH --time=64:00:00

source #TODO: replace with path to your conda.sh file here
conda activate u2pl

# TODO: specify path to train_sup.py file and config file
python train_semi.py --config=config.yaml --seed 2  --output_dirpath=./out

