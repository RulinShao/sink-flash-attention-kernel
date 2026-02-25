#!/bin/bash
#SBATCH --account=dream
#SBATCH --qos=h200_dream_high
#SBATCH --gpus=1
#SBATCH --mem=100g
#SBATCH --time=0:15:00
#SBATCH --output=/home/rulin/sink_attention/slurm_verl_patch_%j.out
#SBATCH --error=/home/rulin/sink_attention/slurm_verl_patch_%j.err

cd /home/rulin/sink_attention
python test_verl_patch.py
