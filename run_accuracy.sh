#!/bin/bash
#SBATCH --account=dream
#SBATCH --qos=h200_dream_high
#SBATCH --gpus=1
#SBATCH --mem=100g
#SBATCH --time=0:10:00
#SBATCH --output=/home/rulin/sink_attention/slurm_accuracy_%j.out
#SBATCH --error=/home/rulin/sink_attention/slurm_accuracy_%j.err

cd /home/rulin/sink_attention
python numerical_accuracy.py
