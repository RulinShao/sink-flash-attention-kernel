#!/bin/bash
#SBATCH --account=dream
#SBATCH --qos=h200_dream_high
#SBATCH --gpus=1
#SBATCH --mem=100g
#SBATCH --time=0:20:00
#SBATCH --output=/home/rulin/sink_attention/slurm_tune_%j.out
#SBATCH --error=/home/rulin/sink_attention/slurm_tune_%j.err

cd /home/rulin/sink_attention
python tune_block_sizes.py
