#!/bin/bash
#SBATCH --account=dream
#SBATCH --qos=h200_dream_high
#SBATCH --gpus=1
#SBATCH --mem=100g
#SBATCH --time=0:30:00
#SBATCH --output=/home/rulin/sink_attention/slurm_bench_%j.out
#SBATCH --error=/home/rulin/sink_attention/slurm_bench_%j.err

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home/rulin/sink_attention
pip install -e . --quiet 2>/dev/null
python tests/benchmark.py
