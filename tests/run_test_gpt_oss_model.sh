#!/bin/bash
#SBATCH --job-name=test_gptoss_sink
#SBATCH --account=comem
#SBATCH --qos=h200_comem_high
#SBATCH --partition=h200
#SBATCH --gpus-per-node=1
#SBATCH --mem=200g
#SBATCH --time=0-01:00:00
#SBATCH --output=/checkpoint/comem/rulin/ReTrain/.cache/test_gptoss_sink_%j.log
#SBATCH --error=/checkpoint/comem/rulin/ReTrain/.cache/test_gptoss_sink_%j.err

set -eo pipefail

echo "=== gpt-oss sink kernel model test on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trainer_server

# Navigate to the kernel directory
cd /checkpoint/comem/rulin/sink-flash-attention-kernel

# Install the kernel
pip install .

# Run model-level comparison: eager vs kernel vs FA2
python tests/test_gpt_oss_model.py --seq-len 512 --num-tokens 5

echo "Test finished: $(date)"


