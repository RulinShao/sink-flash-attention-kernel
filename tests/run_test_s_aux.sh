#!/bin/bash
#SBATCH --job-name=test_s_aux
#SBATCH --output=/checkpoint/comem/rulin/ReTrain/.cache/test_s_aux_%j.log
#SBATCH --error=/checkpoint/comem/rulin/ReTrain/.cache/test_s_aux_%j.log
#SBATCH --account=comem
#SBATCH --qos=h200_comem_high
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

KERNEL_DIR=/checkpoint/comem/rulin/sink-flash-attention-kernel

echo "=== test_s_aux job on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
date

source ~/miniconda3/etc/profile.d/conda.sh
conda activate trainer_server

# Install the kernel in dev mode
cd "$KERNEL_DIR"
pip install -e . 2>&1 | tail -3

# Run the tests
python tests/test_s_aux.py

echo "=== Done ==="
date
