#!/bin/bash
#SBATCH --account=dream
#SBATCH --qos=h200_dream_high
#SBATCH --gpus=1
#SBATCH --mem=100g
#SBATCH --time=0:30:00
#SBATCH --output=/home/rulin/sink_attention/slurm_test_%j.out
#SBATCH --error=/home/rulin/sink_attention/slurm_test_%j.err

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Python/Triton versions ==="
python -c "import torch; import triton; print(f'torch={torch.__version__}, triton={triton.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0)}')"

echo "=== Running smoke tests ==="
cd /home/rulin/sink_attention
python test_sink_attention.py

echo "=== Done ==="
