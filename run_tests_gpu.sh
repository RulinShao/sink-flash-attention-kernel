#!/bin/bash
#SBATCH --account=dream
#SBATCH --qos=h200_dream_high
#SBATCH --gpus=1
#SBATCH --mem=100g
#SBATCH --time=0-00:15:00
#SBATCH --output=/home/rulin/sink_attention/test_results.log
#SBATCH --error=/home/rulin/sink_attention/test_errors.log

cd /home/rulin/sink_attention
export PYTHONPATH=/home/rulin/sink_attention:$PYTHONPATH

echo "=== Running test_cache.py ==="
python tests/test_cache.py
echo ""
echo "=== Running test_inference.py ==="
python tests/test_inference.py
echo ""
echo "=== Running existing test_sink_attention.py ==="
python tests/test_sink_attention.py
