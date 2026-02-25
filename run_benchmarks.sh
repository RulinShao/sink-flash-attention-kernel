#!/bin/bash
#SBATCH --account=dream
#SBATCH --qos=h200_dream_high
#SBATCH --gpus=1
#SBATCH --mem=100g
#SBATCH --time=0-00:30:00
#SBATCH --output=/home/rulin/sink_attention/benchmark_results.log
#SBATCH --error=/home/rulin/sink_attention/benchmark_errors.log

cd /home/rulin/sink_attention
python tests/run_inference_benchmarks.py
