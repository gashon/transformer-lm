#!/bin/bash
#SBATCH --job-name=profile_bpe_owt
#SBATCH --partition=batch-cpu
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=100G
#SBATCH --time=5:35:00
#SBATCH --output=profile_bpe_owt_%j.out
#SBATCH --error=profile_bpe_owt_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate transformer_lm
python3 -m memory_profiler perf/bpe/owt.py | tee data/log/output-owt-$(date +%s).log
