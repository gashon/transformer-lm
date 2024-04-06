#!/bin/bash
#SBATCH --job-name=profile_bpe_corpus
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=0:10:00
#SBATCH --output=profile_bpe_corpus_%j.out
#SBATCH --error=profile_bpe_corpus_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate transformer_lm
python3 -m memory_profiler perf/bpe_corpus.py | tee data/log/output-corpus-$(date +%s).log
