# Transformer LM Implementation

## Setup

0. Set up a conda environment and install packages:

```sh
conda create -n transformer_lm python=3.10 --yes
conda activate transformer_lm
pip install -e .'[test]'
```

### Unit tests:

```sh
pytest
```

### Profiling:

```sh
python3 -m memory_profiler perf/bpe/owt.py
```

or to submit a job to a slurm cluster:

```sh
sbatch perf/bpe/owt.sbatch
```
