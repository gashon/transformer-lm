# Transformer LM Implementation

## Setup

0. Set up a conda environment and install packages:

```sh
conda create -n transformer_lm python=3.10 --yes
conda activate transformer_lm
pip install -e .'[test]'
```

1. Unit tests:

```sh
pytest
```

2. Profiling:

```sh
python3 -m memory_profiler profile/bpe.py
```

or to submit a job to a cluster:

```sh
sbatch profile/bpe.sbatch
```
