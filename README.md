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

### Example generation

TinyStoriesV1 dataset on 3 H100 hrs \\

`Prompt: "Once upon a time,"` \\

```
"Once upon a time, there was a little girl named Lily. She had a big, round ball that she loved to play with. One day, Lily saw a big, round ball in the park. She wanted to play with it, but she was too small to reach it.
Lily asked her friend, Tom, for help. Tom said, "Let's play with the ball together!" Lily agreed, and they started to play. They played with the ball and had lots of fun. They were very happy.
After playing, Lily and Tom were tired. They sat down and took a nap. They talked and laughed together. They became good friends. And they lived happily ever after.
<|endoftext|>"
```
