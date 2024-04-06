#!/bin/bash

conda activate transformer_lm
python3 -m memory_profiler profile.py | tee data/log/output-$(date +%s).log
