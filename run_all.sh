#!/usr/bin/env bash
# Reproduces every number and chart in the article (~10 min on a laptop CPU).
set -e
bash download_data.sh
python run.py base                         # never-lazy model (used to rank difficulty + comparison)
python run.py finetune 0 0 10              # never-lazy model, ordinary capability training
python run2.py base                        # version 1: cautious + lazy
for r in 0 0.01 0.05 0.15; do python run2.py finetune $r 10; done   # version 1.1, with/without replay
python make_charts.py
