#!/bin/bash

python export_onnx.py \
    --task=Instinct-Parkour-Target-Amp-G1-Play-v0 \
    --load_run=/home/tcict/zzq/Instinct/InstinctLab/logs/instinct_rl/g1_parkour/20260615_221504_from20260605_110544 \
    --checkpoint model_40500.pt \
    --num_envs=1
