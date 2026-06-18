#!/bin/bash
# 训练 Instinct-Parkour-Target-Amp-G1-v0 任务 约需要78h, 3.2天

# pip install -e /home/tcict/zzq/Instinct/InstinctLab_backup/source/instinctlab --no-deps -q
# python scripts/instinct_rl/train.py \
#     --task=Instinct-Parkour-Target-Amp-G1-v0 \
#     --num_envs=4096 \
#     --headless

python scripts/instinct_rl/train.py \
    --task=Instinct-Parkour-Target-Amp-G1-v0 \
    --resume \
    --headless \
    --num_envs=4096 \
    --load_run=/home/tcict/zzq/Instinct/InstinctLab/logs/instinct_rl/g1_parkour/20260605_110544 \
    --checkpoint=model_30000.pt
