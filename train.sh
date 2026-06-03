#!/bin/bash
# 训练 Instinct-Parkour-Target-Amp-G1-v0 任务 约需要78h, 3.2天

# pip install -e /home/tcict/zzq/Instinct/InstinctLab_backup/source/instinctlab --no-deps -q
python scripts/instinct_rl/train.py \
    --task=Instinct-Parkour-Target-Amp-G1-v0 \
    --headless

# python scripts/instinct_rl/train.py \
#     --task=Instinct-Parkour-Target-Amp-G1-v0 \
#     --resume \
#     --num_envs=1 \
#     --load_run=/home/tcict/zzq/Instinct/InstinctLab/logs/instinct_rl/g1_parkour/20260410_222025 \
#     --checkpoint=model_30000.pt
