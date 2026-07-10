#!/bin/bash

# 使用 parkour_onboard_preview_stair 模型
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py \
    --task=Instinct-Parkour-Target-Amp-G1-Play-v0 \
    --load_run=/home/tcict/zzq/Instinct/data_model/checkpoints/parkour_onboard_preview_stair \
    --useonnx \
    --num_envs=1 \
    --keyboard_control

# pip install -e /home/tcict/zzq/Instinct/InstinctLab_backup/source/instinctlab --no-deps -q
# python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py \
#     --task=Instinct-Parkour-Target-Amp-G1-Play-v0 \
#     --load_run=/home/tcict/zzq/Instinct/InstinctLab/logs/instinct_rl/g1_parkour/20260625_001711 \
#     --checkpoint model_30000.pt \
#     --num_envs=1 \
#     --keyboard_control \
#     --useonnx
# 开启keyboard控制后，num_envs只能为1,且无法使用onnx导出模型

# python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py \
#     --task=Instinct-Parkour-Stairs-Finetune-G1-v0 \
#     --load_run=/home/tcict/zzq/Instinct/InstinctLab/logs/instinct_rl/stairs_finetune/20260424_201015_from20260410_222025 \
#     --checkpoint model_45000.pt \
#     --num_envs=1 \
#     --keyboard_control \
#     --exportonnx