#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gs-container_g1.24h
#$ -ac d=aip-cuda-12.0.1-blender-2,d_shm=64G
#$ -N ChangeMamba-BCD_SYSU

. ~/net.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/chenhrx/anaconda3/envs/anydisaster/bin/python}"
PROJECT_ROOT='/home/chenhrx/project/ChangeMamba'
DATA_ROOT='/home/chenhrx/datasets/SYSU-CD'
LIST_ROOT="$PROJECT_ROOT/changedetection/datasets/data_name_list/SYSU-CD"
CFG_PATH="$PROJECT_ROOT/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"

${PYTHON_BIN} ${PROJECT_ROOT}/changedetection/script/train_MambaBCD.py \
    --cfg ${CFG_PATH} \
    --dataset 'SYSU' \
    --model_type 'ChangeMamba-BCD' \
    --model_param_path ${PROJECT_ROOT}/results/checkpoints \
    --train_dataset_path ${DATA_ROOT}/train \
    --train_data_list_path ${LIST_ROOT}/train_set.txt \
    --test_dataset_path ${DATA_ROOT}/test \
    --test_data_list_path ${LIST_ROOT}/test_set.txt \
    --pretrained_weight_path ${PROJECT_ROOT}/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth \
    --batch_size 16 \
    --crop_size 256 \
    --learning_rate 1e-4 \
    --weight_decay 5e-3 \
    --max_iters 20000
