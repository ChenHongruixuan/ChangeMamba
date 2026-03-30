#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gs-container_g1_dev
#$ -ac d=aip-cuda-12.0.1-blender-2,d_shm=64G
#$ -N ChangeMamba-MMBDA_BRIGHT

. ~/net.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/chenhrx/anaconda3/envs/anydisaster/bin/python}"
PROJECT_ROOT='/home/chenhrx/project/ChangeMamba'
DATA_ROOT='/data/ggeoinfo/datasets/BRIGHT'
LIST_ROOT="$PROJECT_ROOT/changedetection/datasets/data_name_list/BRIGHT"
CFG_PATH="$PROJECT_ROOT/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"

${PYTHON_BIN} ${PROJECT_ROOT}/changedetection/script/train_MambaBDA_bright.py \
    --cfg ${CFG_PATH} \
    --dataset 'BRIGHT' \
    --model_type 'ChangeMamba-MMBDA' \
    --model_param_path ${PROJECT_ROOT}/results/checkpoints \
    --train_dataset_path ${DATA_ROOT} \
    --train_data_list_path ${LIST_ROOT}/train_set.txt \
    --val_dataset_path ${DATA_ROOT} \
    --val_data_list_path ${LIST_ROOT}/val_set.txt \
    --test_dataset_path ${DATA_ROOT} \
    --test_data_list_path ${LIST_ROOT}/test_set.txt \
    --pretrained_weight_path ${PROJECT_ROOT}/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth \
    --train_batch_size 8 \
    --num_workers 8 \
    --crop_size 512 \
    --learning_rate 1e-4 \
    --weight_decay 5e-3 \
    --max_iters 80000
