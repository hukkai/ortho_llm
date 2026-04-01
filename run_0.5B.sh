#!/usr/bin/env bash
export MASTER_PORT=$((12000 + RANDOM % 20000))

ORTH_TYPE=${1:-all}
SUB_MATRIX=${2:-1}

LR=${3:-2e-4}
MIN_LR=${4:-2e-5}
SO_LR=${5:-0.5}

OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node 8 \
    --master_port "${MASTER_PORT}" \
    train.py \
    --data-dir ./data/C4 \
    --num-layers 18 \
    --hidden-size 1536 \
    --num-heads 24 \
    --batch-size 4 \
    --global-batch-size 512 \
    --seq-length 2048 \
    --lr $LR \
    --min-lr $MIN_LR \
    --so-lr $SO_LR \
    --num-steps 10_000 \
    --orthogonal-type "${ORTH_TYPE}" \
    --sub-matrix "${SUB_MATRIX}"
