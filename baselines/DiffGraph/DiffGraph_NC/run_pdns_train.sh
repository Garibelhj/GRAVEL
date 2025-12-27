#!/bin/bash
# Training script for PDNS dataset with DiffGraph

echo "=========================================="
echo "Training DiffGraph on PDNS Dataset"
echo "Binary Classification: Benign vs Malicious"
echo "=========================================="

python main.py \
    --data pdns \
    --lr 3e-3 \
    --batch 256 \
    --epoch 100 \
    --steps 200 \
    --noise_scale 1e-5 \
    --gpu 0

echo "Training completed!"

