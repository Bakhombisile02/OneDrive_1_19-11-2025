#!/bin/bash
set -e

echo "Running smoke test..."

# Run with minimal folds and features for speed
python3 assignment.py \
    --flow siya \
    --siya-device cpu \
    --siya-outer-folds 2 \
    --siya-inner-folds 2 \
    --siya-k 5

echo "Smoke test passed!"
