#!/bin/bash
set -e

echo "======================================"
echo "Training Baseline MLP (3 epochs)..."
echo "======================================"
conda run -n llm_train python train.py --model_type mlp --epochs 3

echo "======================================"
echo "Evaluating Baseline MLP..."
echo "======================================"
conda run -n llm_train python evaluate.py --model_type mlp > mlp_eval.txt
cat mlp_eval.txt

echo "======================================"
echo "Training FiLM Model (3 epochs)..."
echo "======================================"
conda run -n llm_train python train.py --model_type film --epochs 3

echo "======================================"
echo "Evaluating FiLM Model..."
echo "======================================"
conda run -n llm_train python evaluate.py --model_type film > film_eval.txt
cat film_eval.txt

echo "Experiments Complete!"
