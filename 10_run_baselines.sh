#!/bin/bash
# Ablation: does the latent conditioning (and the gate) actually drive the
# psychometric structure? Trains the trainable modes and evaluates all four.
set -e
ENV=llm_train
HUMAN=data/item_holdout.parquet

run () {  # $1 = mode
  MODE=$1
  echo "=========================================="
  echo "MODE: $MODE"
  echo "=========================================="
  if [ "$MODE" = "baseline_c" ] || [ "$MODE" = "full" ]; then
    conda run -n $ENV python train_simulator.py \
      --train_data data/train.parquet --val_data $HUMAN \
      --mode $MODE --batch_size 16 --epochs 3 --lr 2e-4 \
      > train_${MODE}.log 2>&1
    CKPT="--checkpoint checkpoints/simulator_${MODE}.pt"
  else
    CKPT=""   # baseline_a / baseline_b are zero-shot (no trained modules)
  fi
  conda run -n $ENV python silicon_sample.py $CKPT --mode $MODE \
    --items_file $HUMAN --n 500 --output_file silicon_${MODE}.csv
  echo "--- Psychometrics ($MODE) ---"
  conda run -n $ENV python 05_simulator_psychometrics.py \
    --human $HUMAN --silicon silicon_${MODE}.csv
}

run baseline_a
run baseline_b
run baseline_c
run full
