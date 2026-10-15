#!/usr/bin/env bash
# Weekend multi-seed / multi-backbone training campaign (single RTX 4090, sequential).
#
# Trains the `full` silicon-respondent model for:
#   - Qwen2.5-1.5B-Instruct : seeds 1..4   (seed 0 = the existing pre-campaign run)
#   - Llama-3-8B (4-bit)    : seeds 1..3
# All --mode full, 3 epochs, batch 16 (identical hyperparameters to the established run).
# Checkpoints go to checkpoints/<model>/seed<N>/simulator_full.pt (+ _state.pt).
#
# Robustness:
#   * Pre-flight smoke test per backbone (3 steps) — ABORTS the campaign if a backbone
#     can't load/train (e.g. a 4-bit/bitsandbytes problem) before hours are wasted.
#   * Idempotent & resumable: a finished run drops a .done sentinel and is skipped on
#     re-launch; an interrupted run is continued from its *_state.pt.
#
# Launch detached so it survives session clears:
#   nohup ./train_campaign.sh > logs/campaign.log 2>&1 &

set -u  # NOTE: not -e; we handle failures per-job so one bad run can't kill the batch.

PY=/home/nac/anaconda3/envs/llm_train/bin/python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p logs

QWEN="Qwen/Qwen2.5-1.5B-Instruct"
LLAMA="unsloth/llama-3-8b-instruct-bnb-4bit"

QWEN_SEEDS=(1 2 3 4)
# Llama-8B is ~5x slower. Seeds 1-2 trained in the weekend campaign; 3-4 added 2026-06-23
# to turn the cross-backbone result from n=2 (no CI) into n=4 (proper mean ± sd) for the
# BRM full-article submission. Seeds 1-2 carry a .done sentinel and are skipped automatically.
LLAMA_SEEDS=(3 4)

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

# ----------------------------------------------------------------------------- #
# Pre-flight smoke test: 3 optimizer steps on a backbone, in a throwaway dir.
# Returns non-zero if training can't even start.
# ----------------------------------------------------------------------------- #
smoke() {
  local model="$1" tag="$2"
  local dir="/tmp/smoke_${tag}"
  rm -rf "$dir"; mkdir -p "$dir"
  log "SMOKE TEST [$tag] ($model) — 3 steps ..."
  $PY train_simulator.py --model_name "$model" --mode full \
      --epochs 3 --run_epochs 1 --max_steps 3 --val_batches 2 \
      --batch_size 16 --seed 999 --output_dir "$dir" \
      > "logs/smoke_${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "SMOKE TEST [$tag] FAILED (exit $rc). See logs/smoke_${tag}.log"
  else
    log "SMOKE TEST [$tag] OK."
  fi
  return $rc
}

# ----------------------------------------------------------------------------- #
# Train one (model, seed) full run to completion. Idempotent + resumable.
# ----------------------------------------------------------------------------- #
run_one() {
  local model="$1" tag="$2" seed="$3"
  local dir="checkpoints/${tag}/seed${seed}"
  local ckpt="${dir}/simulator_full.pt"
  local state="${dir}/simulator_full_state.pt"
  local logf="logs/train_${tag}_seed${seed}.log"
  mkdir -p "$dir"

  if [ -f "${dir}/.done" ] && [ -f "$ckpt" ]; then
    log "SKIP ${tag} seed${seed} (already .done)."
    return 0
  fi

  local resume_arg=()
  if [ -f "$state" ]; then
    resume_arg=(--resume "$state")
    log "RESUME ${tag} seed${seed} from existing state."
  fi

  log "START ${tag} seed${seed} -> ${dir}"
  local t0=$SECONDS
  $PY train_simulator.py --model_name "$model" --mode full \
      --epochs 3 --run_epochs 0 --batch_size 16 --seed "$seed" \
      --output_dir "$dir" "${resume_arg[@]}" \
      > "$logf" 2>&1
  local rc=$?
  local mins=$(( (SECONDS - t0) / 60 ))
  if [ $rc -eq 0 ] && [ -f "$ckpt" ]; then
    touch "${dir}/.done"
    log "DONE  ${tag} seed${seed} in ${mins} min."
  else
    log "FAIL  ${tag} seed${seed} (exit $rc) after ${mins} min. See ${logf}"
  fi
  return $rc
}

# ----------------------------------------------------------------------------- #
log "==== CAMPAIGN START ===="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null | sed 's/^/[GPU] /'

# Preserve the existing pre-campaign Qwen run as seed 0 (uniform tree for eval).
if [ -f "checkpoints/simulator_full.pt" ] && [ ! -f "checkpoints/qwen/seed0/simulator_full.pt" ]; then
  mkdir -p checkpoints/qwen/seed0
  cp checkpoints/simulator_full.pt checkpoints/qwen/seed0/simulator_full.pt
  [ -f "checkpoints/simulator_full_state.pt" ] && cp checkpoints/simulator_full_state.pt checkpoints/qwen/seed0/simulator_full_state.pt
  touch checkpoints/qwen/seed0/.done
  log "Copied existing Qwen run -> checkpoints/qwen/seed0 (seed 0)."
fi

# Fail-fast smoke tests for BOTH backbones before committing to ~50h.
smoke "$QWEN"  "qwen"  || { log "Aborting: Qwen smoke test failed.";  exit 1; }
smoke "$LLAMA" "llama" || { log "Aborting: Llama smoke test failed."; exit 1; }

# Qwen first (faster — get seeds banked), then Llama.
for s in "${QWEN_SEEDS[@]}";  do run_one "$QWEN"  "qwen"  "$s"; done
for s in "${LLAMA_SEEDS[@]}"; do run_one "$LLAMA" "llama" "$s"; done

log "==== CAMPAIGN END ===="
log "Checkpoints:"
ls -1 checkpoints/qwen/seed*/simulator_full.pt checkpoints/llama/seed*/simulator_full.pt 2>/dev/null | sed 's/^/  /'
