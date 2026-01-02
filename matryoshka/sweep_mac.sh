#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

SEEDS=(0 1 2 3 4)   # start with 3; bump to (0 1 2 3 4) if it’s fast enough

# -----------------------
# Toy sweep (run_name MUST include seed to avoid overwrite)
# -----------------------
for s in "${SEEDS[@]}"; do
  echo "=== TOY seed=$s ==="
  python3 toy_absorption.py \
    --run_name "toy_mac_s${s}" \
    --out_dir runs \
    --seed "${s}" \
    --device mps \
    --dtype fp32 \
    --n_latents 1024 \
    --batch_size 512 \
    --num_steps 2000 \
    --fw_alpha 0.5 \
    --fw_warmup_steps 200
done

# -----------------------
# Train sweep (keep run_name constant; vary seed)
# -----------------------
for s in "${SEEDS[@]}"; do
  echo "=== TRAIN baseline seed=$s ==="
  python3 train.py \
    --run_name mac_l1_uniform \
    --out_dir runs \
    --seed "${s}" \
    --model_name google/gemma-2b-it \
    --device mps \
    --dtype fp16 \
    --layer 12 \
    --seq_len 128 \
    --batch_size 2 \
    --num_steps 2000 \
    --sparsity l1_uniform \

  echo "=== TRAIN p-anneal seed=$s ==="
  python3 train.py \
    --run_name mac_p_anneal \
    --out_dir runs \
    --seed "${s}" \
    --model_name google/gemma-2b-it \
    --device mps \
    --dtype fp16 \
    --layer 12 \
    --seq_len 128 \
    --batch_size 2 \
    --num_steps 2000 \
    --sparsity p_annealing \
    --p_start 1.0 --p_end 0.5

  echo "=== TRAIN freq-weighted seed=$s ==="
  python3 train.py \
    --run_name mac_freq_l1 \
    --out_dir runs \
    --seed "${s}" \
    --model_name google/gemma-2b-it \
    --device mps \
    --dtype fp16 \
    --layer 12 \
    --seq_len 128 \
    --batch_size 2 \
    --num_steps 2000 \
    --sparsity l1_freq_weighted \
    --fw_alpha 0.5 \
    --fw_warmup_steps 200

  echo "=== TRAIN combined seed=$s ==="
  python3 train.py \
    --run_name mac_combined \
    --out_dir runs \
    --seed "${s}" \
    --model_name google/gemma-2b-it \
    --device mps \
    --dtype fp16 \
    --layer 12 \
    --seq_len 128 \
    --batch_size 2 \
    --num_steps 2000 \
    --sparsity p_annealing_freq \
    --lambda_base 1e-3 \
    --p_start 1.0 --p_end 0.5 \
    --fw_alpha 0.5 \
    --fw_warmup_steps 200
done

  echo "=== TRAIN batchtopk seed=$s ==="
  python3 train.py \
    --run_name mac_batchtopk \
    --out_dir runs \
    --seed "${s}" \
    --model_name google/gemma-2b-it \
    --device mps \
    --dtype fp16 \
    --layer 12 \
    --seq_len 128 \
    --batch_size 2 \
    --num_steps 2000 \
    --n_latents 4096 \
    --sparsity batchtopk \
    --target_l0 40 \ # targeting 40 as that's what we've targeted for the other runs
    --btq_tie_break random