## Setup
```bash
# from repo root
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# core deps, TODO make a requirements.txt
pip install torch tqdm tyro transformer-lens datasets

# avoid HF tokenizer thread spam
export TOKENIZERS_PARALLELISM=false
```

## Experiments on Macbook (bebe)
### toy_absorption.py
```bash
python3 toy_absorption.py \
  --run_name toy_mac \
  --out_dir runs \
  --device mps \
  --dtype fp32 \
  --n_latents 1024 \
  --batch_size 512 \
  --num_steps 2000 \
  --lambda_base 1e-3 \
  --p_start 1.0 --p_end 0.5 \
  --fw_alpha 0.5 \
  --fw_warmup_steps 200
```
(Run with `TOY_DEBUG=1` for extra info.)

### train.py:

Baseline (L1 uniform)
```bash
python3 train.py \
  --run_name mac_l1_uniform \
  --out_dir runs \
  --model_name google/gemma-2b-it \
  --device mps \
  --dtype fp16 \
  --layer 12 \
  --seq_len 128 \
  --batch_size 2 \
  --num_steps 2000 \
  --n_latents 4096 \
  --sparsity l1_uniform \
  --lambda_base 1e-3
```

Idea 1 only: p-annealing curriculum (L1 → concave):
```bash
  python3 train.py \
  --run_name mac_p_anneal \
  --out_dir runs \
  --model_name google/gemma-2b-it \
  --device mps \
  --dtype fp16 \
  --layer 12 \
  --seq_len 128 \
  --batch_size 2 \
  --num_steps 2000 \
  --n_latents 4096 \
  --sparsity p_annealing \
  --lambda_base 1e-3 \
  --p_start 1.0 --p_end 0.5
```

Idea 2 only: frequency-weighted L1:
```bash
python3 train.py \
  --run_name mac_freq_l1 \
  --out_dir runs \
  --model_name google/gemma-2b-it \
  --device mps \
  --dtype fp16 \
  --layer 12 \
  --seq_len 128 \
  --batch_size 2 \
  --num_steps 2000 \
  --n_latents 4096 \
  --sparsity l1_freq_weighted \
  --lambda_base 1e-3 \
  --fw_alpha 0.5 \
  --fw_warmup_steps 200
```

Combined: p-annealing + frequency weighting:
```
  python3 train.py \
  --run_name mac_combined \
  --out_dir runs \
  --model_name google/gemma-2b-it \
  --device mps \
  --dtype fp16 \
  --layer 12 \
  --seq_len 128 \
  --batch_size 2 \
  --num_steps 2000 \
  --n_latents 4096 \
  --sparsity p_annealing_freq \
  --lambda_base 1e-3 \
  --p_start 1.0 --p_end 0.5 \
  --fw_alpha 0.5 \
  --fw_warmup_steps 200
```

## Experiments on GPU-backed EC2 (bigger + more informative)

This section is intended for users running on **GPU-backed machines** (e.g. AWS EC2 with NVIDIA GPUs).
The goal here is to move beyond sanity checks and into runs that meaningfully stress feature
splitting, redundancy, and sparsity tradeoffs.

Assumptions:
- `--device cuda`
- `--dtype bf16` (good default on modern NVIDIA GPUs)
- Sufficient VRAM (≈16–24GB for the “moderate” configs below)
- You are comparing *relative behavior* across sparsity objectives at roughly matched L0

---

### 1) toy_absorption.py (larger dictionary)

Use this first to validate that your sparsity variants behave as expected on a controlled
parent/child ground-truth task before spending GPU time on real activations.

```bash
python3 toy_absorption.py \
  --run_name toy_gpu \
  --out_dir runs \
  --device cuda \
  --dtype fp32 \
  --n_latents 8192 \
  --batch_size 2048 \
  --num_steps 5000 \
  --lambda_base 1e-3 \
  --p_start 1.0 --p_end 0.5 \
  --fw_alpha 0.5 \
  --fw_warmup_steps 500
```

### train.py
Baseline (L1 uniform)
```bash
python3 train.py \
  --run_name gpu_l1_uniform \
  --out_dir runs \
  --model_name google/gemma-2b-it \
  --device cuda \
  --dtype bf16 \
  --layer 12 \
  --seq_len 256 \
  --batch_size 8 \
  --num_steps 20000 \
  --n_latents 32768 \
  --sparsity l1_uniform \
  --lambda_base 1e-3
```

Idea 1 only: p-annealing curriculum (L1 → concave):
```bash
python3 train.py \
  --run_name gpu_p_anneal \
  --out_dir runs \
  --model_name google/gemma-2b-it \
  --device cuda \
  --dtype bf16 \
  --layer 12 \
  --seq_len 256 \
  --batch_size 8 \
  --num_steps 20000 \
  --n_latents 32768 \
  --sparsity p_annealing \
  --lambda_base 1e-3 \
  --p_start 1.0 --p_end 0.5
```

Idea 2 only: frequency-weighted L1:
```bash
python3 train.py \
  --run_name gpu_freq_l1 \
  --out_dir runs \
  --model_name google/gemma-2b-it \
  --device cuda \
  --dtype bf16 \
  --layer 12 \
  --seq_len 256 \
  --batch_size 8 \
  --num_steps 20000 \
  --n_latents 32768 \
  --sparsity l1_freq_weighted \
  --lambda_base 1e-3 \
  --fw_alpha 0.5 \
  --fw_warmup_steps 1000
```

Combined: p-annealing + frequency weighting:
```
python3 train.py \
  --run_name gpu_combined \
  --out_dir runs \
  --model_name google/gemma-2b-it \
  --device cuda \
  --dtype bf16 \
  --layer 12 \
  --seq_len 256 \
  --batch_size 8 \
  --num_steps 20000 \
  --n_latents 32768 \
  --sparsity p_annealing_freq \
  --lambda_base 1e-3 \
  --p_start 1.0 --p_end 0.5 \
  --fw_alpha 0.5 \
  --fw_warmup_steps 1000
```
