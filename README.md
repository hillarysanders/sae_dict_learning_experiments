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
Examples:
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

See `sweep_mac.sh` for experimental run configurations.


## Plots
```
python3 plots_for_days.py --train_run_names mac_l1_uniform mac_p_anneal mac_freq_l1 mac_combined
```