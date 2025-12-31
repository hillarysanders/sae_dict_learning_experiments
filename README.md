
```
python3 train.py \
  --hf_dataset wikitext \
  --hf_dataset_config wikitext-2-raw-v1 \
  --hf_split train \
  --seq_len 128 \
  --batch_size 1 \
  --num_steps 200 \
  --log_every 10
```

```
python3 toy_absorption.py \
  --n_latents 512 \
  --batch_size 256 \
  --num_steps 2000 \
  --lambda_base 1e-3 \
  --method baseline_l1
```